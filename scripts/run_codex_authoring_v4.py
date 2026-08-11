"""Execute the single approved Codex/high v4 authoring session.

The v4 authority state machine makes terminal evidence exist *before* the
authorization claim can commit.  Approval creates a conditional, fail-closed
terminal guard at a frozen path.  A valid claim activates that guard; only a
fully validated canonical output bundle may supersede it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import threading
import time
from typing import Any

import scripts.run_codex_authoring as v2_runner
import scripts.run_codex_authoring_v3 as v3_runner
import skillchain.codex_authoring as base_contract_module
import skillchain.codex_authoring_v3 as v3_contract_module
import skillchain.codex_authoring_v4 as contract_module
from skillchain.codex_authoring_v4 import (
    CODEX_AUTHOR_CANDIDATE_ID,
    CODEX_AUTHOR_RUN_ID,
    CODEX_COMMAND_SHAPE,
    CodexAuthoringContractError,
    CreateOnceConflictError,
    build_codex_runtime_dependency_snapshot_v4,
    classified_atomic_create,
    construct_codex_v4_environment,
    load_codex_authoring_input,
    normalize_codex_authoring_output,
    parse_codex_authoring_request,
    render_codex_authoring_stdin,
    resolve_frozen_codex_binary,
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
    read_stable_regular_file,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
V3_RUNNER_PATH = Path(v3_runner.__file__).resolve(strict=True)
BASE_RUNNER_PATH = Path(v2_runner.__file__).resolve(strict=True)
CONTRACT_MODULE_PATH = Path(contract_module.__file__).resolve(strict=True)
V3_CONTRACT_MODULE_PATH = Path(v3_contract_module.__file__).resolve(strict=True)
BASE_CONTRACT_MODULE_PATH = Path(base_contract_module.__file__).resolve(strict=True)
FREEZE_PATH = (
    ROOT / "specs" / "authoring" / "authoring-freeze-lock-codex-high-v4.json"
)
V2_RUN_ID = "llm-static-codex-primary-20260724-high-v2"
V2_RETIREMENT_CLAIM_PATH = (
    ROOT / "specs" / "authoring" / f"{V2_RUN_ID}-attempt-claim.json"
)
V2_OUTPUT_DIR = ROOT / "runs" / "formal-authoring" / V2_RUN_ID
V2_RECEIPT_PATH = V2_OUTPUT_DIR / "invocation-receipt.json"


@dataclass(frozen=True)
class V4CappedProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_limit_exceeded: bool
    stderr_limit_exceeded: bool
    stdin_completed: bool
    supervision_error: str | None
    process_created: bool
    process_reaped: bool
    pipe_threads_terminated: bool


def _safe_error_text(error: BaseException) -> tuple[str, str]:
    try:
        kind = type(error).__name__
    except BaseException:
        kind = "BaseException"
    try:
        message = str(error)
    except BaseException:
        message = "<error stringification failed>"
    return kind, message


def claim_commit_allows_process_launch(commit: Any) -> bool:
    return (
        commit.committed is True
        and commit.recovered_after_exception is False
        and commit.cleanup_error is None
    )


def terminal_process_state_is_stable(state: dict[str, Any]) -> bool:
    return bool(
        (
            state.get("process_launch_attempted") is False
            and state.get("process_launched") is False
        )
        or (
            state.get("process_launched") is True
            and state.get("process_reaped") is True
            and state.get("pipe_threads_terminated") is True
            and state.get("process_result_committed") is True
        )
    )


def _run_capped_process_v4(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdin: bytes,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
    spawned: threading.Event,
) -> V4CappedProcessResult:
    """Supervise one process with a BaseException-safe Popen-to-reap boundary."""

    deadline = time.monotonic() + timeout_seconds
    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    stdin_completed = threading.Event()
    supervision_failed = threading.Event()
    supervision_errors: list[str] = []
    supervision_error_lock = threading.Lock()
    process: subprocess.Popen[bytes] | None = None
    process_created = False
    process_reaped = False
    timed_out = False
    returncode: int | None = None
    threads: dict[str, threading.Thread] = {}

    def record(stage: str, error: BaseException) -> None:
        kind, message = _safe_error_text(error)
        try:
            with supervision_error_lock:
                supervision_errors.append(f"{stage}: {kind}: {message}")
            supervision_failed.set()
        except BaseException:
            # The caller still receives a fail-closed generic supervision error.
            pass

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
        except BaseException as error:
            record(stage, error)

    def feed_stdin(stream: Any) -> None:
        try:
            view = memoryview(stdin)
            written = 0
            while written < len(view):
                count = stream.write(view[written:])
                if count is None or count <= 0:
                    raise OSError(
                        f"short stdin write stopped after {written} bytes"
                    )
                written += count
            stream.flush()
            if written != len(stdin):
                raise OSError(
                    f"short stdin write: expected {len(stdin)}, wrote {written}"
                )
            stdin_completed.set()
        except BaseException as error:
            record("stdin", error)
        finally:
            try:
                stream.close()
            except BaseException as error:
                record("stdin_close", error)

    fatal_error: BaseException | None = None
    try:
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
        process_created = True
        spawned.set()
        if process.stdout is None or process.stderr is None or process.stdin is None:
            raise CodexAuthoringContractError("Codex process pipes are unavailable")
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
                args=(process.stdin,),
                daemon=True,
            ),
        }
        for name in ("stdout", "stderr", "stdin"):
            threads[name].start()
        while process.poll() is None:
            if (
                stdout_overflow.is_set()
                or stderr_overflow.is_set()
                or supervision_failed.is_set()
            ):
                process.kill()
                break
            if time.monotonic() >= deadline:
                timed_out = True
                process.kill()
                break
            time.sleep(0.02)
        returncode = process.wait(timeout=5)
    except BaseException as error:
        fatal_error = error
        record("process_supervision", error)
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
            except BaseException as error:
                record("process_kill", error)
            try:
                returncode = process.wait(timeout=5)
                process_reaped = True
            except BaseException as error:
                record("process_final_wait", error)
            for thread in threads.values():
                try:
                    if thread.ident is not None:
                        thread.join(timeout=5)
                except BaseException as error:
                    record("thread_join", error)
            for name, stream in (
                ("stdin", process.stdin),
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                if stream is None:
                    continue
                thread = threads.get(name)
                try:
                    if thread is None or not thread.is_alive():
                        stream.close()
                except BaseException as error:
                    record(f"{name}_final_close", error)

    live_threads: list[str] = []
    for name, thread in threads.items():
        try:
            if thread.is_alive():
                live_threads.append(name)
        except BaseException as error:
            record("thread_liveness", error)
            live_threads.append(name)
    if live_threads:
        supervision_errors.append(
            "pipe threads did not terminate: " + ", ".join(live_threads)
        )
    if fatal_error is not None and not supervision_errors:
        kind, message = _safe_error_text(fatal_error)
        supervision_errors.append(f"process_supervision: {kind}: {message}")
    if process_created and not process_reaped:
        supervision_errors.append("process handle was not confirmed reaped")
    if returncode is None:
        try:
            returncode = (
                process.returncode
                if process is not None and process.returncode is not None
                else -1
            )
        except BaseException:
            returncode = -1
    return V4CappedProcessResult(
        returncode=returncode,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
        timed_out=timed_out,
        stdout_limit_exceeded=stdout_overflow.is_set(),
        stderr_limit_exceeded=stderr_overflow.is_set(),
        stdin_completed=stdin_completed.is_set(),
        supervision_error=(
            "; ".join(supervision_errors) if supervision_errors else None
        ),
        process_created=process_created,
        process_reaped=process_reaped,
        pipe_threads_terminated=not live_threads,
    )


def _validate_frozen_source_bindings(
    freeze_bindings: dict[str, tuple[Path, str]],
    runtime: dict[str, Any],
) -> None:
    expected = {
        "runner": (RUNNER_PATH, freeze_bindings["runner"][1]),
        "v3_runner": (V3_RUNNER_PATH, freeze_bindings["v3_runner"][1]),
        "base_runner": (BASE_RUNNER_PATH, freeze_bindings["base_runner"][1]),
        "contract_module": (
            CONTRACT_MODULE_PATH,
            freeze_bindings["contract_module"][1],
        ),
        "v3_contract_module": (
            V3_CONTRACT_MODULE_PATH,
            freeze_bindings["v3_contract_module"][1],
        ),
        "base_contract_module": (
            BASE_CONTRACT_MODULE_PATH,
            freeze_bindings["base_contract_module"][1],
        ),
        "source_manifest": freeze_bindings["source_manifest"],
        "prompt_isolation_evidence": freeze_bindings[
            "prompt_isolation_evidence"
        ],
        "binary_evidence": freeze_bindings["binary_binding_evidence"],
    }
    for runtime_name, binding in expected.items():
        if v2_runner._runtime_binding(runtime, runtime_name) != binding:
            raise CodexAuthoringContractError(
                f"Codex v4 runtime {runtime_name} binding differs from the freeze"
            )


def _validate_dependency_snapshot(manifest: dict[str, Any]) -> None:
    v2_runner._self_hash(
        manifest,
        "source_manifest_payload_sha256",
        "Codex v4 source manifest",
    )
    unsigned = dict(manifest)
    unsigned.pop("source_manifest_payload_sha256", None)
    live = build_codex_runtime_dependency_snapshot_v4(ROOT)
    if canonical_json_bytes(unsigned) != canonical_json_bytes(live):
        raise CodexAuthoringContractError(
            "Codex v4 source or runtime dependency snapshot drifted"
        )


def _terminal_guard(
    guard_path: Path,
    expected_file_sha256: str,
    *,
    claim_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    guard = v2_runner._load_object(
        guard_path,
        expected_file_sha256,
        "Codex v4 conditional terminal guard",
    )
    v2_runner._self_hash(
        guard,
        "terminal_guard_payload_sha256",
        "Codex v4 conditional terminal guard",
    )
    required = {
        "schema_version": 1,
        "artifact_kind": "codex_authoring_conditional_terminal_guard",
        "status": "inactive_until_exact_claim_commits",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "activation_claim_file": claim_path.relative_to(ROOT).as_posix(),
        "canonical_output_directory": output_dir.relative_to(ROOT).as_posix(),
        "canonical_receipt_file": receipt_path.relative_to(ROOT).as_posix(),
        "active_status": "authorization_consumed_result_rejected",
        "active_formal_codex_session_eligible": False,
        "active_process_launch_state": "unknown",
        "activation_requires_claim_to_bind_this_guard_sha256": True,
        "canonical_override_requires_full_bundle_validation": True,
        "model_retry_fallback_repair_followup_authorized": False,
        "storage_disaster_guarantee": "not_claimed",
    }
    if set(guard) != set(required) | {"terminal_guard_payload_sha256"}:
        raise CodexAuthoringContractError(
            "Codex v4 conditional terminal guard fields drifted"
        )
    for key, expected in required.items():
        if guard.get(key) != expected:
            raise CodexAuthoringContractError(
                f"Codex v4 conditional terminal guard drifted: {key}"
            )
    return guard


def _approval(
    approval_path: Path,
    approval_file_sha256: str,
    freeze: dict[str, Any],
    freeze_file_sha256: str,
    *,
    guard_path: Path,
    guard_file_sha256: str,
    claim_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    raw = v2_runner._load_object(
        approval_path,
        approval_file_sha256,
        "Codex v4 owner approval",
    )
    v2_runner._self_hash(
        raw,
        "approval_payload_sha256",
        "Codex v4 owner approval",
    )
    guard = _terminal_guard(
        guard_path,
        guard_file_sha256,
        claim_path=claim_path,
        output_dir=output_dir,
        receipt_path=receipt_path,
    )
    retirement = v2_runner._load_object(
        V2_RETIREMENT_CLAIM_PATH,
        raw.get("v2_retirement_claim_file_sha256"),
        "Codex v2 non-inference retirement claim",
    )
    v2_runner._self_hash(
        retirement,
        "retirement_payload_sha256",
        "Codex v2 non-inference retirement claim",
    )
    v2_incident_path, v2_incident_file_sha256 = v2_runner._binding(
        freeze,
        "v2_preflight_incident",
    )
    v2_incident = v2_runner._load_object(
        v2_incident_path,
        v2_incident_file_sha256,
        "Codex v2 preflight incident",
    )
    v2_runner._self_hash(
        v2_incident,
        "incident_payload_sha256",
        "Codex v2 preflight incident",
    )
    v3_incident_path, v3_incident_file_sha256 = v2_runner._binding(
        freeze,
        "v3_preapproval_audit_incident",
    )
    v3_incident = v2_runner._load_object(
        v3_incident_path,
        v3_incident_file_sha256,
        "Codex v3 preapproval audit incident",
    )
    v2_runner._self_hash(
        v3_incident,
        "incident_payload_sha256",
        "Codex v3 preapproval audit incident",
    )
    required = {
        "schema_version": 1,
        "status": "owner_approved_for_one_codex_author_session",
        "approved_by": "project-owner",
        "freeze_lock_file": FREEZE_PATH.relative_to(ROOT).as_posix(),
        "freeze_lock_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": freeze.get("freeze_payload_sha256"),
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
        "v2_retirement_claim_file": (
            V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
        ),
        "v2_retirement_claim_file_sha256": sha256_bytes(
            read_stable_regular_file(
                V2_RETIREMENT_CLAIM_PATH,
                label="Codex v2 non-inference retirement claim",
            )
        ),
        "v2_retirement_claim_payload_sha256": retirement.get(
            "retirement_payload_sha256"
        ),
        "v2_preflight_incident_file_sha256": v2_incident_file_sha256,
        "v2_preflight_incident_payload_sha256": v2_incident.get(
            "incident_payload_sha256"
        ),
        "v3_preapproval_audit_incident_file_sha256": v3_incident_file_sha256,
        "v3_preapproval_audit_incident_payload_sha256": v3_incident.get(
            "incident_payload_sha256"
        ),
        "terminal_guard_file": guard_path.relative_to(ROOT).as_posix(),
        "terminal_guard_file_sha256": guard_file_sha256,
        "terminal_guard_payload_sha256": guard.get(
            "terminal_guard_payload_sha256"
        ),
    }
    if set(raw) != set(required) | {"approval_payload_sha256"}:
        raise CodexAuthoringContractError(
            "Codex v4 owner approval contains missing or unexpected fields"
        )
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise CodexAuthoringContractError(
                f"Codex v4 owner approval does not accept frozen field: {key}"
            )
    if (
        retirement.get("status")
        != "authorization_retired_without_codex_process_launch"
        or retirement.get("run_id") != V2_RUN_ID
        or retirement.get("successor_candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or retirement.get("successor_run_id") != CODEX_AUTHOR_RUN_ID
        or retirement.get("codex_exec_process_spawned") is not False
        or retirement.get("inference_requested") is not False
        or retirement.get("v2_authorization_reusable") is not False
        or retirement.get("preflight_incident_file_sha256")
        != v2_incident_file_sha256
        or retirement.get("preflight_incident_payload_sha256")
        != v2_incident.get("incident_payload_sha256")
        or v2_incident.get("claim_created") is not False
        or v2_incident.get("codex_exec_process_spawned") is not False
        or v2_incident.get("inference_requested") is not False
        or v3_incident.get("owner_approval_created") is not False
        or v3_incident.get("claim_created") is not False
        or v3_incident.get("inference_requested") is not False
        or os.path.lexists(V2_OUTPUT_DIR)
        or os.path.lexists(V2_RECEIPT_PATH)
    ):
        raise CodexAuthoringContractError(
            "Codex v2 authorization was not safely retired for v4"
        )
    return raw


def _binary_evidence(
    path: Path,
    expected_sha256: str,
    runtime: dict[str, Any],
) -> None:
    evidence = v2_runner._load_object(
        path,
        expected_sha256,
        "Codex v4 absolute-binary evidence",
    )
    v2_runner._self_hash(
        evidence,
        "binary_evidence_payload_sha256",
        "Codex v4 absolute-binary evidence",
    )
    if (
        evidence.get("status") != "verified_without_inference"
        or evidence.get("inference_request_performed") is not False
        or evidence.get("binary") != runtime.get("binary")
    ):
        raise CodexAuthoringContractError(
            "Codex v4 absolute-binary evidence differs from runtime"
        )


def _real_directory(path: Path, label: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CodexAuthoringContractError(
            f"{label} must be a real directory: {path}"
        )


CLAIM_FIELDS = {
    "schema_version",
    "status",
    "candidate_id",
    "run_id",
    "claimant_nonce",
    "freeze_lock_file_sha256",
    "freeze_payload_sha256",
    "owner_approval_file",
    "owner_approval_file_sha256",
    "owner_approval_payload_sha256",
    "terminal_guard_file",
    "terminal_guard_file_sha256",
    "v2_retirement_claim_file_sha256",
    "authoring_input_file_sha256",
    "canonical_request_file_sha256",
    "stdin_request_file_sha256",
    "output_schema_file_sha256",
    "materialized_output_schema_sha256",
    "runtime_lock_file_sha256",
    "runtime_payload_sha256",
    "source_manifest_file_sha256",
    "binary_evidence_file_sha256",
    "prompt_isolation_evidence_file_sha256",
    "environment_policy_sha256",
    "environment_instance_sha256",
    "command_sha256",
    "requested_model",
    "reasoning_effort",
    "max_codex_exec_sessions",
    "retry_fallback_repair_followup_policy",
    "required_output_directory",
    "required_receipt_file",
    "claim_payload_sha256",
}


def _local_freeze_binding(
    freeze: dict[str, Any],
    name: str,
) -> tuple[Path, str]:
    bindings = freeze.get("bindings")
    item = bindings.get(name) if isinstance(bindings, dict) else None
    if not isinstance(item, dict) or set(item) != {"file", "file_sha256"}:
        raise CodexAuthoringContractError(
            f"Codex v4 terminal validation binding is invalid: {name}"
        )
    relative = item.get("file")
    digest = item.get("file_sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise CodexAuthoringContractError(
            f"Codex v4 terminal validation binding values are invalid: {name}"
        )
    candidate = ROOT / relative
    resolved = candidate.resolve(strict=True)
    if candidate.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise CodexAuthoringContractError(
            f"Codex v4 terminal validation binding is unsafe: {name}"
        )
    content = read_stable_regular_file(
        resolved,
        label=f"Codex v4 terminal validation binding {name}",
    )
    if sha256_bytes(content) != digest:
        raise CodexAuthoringContractError(
            f"Codex v4 terminal validation binding drifted: {name}"
        )
    return resolved, digest


def _load_terminal_validation_context(
    *,
    guard_path: Path,
    guard_file_sha256: str,
    claim_path: Path,
    output_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Any,
    Any,
]:
    """Validate the durable guard→claim→freeze→approval authority chain."""

    receipt_path = output_dir / "invocation-receipt.json"
    _terminal_guard(
        guard_path,
        guard_file_sha256,
        claim_path=claim_path,
        output_dir=output_dir,
        receipt_path=receipt_path,
    )
    claim_bytes = read_stable_regular_file(
        claim_path,
        label="Codex v4 authorization claim",
    )
    claim = parse_canonical_json(
        claim_bytes,
        label="Codex v4 authorization claim",
    )
    if not isinstance(claim, dict) or set(claim) != CLAIM_FIELDS:
        raise CodexAuthoringContractError(
            "Codex v4 authorization claim fields drifted"
        )
    unsigned_claim = dict(claim)
    observed_claim_sha = unsigned_claim.pop("claim_payload_sha256", None)
    nonce = claim.get("claimant_nonce")
    digest_fields = (
        "freeze_lock_file_sha256",
        "freeze_payload_sha256",
        "owner_approval_file_sha256",
        "owner_approval_payload_sha256",
        "terminal_guard_file_sha256",
        "v2_retirement_claim_file_sha256",
        "authoring_input_file_sha256",
        "canonical_request_file_sha256",
        "stdin_request_file_sha256",
        "output_schema_file_sha256",
        "materialized_output_schema_sha256",
        "runtime_lock_file_sha256",
        "runtime_payload_sha256",
        "source_manifest_file_sha256",
        "binary_evidence_file_sha256",
        "prompt_isolation_evidence_file_sha256",
        "environment_policy_sha256",
        "environment_instance_sha256",
        "command_sha256",
    )
    if (
        observed_claim_sha != sha256_bytes(canonical_json_bytes(unsigned_claim))
        or claim.get("schema_version") != 3
        or claim.get("status") != "consumed_before_codex_process_launch"
        or claim.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or claim.get("run_id") != CODEX_AUTHOR_RUN_ID
        or not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
        or any(
            not isinstance(claim.get(field), str)
            or len(claim[field]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in claim[field]
            )
            for field in digest_fields
        )
        or claim.get("terminal_guard_file")
        != guard_path.relative_to(ROOT).as_posix()
        or claim.get("terminal_guard_file_sha256") != guard_file_sha256
        or claim.get("requested_model") != "gpt-5.6-sol"
        or claim.get("reasoning_effort") != "high"
        or claim.get("max_codex_exec_sessions") != 1
        or claim.get("retry_fallback_repair_followup_policy") != "forbidden"
        or claim.get("required_output_directory")
        != output_dir.relative_to(ROOT).as_posix()
        or claim.get("required_receipt_file")
        != receipt_path.relative_to(ROOT).as_posix()
        or claim.get("materialized_output_schema_sha256")
        != claim.get("output_schema_file_sha256")
    ):
        raise CodexAuthoringContractError(
            "Codex v4 authorization claim semantics drifted"
        )

    freeze = v2_runner._load_object(
        FREEZE_PATH,
        claim["freeze_lock_file_sha256"],
        "Codex v4 freeze lock",
    )
    v2_runner._self_hash(
        freeze,
        "freeze_payload_sha256",
        "Codex v4 freeze lock",
    )
    authorization = freeze.get("authorization")
    paths = authorization.get("paths") if isinstance(authorization, dict) else None
    terminal_guard_policy = freeze.get("terminal_guard")
    if (
        freeze.get("freeze_payload_sha256") != claim["freeze_payload_sha256"]
        or freeze.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or not isinstance(paths, dict)
        or paths.get("run_id") != CODEX_AUTHOR_RUN_ID
        or paths.get("claim_file") != claim_path.relative_to(ROOT).as_posix()
        or paths.get("output_directory")
        != output_dir.relative_to(ROOT).as_posix()
        or paths.get("receipt_file")
        != receipt_path.relative_to(ROOT).as_posix()
        or paths.get("terminal_guard_file")
        != guard_path.relative_to(ROOT).as_posix()
        or not isinstance(terminal_guard_policy, dict)
        or terminal_guard_policy.get("expected_file_sha256")
        != guard_file_sha256
    ):
        raise CodexAuthoringContractError(
            "Codex v4 freeze differs from the terminal authority chain"
        )

    approval_reference = claim.get("owner_approval_file")
    if (
        not isinstance(approval_reference, str)
        or Path(approval_reference).is_absolute()
        or ".." in Path(approval_reference).parts
    ):
        raise CodexAuthoringContractError(
            "Codex v4 owner approval reference is unsafe"
        )
    approval_path = (ROOT / approval_reference).resolve(strict=True)
    if (
        ROOT.resolve() not in approval_path.parents
        or paths.get("approval_record_file") != approval_reference
    ):
        raise CodexAuthoringContractError(
            "Codex v4 owner approval reference differs from the freeze"
        )
    approval = v2_runner._load_object(
        approval_path,
        claim["owner_approval_file_sha256"],
        "Codex v4 owner approval",
    )
    v2_runner._self_hash(
        approval,
        "approval_payload_sha256",
        "Codex v4 owner approval",
    )
    if (
        approval.get("approval_payload_sha256")
        != claim["owner_approval_payload_sha256"]
        or approval.get("status")
        != "owner_approved_for_one_codex_author_session"
        or approval.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or approval.get("run_id") != CODEX_AUTHOR_RUN_ID
        or approval.get("freeze_lock_file_sha256")
        != claim["freeze_lock_file_sha256"]
        or approval.get("freeze_payload_sha256")
        != claim["freeze_payload_sha256"]
        or approval.get("terminal_guard_file_sha256") != guard_file_sha256
        or approval.get("requested_model") != "gpt-5.6-sol"
        or approval.get("reasoning_effort") != "high"
    ):
        raise CodexAuthoringContractError(
            "Codex v4 owner approval differs from the terminal authority chain"
        )

    packet_path, packet_sha256 = _local_freeze_binding(
        freeze,
        "authoring_input",
    )
    semantic_path, semantic_sha256 = _local_freeze_binding(
        freeze,
        "semantic_source",
    )
    claim_binding_map = {
        "canonical_request": "canonical_request_file_sha256",
        "stdin_request": "stdin_request_file_sha256",
        "output_schema": "output_schema_file_sha256",
        "runtime_lock": "runtime_lock_file_sha256",
        "source_manifest": "source_manifest_file_sha256",
        "binary_binding_evidence": "binary_evidence_file_sha256",
        "prompt_isolation_evidence": "prompt_isolation_evidence_file_sha256",
    }
    bound_paths: dict[str, Path] = {}
    for binding_name, claim_field in claim_binding_map.items():
        bound_path, bound_digest = _local_freeze_binding(freeze, binding_name)
        if claim.get(claim_field) != bound_digest:
            raise CodexAuthoringContractError(
                f"Codex v4 claim differs from freeze binding: {binding_name}"
            )
        bound_paths[binding_name] = bound_path
    if (
        packet_sha256 != claim["authoring_input_file_sha256"]
        or claim["v2_retirement_claim_file_sha256"]
        != approval.get("v2_retirement_claim_file_sha256")
    ):
        raise CodexAuthoringContractError(
            "Codex v4 claim input or retirement binding drifted"
        )
    authoring_input = load_codex_authoring_input(
        packet_path,
        expected_file_sha256=packet_sha256,
    )
    semantic_source = load_authoring_packet(
        semantic_path,
        expected_file_sha256=semantic_sha256,
    )
    request_bytes = read_stable_regular_file(
        bound_paths["canonical_request"],
        label="Codex v4 frozen canonical request",
    )
    stdin_bytes = read_stable_regular_file(
        bound_paths["stdin_request"],
        label="Codex v4 frozen stdin request",
    )
    schema_bytes = read_stable_regular_file(
        bound_paths["output_schema"],
        label="Codex v4 frozen output schema",
    )
    request = parse_codex_authoring_request(request_bytes)
    if (
        request.authoring_input != authoring_input
        or render_codex_authoring_stdin(request) != stdin_bytes
        or request.output_contract.json_schema_canonical_json.encode("utf-8")
        != schema_bytes
        or claim.get("command_sha256")
        != sha256_bytes(canonical_json_bytes(list(CODEX_COMMAND_SHAPE)))
    ):
        raise CodexAuthoringContractError(
            "Codex v4 request, stdin, schema, or command shape drifted"
        )
    runtime = v2_runner._load_object(
        bound_paths["runtime_lock"],
        claim["runtime_lock_file_sha256"],
        "Codex v4 runtime lock",
    )
    v2_runner._self_hash(
        runtime,
        "runtime_payload_sha256",
        "Codex v4 runtime lock",
    )
    environment_policy = runtime.get("environment_policy")
    if (
        runtime.get("runtime_payload_sha256") != claim["runtime_payload_sha256"]
        or not isinstance(environment_policy, dict)
        or sha256_bytes(canonical_json_bytes(environment_policy))
        != claim["environment_policy_sha256"]
        or runtime.get("command_shape") != list(CODEX_COMMAND_SHAPE)
    ):
        raise CodexAuthoringContractError(
            "Codex v4 runtime differs from the claim"
        )
    source_manifest = v2_runner._load_object(
        bound_paths["source_manifest"],
        claim["source_manifest_file_sha256"],
        "Codex v4 source manifest",
    )
    _validate_dependency_snapshot(source_manifest)
    _binary_evidence(
        bound_paths["binary_binding_evidence"],
        claim["binary_evidence_file_sha256"],
        runtime,
    )
    return claim, freeze, approval, authoring_input, semantic_source


def _validate_published_bundle(
    output_dir: Path,
    expected: dict[str, bytes],
) -> bool:
    """Validate an exact canonical bundle after a possibly ambiguous rename."""
    try:
        _real_directory(output_dir, "Codex v4 canonical output directory")
        children = list(output_dir.iterdir())
        if {child.name for child in children} != set(expected):
            return False
        for child in children:
            if child.is_symlink() or not child.is_file():
                return False
            observed = read_stable_regular_file(
                child,
                label=f"Codex v4 canonical bundle file {child.name}",
            )
            if observed != expected[child.name]:
                return False
        receipt = parse_canonical_json(
            expected["invocation-receipt.json"],
            label="Codex v4 canonical receipt",
        )
        if not isinstance(receipt, dict):
            return False
        unsigned = dict(receipt)
        observed_payload_sha = unsigned.pop("receipt_payload_sha256", None)
        return (
            observed_payload_sha == sha256_bytes(canonical_json_bytes(unsigned))
            and receipt.get("candidate_id") == CODEX_AUTHOR_CANDIDATE_ID
            and receipt.get("run_id") == CODEX_AUTHOR_RUN_ID
            and receipt.get("canonical_bundle_complete") is True
        )
    except BaseException:
        return False


def _publish_and_classify_bundle(
    staging: Path,
    output_dir: Path,
    expected: dict[str, bytes],
    *,
    claim_path: Path,
    guard_path: Path,
    guard_file_sha256: str,
) -> None:
    """Publish once and resolve an ambiguous rename by exact disk validation."""
    publish_error: BaseException | None = None
    try:
        atomic_publish_new_directory(staging, output_dir)
    except BaseException as error:
        publish_error = error
    canonical_valid = _validate_published_bundle(
        output_dir,
        expected,
    ) and validate_canonical_bundle_from_disk(
        output_dir,
        claim_path=claim_path,
        guard_path=guard_path,
        guard_file_sha256=guard_file_sha256,
    )
    if canonical_valid:
        shutil.rmtree(staging, ignore_errors=True)
        return
    shutil.rmtree(staging, ignore_errors=True)
    if publish_error is not None:
        raise publish_error
    raise CodexAuthoringContractError(
        "Codex v4 canonical bundle publication could not be validated"
    )


def validate_canonical_bundle_from_disk(
    output_dir: Path,
    *,
    claim_path: Path,
    guard_path: Path,
    guard_file_sha256: str,
) -> bool:
    """Independently validate a canonical bundle using its signed receipt."""
    try:
        _real_directory(output_dir, "Codex v4 canonical output directory")
        receipt_path = output_dir / "invocation-receipt.json"
        receipt_bytes = read_stable_regular_file(
            receipt_path,
            label="Codex v4 canonical terminal receipt",
        )
        receipt = parse_canonical_json(
            receipt_bytes,
            label="Codex v4 canonical terminal receipt",
        )
        if not isinstance(receipt, dict):
            return False
        unsigned = dict(receipt)
        observed_payload_sha = unsigned.pop("receipt_payload_sha256", None)
        files = receipt.get("canonical_bundle_files")
        required_files = {
            "codex-events.jsonl",
            "codex-stderr.bin",
            "authoring-request.json",
            "authoring-stdin.txt",
            "authoring-output-schema.json",
            "session-evidence.json",
        }
        optional_files = {
            "author-content.raw.json",
            "pre-review-draft.json",
        }
        if (
            observed_payload_sha != sha256_bytes(canonical_json_bytes(unsigned))
            or receipt.get("artifact_kind")
            != "codex_authoring_canonical_terminal_receipt"
            or receipt.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
            or receipt.get("run_id") != CODEX_AUTHOR_RUN_ID
            or receipt.get("canonical_bundle_complete") is not True
            or receipt.get("terminal_guard_overridden_by_this_valid_bundle")
            is not True
            or receipt.get("terminal_guard_file")
            != guard_path.relative_to(ROOT).as_posix()
            or receipt.get("terminal_guard_file_sha256") != guard_file_sha256
            or not isinstance(files, dict)
            or any(
                not isinstance(name, str) or not isinstance(digest, str)
                for name, digest in files.items()
            )
            or not required_files.issubset(files)
            or set(files) - required_files - optional_files
        ):
            return False
        (
            validated_claim,
            _validated_freeze,
            _validated_approval,
            authoring_input,
            semantic_source,
        ) = _load_terminal_validation_context(
            guard_path=guard_path,
            guard_file_sha256=guard_file_sha256,
            claim_path=claim_path,
            output_dir=output_dir,
        )
        claim_bytes = read_stable_regular_file(
            claim_path,
            label="Codex v4 authorization claim",
        )
        claim = parse_canonical_json(
            claim_bytes,
            label="Codex v4 authorization claim",
        )
        if not isinstance(claim, dict):
            return False
        unsigned_claim = dict(claim)
        claim_payload_sha = unsigned_claim.pop("claim_payload_sha256", None)
        if (
            claim_payload_sha != sha256_bytes(canonical_json_bytes(unsigned_claim))
            or receipt.get("attempt_claim_file_sha256") != sha256_bytes(claim_bytes)
            or claim.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
            or claim.get("run_id") != CODEX_AUTHOR_RUN_ID
            or claim.get("terminal_guard_file")
            != guard_path.relative_to(ROOT).as_posix()
            or claim.get("terminal_guard_file_sha256") != guard_file_sha256
            or claim != validated_claim
        ):
            return False
        expected_names = set(files) | {"invocation-receipt.json"}
        children = list(output_dir.iterdir())
        if {child.name for child in children} != expected_names:
            return False
        for name, digest in files.items():
            content = read_stable_regular_file(
                output_dir / name,
                label=f"Codex v4 canonical bundle file {name}",
            )
            if sha256_bytes(content) != digest:
                return False
        evidence_bytes = read_stable_regular_file(
            output_dir / "session-evidence.json",
            label="Codex v4 canonical session evidence",
        )
        evidence = parse_canonical_json(
            evidence_bytes,
            label="Codex v4 canonical session evidence",
        )
        if not isinstance(evidence, dict):
            return False
        if any(receipt.get(key) != value for key, value in evidence.items()):
            return False
        status = evidence.get("status")
        formal_eligible = evidence.get("formal_codex_session_eligible")
        allowed_statuses = {
            "claim_consumed_prelaunch_cleanup_failed",
            "codex_process_timed_out",
            "codex_process_output_limit_exceeded",
            "codex_process_nonzero_exit",
            "codex_process_supervision_failed",
            "codex_session_event_audit_rejected",
            "codex_session_incomplete",
            "codex_session_completed_draft_rejected",
            "codex_session_completed_draft_ready_for_review",
        }
        if status not in allowed_statuses:
            return False
        if (
            evidence.get("schema_version") != 4
            or evidence.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
            or evidence.get("run_id") != CODEX_AUTHOR_RUN_ID
            or evidence.get("authorization_consumed") is not True
            or evidence.get("requested_model") != "gpt-5.6-sol"
            or evidence.get("reasoning_effort") != "high"
            or evidence.get("repository_retry_performed") is not False
            or evidence.get("followup_performed") is not False
            or evidence.get("repair_performed") is not False
            or evidence.get("fallback_performed") is not False
            or not (
                (
                    evidence.get("process_launch_attempted") is False
                    and evidence.get("process_launched") is False
                )
                or (
                    evidence.get("process_launched") is True
                    and evidence.get("process_reaped") is True
                    and evidence.get("pipe_threads_terminated") is True
                    and evidence.get("process_result_committed") is True
                )
            )
            or evidence.get("attempt_claim_file_sha256")
            != sha256_bytes(claim_bytes)
            or evidence.get("attempt_claim_payload_sha256")
            != claim_payload_sha
            or evidence.get("terminal_guard_file_sha256")
            != guard_file_sha256
            or evidence.get("event_log_sha256")
            != files["codex-events.jsonl"]
            or evidence.get("stderr_sha256") != files["codex-stderr.bin"]
            or evidence.get("canonical_request_file_sha256")
            != files["authoring-request.json"]
            or evidence.get("stdin_request_file_sha256")
            != files["authoring-stdin.txt"]
            or evidence.get("output_schema_file_sha256")
            != files["authoring-output-schema.json"]
        ):
            return False
        claim_to_evidence_fields = {
            "freeze_lock_file_sha256": "freeze_lock_file_sha256",
            "freeze_payload_sha256": "freeze_payload_sha256",
            "owner_approval_file_sha256": "owner_approval_file_sha256",
            "owner_approval_payload_sha256": "owner_approval_payload_sha256",
            "authoring_input_file_sha256": "authoring_input_file_sha256",
            "canonical_request_file_sha256": "canonical_request_file_sha256",
            "stdin_request_file_sha256": "stdin_request_file_sha256",
            "output_schema_file_sha256": "output_schema_file_sha256",
            "runtime_lock_file_sha256": "runtime_lock_file_sha256",
            "runtime_payload_sha256": "runtime_payload_sha256",
            "source_manifest_file_sha256": "source_manifest_file_sha256",
            "binary_evidence_file_sha256": "binary_evidence_file_sha256",
            "prompt_isolation_evidence_file_sha256": (
                "prompt_isolation_evidence_file_sha256"
            ),
            "environment_policy_sha256": "environment_policy_sha256",
            "environment_instance_sha256": "environment_instance_sha256",
            "command_sha256": "command_sha256",
        }
        if any(
            claim.get(claim_field) != evidence.get(evidence_field)
            for claim_field, evidence_field in claim_to_evidence_fields.items()
        ):
            return False
        raw_present = "author-content.raw.json" in files
        draft_present = "pre-review-draft.json" in files
        if (
            evidence.get("raw_final_sha256")
            != (
                files["author-content.raw.json"]
                if raw_present
                else None
            )
            or evidence.get("pre_review_draft_sha256")
            != (
                files["pre-review-draft.json"]
                if draft_present
                else None
            )
        ):
            return False
        success = status == "codex_session_completed_draft_ready_for_review"
        if (
            formal_eligible is not success
            or (success and (not raw_present or not draft_present))
            or (draft_present and not success)
            or (
                success
                and (
                    evidence.get("process_launched") is not True
                    or evidence.get("process_reaped") is not True
                    or evidence.get("exit_code") != 0
                    or evidence.get("timed_out") is not False
                    or evidence.get("stdout_limit_exceeded") is not False
                    or evidence.get("stderr_limit_exceeded") is not False
                    or evidence.get("stdin_completed") is not True
                    or evidence.get("process_supervision_error") is not None
                    or evidence.get("visible_tool_activity") is not False
                )
            )
        ):
            return False
        audit = v2_runner._event_audit(
            read_stable_regular_file(
                output_dir / "codex-events.jsonl",
                label="Codex v4 canonical event log",
            )
        )
        if (
            evidence.get("thread_id") != audit.thread_id
            or evidence.get("input_tokens") != audit.input_tokens
            or evidence.get("cached_input_tokens") != audit.cached_input_tokens
            or evidence.get("output_tokens") != audit.output_tokens
            or evidence.get("event_types") != list(audit.event_types)
            or evidence.get("visible_tool_activity")
            != audit.visible_tool_activity
            or evidence.get("visible_agent_message_count")
            != len(audit.agent_messages)
            or evidence.get("disallowed_item_types")
            != list(audit.disallowed_item_types)
        ):
            return False
        audit_formal_success = True
        try:
            audit.require_formal_success()
        except BaseException:
            audit_formal_success = False
        status_consistent = {
            "claim_consumed_prelaunch_cleanup_failed": (
                evidence.get("process_launch_attempted") is False
                and evidence.get("process_launched") is False
                and (
                    evidence.get("claim_commit_recovered_after_exception") is True
                    or evidence.get("claim_commit_cleanup_error") is not None
                )
            ),
            "codex_process_timed_out": (
                evidence.get("process_launched") is True
                and evidence.get("timed_out") is True
            ),
            "codex_process_output_limit_exceeded": (
                evidence.get("process_launched") is True
                and (
                    evidence.get("stdout_limit_exceeded") is True
                    or evidence.get("stderr_limit_exceeded") is True
                )
            ),
            "codex_process_nonzero_exit": (
                evidence.get("process_launched") is True
                and isinstance(evidence.get("exit_code"), int)
                and evidence.get("exit_code") != 0
            ),
            "codex_process_supervision_failed": (
                evidence.get("process_launched") is True
                and (
                    evidence.get("process_supervision_error") is not None
                    or evidence.get("stdin_completed") is not True
                )
            ),
            "codex_session_event_audit_rejected": (
                evidence.get("process_launched") is True
                and not audit_formal_success
            ),
            "codex_session_incomplete": (
                evidence.get("process_launched") is True
                and audit.turn_completed_count != 1
            ),
            "codex_session_completed_draft_rejected": (
                evidence.get("process_launched") is True
                and audit_formal_success
                and not draft_present
            ),
            "codex_session_completed_draft_ready_for_review": (
                evidence.get("process_launched") is True
                and audit_formal_success
                and raw_present
                and draft_present
            ),
        }
        if status_consistent.get(status) is not True:
            return False
        if success:
            raw_bytes = read_stable_regular_file(
                output_dir / "author-content.raw.json",
                label="Codex v4 canonical raw final message",
            )
            agent_message = audit.agent_messages[0].encode("utf-8")
            if raw_bytes not in {agent_message, agent_message + b"\n"}:
                return False
            compiled = normalize_codex_authoring_output(
                raw_bytes,
                authoring_input=authoring_input,
                semantic_source=semantic_source,
            )
            archived_draft = read_stable_regular_file(
                output_dir / "pre-review-draft.json",
                label="Codex v4 canonical pre-review draft",
            )
            if compiled.canonical_bytes() != archived_draft:
                return False
        elif formal_eligible is not False:
            return False
        return True
    except BaseException:
        return False


def resolve_terminal_state(
    *,
    guard_path: Path,
    guard_file_sha256: str,
    claim_path: Path,
    output_dir: Path,
) -> str:
    """Resolve the durable terminal state without trusting runner exit status."""
    try:
        if not os.path.lexists(claim_path):
            _terminal_guard(
                guard_path,
                guard_file_sha256,
                claim_path=claim_path,
                output_dir=output_dir,
                receipt_path=output_dir / "invocation-receipt.json",
            )
            return "terminal_guard_inactive_no_claim"
        _load_terminal_validation_context(
            guard_path=guard_path,
            guard_file_sha256=guard_file_sha256,
            claim_path=claim_path,
            output_dir=output_dir,
        )
        if validate_canonical_bundle_from_disk(
            output_dir,
            claim_path=claim_path,
            guard_path=guard_path,
            guard_file_sha256=guard_file_sha256,
        ):
            return "canonical_terminal_receipt_valid"
        return "terminal_guard_active_authorization_consumed_result_rejected"
    except BaseException:
        return "terminal_state_unverifiable_authorization_not_reusable"


def _write_or_verify_staging_file(
    staging: Path,
    name: str,
    content: bytes,
) -> None:
    path = staging / name
    if os.path.lexists(path):
        if (
            path.is_symlink()
            or not path.is_file()
            or read_stable_regular_file(
                path,
                label=f"Codex v4 staging file {name}",
            )
            != content
        ):
            raise CodexAuthoringContractError(
                f"Codex v4 staging file conflicts: {name}"
            )
        return
    atomic_create_file(path, content)


def _event_evidence(
    *,
    state: dict[str, Any],
    executable: Path,
    runtime: dict[str, Any],
    freeze: dict[str, Any],
    freeze_file_sha256: str,
    approval_path: Path,
    approval_file_sha256: str,
    approval: dict[str, Any],
    claim_bytes: bytes,
    claim_payload_sha256: str,
    packet_sha256: str,
    request_sha256: str,
    stdin_sha256: str,
    schema_sha256: str,
    runtime_file_sha256: str,
    source_manifest_file_sha256: str,
    binary_evidence_file_sha256: str,
    prompt_isolation_file_sha256: str,
    environment_policy_sha256: str,
    environment_instance_sha256: str,
    normalized_command: list[str],
    guard_path: Path,
    guard_file_sha256: str,
) -> dict[str, Any]:
    audit = state["audit"]
    return {
        "schema_version": 4,
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "status": state["status"],
        "execution_type": "codex_mediated_static_author_v1",
        "evidence_tier": "platform-mediated_non-provider-attested",
        "formal_codex_session_eligible": (
            state["status"] == "codex_session_completed_draft_ready_for_review"
        ),
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
        "cached_input_tokens": audit.cached_input_tokens if audit else None,
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
        "terminal_guard_is_audit_fallback_not_model_fallback": True,
        "visible_tool_activity": audit.visible_tool_activity if audit else None,
        "visible_agent_message_count": (
            len(audit.agent_messages) if audit else None
        ),
        "disallowed_item_types": (
            list(audit.disallowed_item_types) if audit else []
        ),
        "input_isolation": "behaviorally_constrained_not_mechanically_proven",
        "repository_working_directory_excluded": True,
        "external_scratch_was_empty_before_launch": True,
        "parent_path_inherited": False,
        "parent_pathext_inherited": False,
        "parent_temp_inherited": False,
        "absolute_binary_path_used": True,
        "binary_path": str(executable),
        "binary_sha256": runtime["binary"]["sha256"],
        "process_launched": state["process_launched"],
        "process_launch_attempted": state["process_launch_attempted"],
        "process_reaped": state["process_reaped"],
        "pipe_threads_terminated": state["pipe_threads_terminated"],
        "process_result_committed": state["process_result_committed"],
        "exit_code": state["exit_code"],
        "timed_out": state["timed_out"],
        "stdout_limit_exceeded": state["stdout_limit_exceeded"],
        "stderr_limit_exceeded": state["stderr_limit_exceeded"],
        "stdin_completed": state["stdin_completed"],
        "process_supervision_error": state["supervision_error"],
        "elapsed_ms": state["elapsed_ms"],
        "event_types": list(audit.event_types) if audit else [],
        "event_audit_error": state["event_audit_error"],
        "event_log_sha256": sha256_bytes(state["stdout"]),
        "stderr_sha256": sha256_bytes(state["stderr"]),
        "raw_final_sha256": state["raw_final_sha256"],
        "pre_review_draft_sha256": state["draft_sha256"],
        "error_type": state["error_type"],
        "error_message": state["error_message"],
        "failure_stage": state["failure_stage"],
        "scratch_cleanup_succeeded": state["scratch_cleanup_succeeded"],
        "scratch_cleanup_error": state["scratch_cleanup_error"],
        "claim_commit_recovered_after_exception": state[
            "claim_commit_recovered_after_exception"
        ],
        "claim_commit_cleanup_error": state["claim_commit_cleanup_error"],
        "freeze_lock_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "owner_approval_file": approval_path.relative_to(ROOT).as_posix(),
        "owner_approval_file_sha256": approval_file_sha256,
        "owner_approval_payload_sha256": approval["approval_payload_sha256"],
        "v2_retirement_claim_file_sha256": approval[
            "v2_retirement_claim_file_sha256"
        ],
        "terminal_guard_file": guard_path.relative_to(ROOT).as_posix(),
        "terminal_guard_file_sha256": guard_file_sha256,
        "attempt_claim_file_sha256": sha256_bytes(claim_bytes),
        "attempt_claim_payload_sha256": claim_payload_sha256,
        "authoring_input_file_sha256": packet_sha256,
        "canonical_request_file_sha256": request_sha256,
        "stdin_request_file_sha256": stdin_sha256,
        "output_schema_file_sha256": schema_sha256,
        "runtime_lock_file_sha256": runtime_file_sha256,
        "runtime_payload_sha256": runtime["runtime_payload_sha256"],
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "binary_evidence_file_sha256": binary_evidence_file_sha256,
        "prompt_isolation_evidence_file_sha256": (
            prompt_isolation_file_sha256
        ),
        "environment_policy_sha256": environment_policy_sha256,
        "environment_instance_sha256": environment_instance_sha256,
        "command_sha256": sha256_bytes(canonical_json_bytes(normalized_command)),
        "descendant_process_termination_evidence": (
            "cli_process_reaped; full_descendant_tree_not_independently_attested"
            if state["process_launched"] is True and state["process_reaped"]
            else "cli_process_reap_not_applicable_or_not_confirmed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-lock-file-sha256", required=True)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--approval-record-file-sha256", required=True)
    args = parser.parse_args()

    freeze_file_sha256 = args.freeze_lock_file_sha256
    freeze = v2_runner._load_object(
        FREEZE_PATH,
        freeze_file_sha256,
        "Codex v4 freeze lock",
    )
    v2_runner._self_hash(freeze, "freeze_payload_sha256", "Codex v4 freeze lock")
    if (
        freeze.get("status") != "frozen_candidate_awaiting_owner_confirmation"
        or freeze.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or freeze.get("invocation_authorized") is not False
    ):
        raise CodexAuthoringContractError(
            "Codex v4 freeze lock is not an approvable candidate"
        )

    required_binding_names = (
        "model_role_selection",
        "protocol",
        "model_access_evidence",
        "binary_binding_evidence",
        "prompt_isolation_evidence",
        "semantic_source",
        "authoring_input",
        "output_schema",
        "canonical_request",
        "stdin_request",
        "runtime_lock",
        "runner",
        "v3_runner",
        "base_runner",
        "contract_module",
        "v3_contract_module",
        "base_contract_module",
        "builder",
        "approval_builder",
        "source_manifest",
        "v2_preflight_incident",
        "v2_owner_approval",
        "v3_preapproval_audit_incident",
        "v3_freeze",
    )
    freeze_bindings = {
        name: v2_runner._binding(freeze, name)
        for name in required_binding_names
    }
    packet_path, packet_sha256 = freeze_bindings["authoring_input"]
    semantic_path, semantic_sha256 = freeze_bindings["semantic_source"]
    request_path, request_sha256 = freeze_bindings["canonical_request"]
    stdin_path, stdin_sha256 = freeze_bindings["stdin_request"]
    schema_path, schema_sha256 = freeze_bindings["output_schema"]
    runtime_path, runtime_file_sha256 = freeze_bindings["runtime_lock"]
    runtime = v2_runner._load_object(
        runtime_path,
        runtime_file_sha256,
        "Codex v4 runtime lock",
    )
    v2_runner._self_hash(
        runtime,
        "runtime_payload_sha256",
        "Codex v4 runtime lock",
    )
    if (
        runtime.get("status") != "frozen"
        or runtime.get("runtime_id") != "codex-cli-author-runtime-20260724-v4"
    ):
        raise CodexAuthoringContractError("Codex v4 runtime lock is not frozen")
    _validate_frozen_source_bindings(freeze_bindings, runtime)

    source_manifest_path, source_manifest_file_sha256 = freeze_bindings[
        "source_manifest"
    ]
    source_manifest = v2_runner._load_object(
        source_manifest_path,
        source_manifest_file_sha256,
        "Codex v4 source manifest",
    )
    _validate_dependency_snapshot(source_manifest)
    binary_evidence_path, binary_evidence_file_sha256 = freeze_bindings[
        "binary_binding_evidence"
    ]
    _binary_evidence(
        binary_evidence_path,
        binary_evidence_file_sha256,
        runtime,
    )
    prompt_isolation_path, prompt_isolation_file_sha256 = freeze_bindings[
        "prompt_isolation_evidence"
    ]
    prompt_isolation = v2_runner._load_object(
        prompt_isolation_path,
        prompt_isolation_file_sha256,
        "Codex prompt-isolation evidence",
    )
    v2_runner._self_hash(
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
    request_bytes = v2_runner._read_bound_bytes(
        request_path,
        request_sha256,
        "Codex v4 canonical request",
    )
    stdin_bytes = v2_runner._read_bound_bytes(
        stdin_path,
        stdin_sha256,
        "Codex v4 stdin request",
    )
    schema_bytes = v2_runner._read_bound_bytes(
        schema_path,
        schema_sha256,
        "Codex v4 output schema",
    )
    request = parse_codex_authoring_request(request_bytes)
    try:
        raw_schema = parse_canonical_json(
            schema_bytes,
            label="Codex v4 output schema",
        )
    except ArtifactFormatError as error:
        raise CodexAuthoringContractError(
            "Codex v4 output schema is not canonical JSON"
        ) from error
    if not isinstance(raw_schema, dict):
        raise CodexAuthoringContractError(
            "Codex v4 output schema must contain an object"
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
            "Codex v4 packet/request/schema/stdin or budget drifted"
        )

    authorization = freeze.get("authorization")
    paths = authorization.get("paths") if isinstance(authorization, dict) else None
    if not isinstance(paths, dict) or paths.get("run_id") != CODEX_AUTHOR_RUN_ID:
        raise CodexAuthoringContractError(
            "Codex v4 authorization paths are missing or drifted"
        )
    output_dir = v2_runner._repo_path(
        paths.get("output_directory"),
        "Codex v4 output directory",
        must_exist=False,
    )
    claim_path = v2_runner._repo_path(
        paths.get("claim_file"),
        "Codex v4 claim",
        must_exist=False,
    )
    receipt_path = v2_runner._repo_path(
        paths.get("receipt_file"),
        "Codex v4 receipt",
        must_exist=False,
    )
    guard_path = v2_runner._repo_path(
        paths.get("terminal_guard_file"),
        "Codex v4 conditional terminal guard",
        must_exist=True,
    )
    expected_approval_path = v2_runner._repo_path(
        paths.get("approval_record_file"),
        "frozen Codex v4 owner approval",
        must_exist=True,
    )
    approval_path = v2_runner._repo_path(
        Path(args.approval_record).as_posix(),
        "Codex v4 owner approval",
        must_exist=True,
    )
    if (
        approval_path != expected_approval_path
        or receipt_path != output_dir / "invocation-receipt.json"
    ):
        raise CodexAuthoringContractError(
            "Codex v4 approval or receipt path differs from the freeze"
        )
    guard_policy = freeze.get("terminal_guard")
    if (
        not isinstance(guard_policy, dict)
        or guard_policy.get("file")
        != guard_path.relative_to(ROOT).as_posix()
        or not isinstance(guard_policy.get("expected_file_sha256"), str)
    ):
        raise CodexAuthoringContractError(
            "Codex v4 terminal guard policy is missing"
        )
    guard_file_sha256 = guard_policy["expected_file_sha256"]
    approval = _approval(
        approval_path,
        args.approval_record_file_sha256,
        freeze,
        freeze_file_sha256,
        guard_path=guard_path,
        guard_file_sha256=guard_file_sha256,
        claim_path=claim_path,
        output_dir=output_dir,
        receipt_path=receipt_path,
    )
    if any(os.path.lexists(path) for path in (output_dir, claim_path, receipt_path)):
        raise FileExistsError(
            "Codex v4 authorization is consumed or a canonical output path exists"
        )

    executable = resolve_frozen_codex_binary(runtime)
    staging = new_staging_directory(output_dir)
    temp_root: Path | None = None
    try:
        temp_root, scratch, control = v2_runner._new_external_scratch_root()
        invocation_temp = control / "child-temp"
        invocation_temp.mkdir()
        safe_env, environment_policy_sha256, environment_instance_sha256 = (
            construct_codex_v4_environment(
                runtime=runtime,
                executable=executable,
                invocation_temp=invocation_temp,
                parent_environment=os.environ,
            )
        )
        materialized_schema_path = control / "output-schema.json"
        atomic_create_file(materialized_schema_path, schema_bytes)
        final_message_path = staging / "author-content.raw.json"
        command = v2_runner._command(
            executable,
            scratch=scratch,
            schema=materialized_schema_path,
            final_message=final_message_path,
        )
        normalized_command = v2_runner._normalized_command(
            command,
            executable=executable,
            scratch=scratch,
            schema=materialized_schema_path,
            final_message=final_message_path,
        )
        if (
            runtime.get("command_shape") != list(CODEX_COMMAND_SHAPE)
            or normalized_command != list(CODEX_COMMAND_SHAPE)
            or any(scratch.iterdir())
        ):
            raise CodexAuthoringContractError(
                "Codex v4 runtime command shape drifted"
            )
    except BaseException:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Every allocation needed to enter the post-claim boundary happens here.
    process_spawned = threading.Event()
    state: dict[str, Any] = {
        "started_ns": time.perf_counter_ns(),
        "exit_code": None,
        "stdout": b"",
        "stderr": b"",
        "status": "codex_process_failed",
        "error_type": None,
        "error_message": None,
        "draft_bytes": None,
        "draft_sha256": None,
        "raw_final_bytes": None,
        "raw_final_sha256": None,
        "audit": None,
        "event_audit_error": None,
        "timed_out": False,
        "stdout_limit_exceeded": False,
        "stderr_limit_exceeded": False,
        "process_launched": False,
        "process_launch_attempted": False,
        "process_reaped": False,
        "pipe_threads_terminated": False,
        "process_result_committed": False,
        "stdin_completed": False,
        "supervision_error": None,
        "failure_stage": "claim_commit",
        "scratch_cleanup_succeeded": False,
        "scratch_cleanup_error": None,
        "elapsed_ms": 0,
        "claim_commit_recovered_after_exception": False,
        "claim_commit_cleanup_error": None,
    }
    claimant_nonce = secrets.token_hex(32)
    claim_payload = {
        "schema_version": 3,
        "status": "consumed_before_codex_process_launch",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "claimant_nonce": claimant_nonce,
        "freeze_lock_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "owner_approval_file": approval_path.relative_to(ROOT).as_posix(),
        "owner_approval_file_sha256": args.approval_record_file_sha256,
        "owner_approval_payload_sha256": approval["approval_payload_sha256"],
        "terminal_guard_file": guard_path.relative_to(ROOT).as_posix(),
        "terminal_guard_file_sha256": guard_file_sha256,
        "v2_retirement_claim_file_sha256": approval[
            "v2_retirement_claim_file_sha256"
        ],
        "authoring_input_file_sha256": packet_sha256,
        "canonical_request_file_sha256": request_sha256,
        "stdin_request_file_sha256": stdin_sha256,
        "output_schema_file_sha256": schema_sha256,
        "materialized_output_schema_sha256": sha256_bytes(schema_bytes),
        "runtime_lock_file_sha256": runtime_file_sha256,
        "runtime_payload_sha256": runtime["runtime_payload_sha256"],
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "binary_evidence_file_sha256": binary_evidence_file_sha256,
        "prompt_isolation_evidence_file_sha256": prompt_isolation_file_sha256,
        "environment_policy_sha256": environment_policy_sha256,
        "environment_instance_sha256": environment_instance_sha256,
        "command_sha256": sha256_bytes(canonical_json_bytes(normalized_command)),
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "max_codex_exec_sessions": 1,
        "retry_fallback_repair_followup_policy": "forbidden",
        "required_output_directory": output_dir.relative_to(ROOT).as_posix(),
        "required_receipt_file": receipt_path.relative_to(ROOT).as_posix(),
    }
    claim_payload_sha256 = sha256_bytes(canonical_json_bytes(claim_payload))
    claim_bytes = canonical_json_bytes(
        {**claim_payload, "claim_payload_sha256": claim_payload_sha256}
    )

    try:
        claim_commit = classified_atomic_create(
            claim_path,
            claim_bytes,
            label="Codex v4 authorization claim",
        )
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # From this point the pre-existing guard is active.  Any uncaught
    # BaseException, including failure while entering the inner routine, leaves
    # a durable fail-closed terminal adjudication.
    try:
        state["claim_commit_recovered_after_exception"] = (
            claim_commit.recovered_after_exception
        )
        state["claim_commit_cleanup_error"] = claim_commit.cleanup_error
        state["failure_stage"] = "process_spawn"
        try:
            if not claim_commit_allows_process_launch(claim_commit):
                raise CodexAuthoringContractError(
                    "claim commit required exception recovery; process launch forbidden"
                )
            state["process_launch_attempted"] = True
            state["process_launched"] = None
            completed = _run_capped_process_v4(
                command,
                cwd=scratch,
                environment=safe_env,
                stdin=stdin_bytes,
                timeout_seconds=authoring_input.session_budget.timeout_seconds,
                stdout_limit=authoring_input.session_budget.max_event_log_bytes,
                stderr_limit=authoring_input.session_budget.max_stderr_bytes,
                spawned=process_spawned,
            )
            state["process_launched"] = completed.process_created
            state["process_reaped"] = completed.process_reaped
            state["pipe_threads_terminated"] = (
                completed.pipe_threads_terminated
            )
            state["exit_code"] = completed.returncode
            state["stdout"] = completed.stdout
            state["stderr"] = completed.stderr
            state["timed_out"] = completed.timed_out
            state["stdout_limit_exceeded"] = completed.stdout_limit_exceeded
            state["stderr_limit_exceeded"] = completed.stderr_limit_exceeded
            state["stdin_completed"] = completed.stdin_completed
            state["supervision_error"] = completed.supervision_error
            state["process_result_committed"] = True
            state["failure_stage"] = "process_result"
            try:
                state["audit"] = v2_runner._event_audit(state["stdout"])
            except BaseException as event_error:
                kind, message = _safe_error_text(event_error)
                state["event_audit_error"] = f"{kind}: {message}"
            if (
                state["timed_out"]
                or state["stdout_limit_exceeded"]
                or state["stderr_limit_exceeded"]
                or state["supervision_error"] is not None
                or not state["stdin_completed"]
                or state["exit_code"] != 0
            ):
                raise CodexAuthoringContractError(
                    "Codex v4 process did not complete within the frozen envelope"
                )
            if state["audit"] is None:
                raise CodexAuthoringContractError(
                    f"Codex v4 event audit failed: {state['event_audit_error']}"
                )
            state["failure_stage"] = "event_audit"
            state["audit"].require_formal_success()
            state["failure_stage"] = "output_capture"
            if (
                not final_message_path.is_file()
                or final_message_path.is_symlink()
            ):
                raise CodexAuthoringContractError(
                    "Codex v4 final message is missing"
                )
            raw_final = read_stable_regular_file(
                final_message_path,
                label="Codex v4 final message",
                max_bytes=authoring_input.session_budget.max_final_output_bytes,
            )
            state["raw_final_bytes"] = raw_final
            state["raw_final_sha256"] = sha256_bytes(raw_final)
            agent_message_bytes = state["audit"].agent_messages[0].encode("utf-8")
            if raw_final not in {agent_message_bytes, agent_message_bytes + b"\n"}:
                raise CodexAuthoringContractError(
                    "Codex v4 JSONL message differs from output-last-message"
                )
            state["failure_stage"] = "trusted_compiler"
            draft = normalize_codex_authoring_output(
                raw_final,
                authoring_input=authoring_input,
                semantic_source=semantic_source,
            )
            state["draft_bytes"] = draft.canonical_bytes()
            state["draft_sha256"] = sha256_bytes(state["draft_bytes"])
            state["status"] = "codex_session_completed_draft_ready_for_review"
            state["failure_stage"] = None
        except BaseException as error:
            if process_spawned.is_set():
                state["process_launched"] = True
            elif state["process_launch_attempted"]:
                state["process_launched"] = None
            else:
                state["process_launched"] = False
            state["error_type"], state["error_message"] = _safe_error_text(error)
            if (
                claim_commit.cleanup_error is not None
                or claim_commit.recovered_after_exception
            ):
                state["status"] = "claim_consumed_prelaunch_cleanup_failed"
            elif state["process_launched"] is not True:
                state["status"] = "codex_process_launch_outcome_unknown"
            elif state["timed_out"]:
                state["status"] = "codex_process_timed_out"
            elif (
                state["stdout_limit_exceeded"]
                or state["stderr_limit_exceeded"]
            ):
                state["status"] = "codex_process_output_limit_exceeded"
            elif state["exit_code"] not in (None, 0):
                state["status"] = "codex_process_nonzero_exit"
            elif (
                state["supervision_error"] is not None
                or not state["stdin_completed"]
            ):
                state["status"] = "codex_process_supervision_failed"
            elif state["audit"] is None:
                state["status"] = "codex_session_event_audit_rejected"
            elif state["audit"].turn_completed_count != 1:
                state["status"] = "codex_session_incomplete"
            else:
                state["status"] = "codex_session_completed_draft_rejected"
        finally:
            try:
                shutil.rmtree(temp_root)
                state["scratch_cleanup_succeeded"] = True
            except BaseException as cleanup_error:
                kind, message = _safe_error_text(cleanup_error)
                state["scratch_cleanup_error"] = f"{kind}: {message}"
            state["elapsed_ms"] = max(
                0,
                (time.perf_counter_ns() - state["started_ns"]) // 1_000_000,
            )

        if state["raw_final_bytes"] is None and os.path.lexists(
            final_message_path
        ):
            state["raw_final_bytes"] = read_stable_regular_file(
                final_message_path,
                label="Codex v4 staged raw final message after rejected output",
                max_bytes=authoring_input.session_budget.max_final_output_bytes,
            )
            state["raw_final_sha256"] = sha256_bytes(state["raw_final_bytes"])

        if not terminal_process_state_is_stable(state):
            raise CodexAuthoringContractError(
                "Codex v4 process state is not terminal; guard remains authoritative"
            )

        evidence = _event_evidence(
            state=state,
            executable=executable,
            runtime=runtime,
            freeze=freeze,
            freeze_file_sha256=freeze_file_sha256,
            approval_path=approval_path,
            approval_file_sha256=args.approval_record_file_sha256,
            approval=approval,
            claim_bytes=claim_bytes,
            claim_payload_sha256=claim_payload_sha256,
            packet_sha256=packet_sha256,
            request_sha256=request_sha256,
            stdin_sha256=stdin_sha256,
            schema_sha256=schema_sha256,
            runtime_file_sha256=runtime_file_sha256,
            source_manifest_file_sha256=source_manifest_file_sha256,
            binary_evidence_file_sha256=binary_evidence_file_sha256,
            prompt_isolation_file_sha256=prompt_isolation_file_sha256,
            environment_policy_sha256=environment_policy_sha256,
            environment_instance_sha256=environment_instance_sha256,
            normalized_command=normalized_command,
            guard_path=guard_path,
            guard_file_sha256=guard_file_sha256,
        )
        evidence_bytes = canonical_json_bytes(evidence)
        expected_bundle: dict[str, bytes] = {
            "codex-events.jsonl": state["stdout"],
            "codex-stderr.bin": state["stderr"],
            "authoring-request.json": request_bytes,
            "authoring-stdin.txt": stdin_bytes,
            "authoring-output-schema.json": schema_bytes,
            "session-evidence.json": evidence_bytes,
        }
        if state["raw_final_bytes"] is not None:
            expected_bundle["author-content.raw.json"] = state["raw_final_bytes"]
        if state["draft_bytes"] is not None:
            expected_bundle["pre-review-draft.json"] = state["draft_bytes"]
        evidence_files = {
            name: sha256_bytes(content)
            for name, content in sorted(expected_bundle.items())
        }
        receipt = {
            **evidence,
            "artifact_kind": "codex_authoring_canonical_terminal_receipt",
            "canonical_bundle_complete": True,
            "canonical_bundle_files": evidence_files,
            "terminal_guard_overridden_by_this_valid_bundle": True,
        }
        receipt_bytes = canonical_json_bytes(
            {
                **receipt,
                "receipt_payload_sha256": sha256_bytes(
                    canonical_json_bytes(receipt)
                ),
            }
        )
        expected_bundle["invocation-receipt.json"] = receipt_bytes
        for name, content in expected_bundle.items():
            _write_or_verify_staging_file(staging, name, content)
        _publish_and_classify_bundle(
            staging,
            output_dir,
            expected_bundle,
            claim_path=claim_path,
            guard_path=guard_path,
            guard_file_sha256=guard_file_sha256,
        )
    except BaseException as terminal_error:
        # No file creation is required here: the exact, pre-approved guard is
        # already active because the claim binds its SHA.
        kind, message = _safe_error_text(terminal_error)
        print(
            json.dumps(
                {
                    "status": "authorization_consumed_terminal_guard_active",
                    "run_id": CODEX_AUTHOR_RUN_ID,
                    "authorization_consumed": True,
                    "canonical_receipt": None,
                    "terminal_guard": guard_path.relative_to(ROOT).as_posix(),
                    "terminalization_error_type": kind,
                    "terminalization_error_message": message,
                },
                sort_keys=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": state["status"],
                "run_id": CODEX_AUTHOR_RUN_ID,
                "authorization_consumed": True,
                "canonical_receipt": receipt_path.relative_to(ROOT).as_posix(),
                "terminal_guard": guard_path.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        )
    )
    return (
        0
        if state["status"] == "codex_session_completed_draft_ready_for_review"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
