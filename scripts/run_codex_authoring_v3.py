"""Execute the single approved Codex/high v3 authoring session.

The mature JSONL auditing, process supervision, and atomic publication helpers
are imported from the byte-preserved v2 runner.  This entry point owns a new
candidate/run namespace and replaces the v2 ambient-PATH boundary with the
absolute-binary, deterministic-environment policy frozen by v3.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any

import scripts.run_codex_authoring as v2_runner
import skillchain.codex_authoring as base_contract_module
import skillchain.codex_authoring_v3 as contract_module
from skillchain.codex_authoring_v3 import (
    CODEX_AUTHOR_CANDIDATE_ID,
    CODEX_AUTHOR_RUN_ID,
    CODEX_COMMAND_SHAPE,
    CodexAuthoringContractError,
    build_codex_runtime_dependency_snapshot_v3,
    construct_codex_v3_environment,
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
BASE_RUNNER_PATH = Path(v2_runner.__file__).resolve(strict=True)
CONTRACT_MODULE_PATH = Path(contract_module.__file__).resolve(strict=True)
BASE_CONTRACT_MODULE_PATH = Path(
    base_contract_module.__file__
).resolve(strict=True)
FREEZE_PATH = (
    ROOT / "specs" / "authoring" / "authoring-freeze-lock-codex-high-v3.json"
)
V2_RUN_ID = "llm-static-codex-primary-20260724-high-v2"
V2_RETIREMENT_CLAIM_PATH = (
    ROOT / "specs" / "authoring" / f"{V2_RUN_ID}-attempt-claim.json"
)
V2_OUTPUT_DIR = ROOT / "runs" / "formal-authoring" / V2_RUN_ID
V2_RECEIPT_PATH = V2_OUTPUT_DIR / "invocation-receipt.json"


def _consume_claim(claim_path: Path, claim_bytes: bytes) -> None:
    """Atomically consume authority; callers may launch only after this returns."""
    atomic_create_file(claim_path, claim_bytes)


def _validate_frozen_source_bindings(
    freeze_bindings: dict[str, tuple[Path, str]],
    runtime: dict[str, Any],
) -> None:
    expected = {
        "runner": ("runner", (RUNNER_PATH, freeze_bindings["runner"][1])),
        "base_runner": (
            "base_runner",
            (BASE_RUNNER_PATH, freeze_bindings["base_runner"][1]),
        ),
        "contract_module": (
            "contract_module",
            (CONTRACT_MODULE_PATH, freeze_bindings["contract_module"][1]),
        ),
        "base_contract_module": (
            "base_contract_module",
            (
                BASE_CONTRACT_MODULE_PATH,
                freeze_bindings["base_contract_module"][1],
            ),
        ),
        "source_manifest": (
            "source_manifest",
            freeze_bindings["source_manifest"],
        ),
        "prompt_isolation_evidence": (
            "prompt_isolation_evidence",
            freeze_bindings["prompt_isolation_evidence"],
        ),
        "binary_evidence": (
            "binary_binding_evidence",
            freeze_bindings["binary_binding_evidence"],
        ),
    }
    for runtime_name, (freeze_name, binding) in expected.items():
        if freeze_bindings.get(freeze_name) != binding:
            raise CodexAuthoringContractError(
                f"Codex v3 freeze {freeze_name} binding is not canonical"
            )
        if v2_runner._runtime_binding(runtime, runtime_name) != binding:
            raise CodexAuthoringContractError(
                f"Codex v3 runtime {runtime_name} binding differs from the freeze"
            )


def _validate_dependency_snapshot(manifest: dict[str, Any]) -> None:
    v2_runner._self_hash(
        manifest,
        "source_manifest_payload_sha256",
        "Codex v3 source manifest",
    )
    unsigned = dict(manifest)
    unsigned.pop("source_manifest_payload_sha256", None)
    live = build_codex_runtime_dependency_snapshot_v3(ROOT)
    if canonical_json_bytes(unsigned) != canonical_json_bytes(live):
        raise CodexAuthoringContractError(
            "Codex v3 source or runtime dependency snapshot drifted"
        )


def _approval(
    approval_path: Path,
    approval_file_sha256: str,
    freeze: dict[str, Any],
    freeze_file_sha256: str,
) -> dict[str, Any]:
    raw = v2_runner._load_object(
        approval_path,
        approval_file_sha256,
        "Codex v3 owner approval",
    )
    v2_runner._self_hash(
        raw,
        "approval_payload_sha256",
        "Codex v3 owner approval",
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
    }
    if set(raw) != set(required) | {"approval_payload_sha256"}:
        raise CodexAuthoringContractError(
            "Codex v3 owner approval contains missing or unexpected fields"
        )
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise CodexAuthoringContractError(
                f"Codex v3 owner approval does not accept frozen field: {key}"
            )
    if (
        retirement.get("status")
        != "authorization_retired_without_codex_process_launch"
        or retirement.get("run_id") != V2_RUN_ID
        or retirement.get("codex_exec_process_spawned") is not False
        or retirement.get("inference_requested") is not False
        or retirement.get("v2_authorization_reusable") is not False
        or os.path.lexists(V2_OUTPUT_DIR)
        or os.path.lexists(V2_RECEIPT_PATH)
    ):
        raise CodexAuthoringContractError(
            "Codex v2 authorization was not safely retired"
        )
    retirement_policy = freeze.get("prior_authorization_retirement")
    if (
        not isinstance(retirement_policy, dict)
        or retirement_policy.get("retirement_claim_file")
        != V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
        or retirement_policy.get("retirement_claim_expected_file_sha256")
        != required["v2_retirement_claim_file_sha256"]
        or retirement_policy.get("v2_approval_reusable") is not False
    ):
        raise CodexAuthoringContractError(
            "Codex v3 freeze does not bind the v2 retirement"
        )
    return raw


def _binary_evidence(
    path: Path,
    expected_sha256: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    evidence = v2_runner._load_object(
        path,
        expected_sha256,
        "Codex v3 absolute-binary evidence",
    )
    v2_runner._self_hash(
        evidence,
        "binary_evidence_payload_sha256",
        "Codex v3 absolute-binary evidence",
    )
    if (
        evidence.get("status") != "verified_without_inference"
        or evidence.get("inference_request_performed") is not False
        or evidence.get("binary") != runtime.get("binary")
    ):
        raise CodexAuthoringContractError(
            "Codex v3 absolute-binary evidence differs from runtime"
        )
    return evidence


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
        "Codex v3 freeze lock",
    )
    v2_runner._self_hash(
        freeze,
        "freeze_payload_sha256",
        "Codex v3 freeze lock",
    )
    if (
        freeze.get("status") != "frozen_candidate_awaiting_owner_confirmation"
        or freeze.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or freeze.get("invocation_authorized") is not False
    ):
        raise CodexAuthoringContractError(
            "Codex v3 freeze lock is not an approvable candidate"
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
        "base_runner",
        "contract_module",
        "base_contract_module",
        "builder",
        "approval_builder",
        "source_manifest",
        "v2_preflight_incident",
        "v2_owner_approval",
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
        "Codex v3 runtime lock",
    )
    v2_runner._self_hash(
        runtime,
        "runtime_payload_sha256",
        "Codex v3 runtime lock",
    )
    if (
        runtime.get("status") != "frozen"
        or runtime.get("runtime_id") != "codex-cli-author-runtime-20260724-v3"
    ):
        raise CodexAuthoringContractError("Codex v3 runtime lock is not frozen")
    _validate_frozen_source_bindings(freeze_bindings, runtime)

    source_manifest_path, source_manifest_file_sha256 = freeze_bindings[
        "source_manifest"
    ]
    source_manifest = v2_runner._load_object(
        source_manifest_path,
        source_manifest_file_sha256,
        "Codex v3 source manifest",
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
        "Codex v3 canonical request",
    )
    stdin_bytes = v2_runner._read_bound_bytes(
        stdin_path,
        stdin_sha256,
        "Codex v3 stdin request",
    )
    schema_bytes = v2_runner._read_bound_bytes(
        schema_path,
        schema_sha256,
        "Codex v3 output schema",
    )
    request = parse_codex_authoring_request(request_bytes)
    try:
        raw_schema = parse_canonical_json(
            schema_bytes,
            label="Codex v3 output schema",
        )
    except ArtifactFormatError as error:
        raise CodexAuthoringContractError(
            "Codex v3 output schema is not canonical JSON"
        ) from error
    if not isinstance(raw_schema, dict):
        raise CodexAuthoringContractError(
            "Codex v3 output schema must contain an object"
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
            "Codex v3 packet/request/schema/stdin or budget drifted"
        )

    authorization = freeze.get("authorization")
    paths = authorization.get("paths") if isinstance(authorization, dict) else None
    if not isinstance(paths, dict):
        raise CodexAuthoringContractError(
            "Codex v3 authorization paths are missing"
        )
    if paths.get("run_id") != CODEX_AUTHOR_RUN_ID:
        raise CodexAuthoringContractError(
            "Codex v3 authorization run ID drifted"
        )
    expected_approval_path = v2_runner._repo_path(
        paths.get("approval_record_file"),
        "frozen Codex v3 owner approval",
        must_exist=True,
    )
    approval_path = v2_runner._repo_path(
        Path(args.approval_record).as_posix(),
        "Codex v3 owner approval",
        must_exist=True,
    )
    if approval_path != expected_approval_path:
        raise CodexAuthoringContractError(
            "Codex v3 owner approval path differs from the freeze"
        )
    approval = _approval(
        approval_path,
        args.approval_record_file_sha256,
        freeze,
        freeze_file_sha256,
    )

    output_dir = v2_runner._repo_path(
        paths.get("output_directory"),
        "Codex v3 output directory",
        must_exist=False,
    )
    claim_path = v2_runner._repo_path(
        paths.get("claim_file"),
        "Codex v3 claim",
        must_exist=False,
    )
    receipt_path = v2_runner._repo_path(
        paths.get("receipt_file"),
        "Codex v3 receipt",
        must_exist=False,
    )
    if receipt_path != output_dir / "invocation-receipt.json":
        raise CodexAuthoringContractError(
            "Codex v3 receipt must be published inside the output directory"
        )
    if any(os.path.lexists(path) for path in (output_dir, claim_path, receipt_path)):
        raise FileExistsError(
            "Codex v3 authorization is consumed or an output path exists"
        )

    executable = resolve_frozen_codex_binary(runtime)
    staging = new_staging_directory(output_dir)
    temp_root: Path | None = None
    environment_policy_sha256: str | None = None
    environment_instance_sha256: str | None = None
    try:
        temp_root, scratch, control = v2_runner._new_external_scratch_root()
        invocation_temp = control / "child-temp"
        invocation_temp.mkdir()
        safe_env, environment_policy_sha256, environment_instance_sha256 = (
            construct_codex_v3_environment(
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
                "Codex v3 runtime command shape drifted"
            )
    except Exception:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    claim_payload = {
        "schema_version": 2,
        "status": "consumed_before_codex_process_launch",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "freeze_lock_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "owner_approval_file": approval_path.relative_to(ROOT).as_posix(),
        "owner_approval_file_sha256": args.approval_record_file_sha256,
        "owner_approval_payload_sha256": approval["approval_payload_sha256"],
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
        "prompt_isolation_evidence_file_sha256": (
            prompt_isolation_file_sha256
        ),
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
        {
            **claim_payload,
            "claim_payload_sha256": claim_payload_sha256,
        }
    )
    try:
        _consume_claim(claim_path, claim_bytes)
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
    audit: v2_runner.CodexEventAudit | None = None
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
        completed = v2_runner._run_capped_process(
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
            audit = v2_runner._event_audit(stdout)
        except Exception as event_error:
            event_audit_error = (
                f"{type(event_error).__name__}: {event_error}"
            )
        if (
            timed_out
            or stdout_limit_exceeded
            or stderr_limit_exceeded
            or supervision_error is not None
            or not stdin_completed
            or exit_code != 0
        ):
            raise CodexAuthoringContractError(
                "Codex v3 process did not complete within the frozen envelope"
            )
        if audit is None:
            raise CodexAuthoringContractError(
                f"Codex v3 event audit failed: {event_audit_error}"
            )
        failure_stage = "event_audit"
        audit.require_formal_success()
        failure_stage = "output_capture"
        if not final_message_path.is_file() or final_message_path.is_symlink():
            raise CodexAuthoringContractError("Codex v3 final message is missing")
        raw_final = read_stable_regular_file(
            final_message_path,
            label="Codex v3 final message",
            max_bytes=authoring_input.session_budget.max_final_output_bytes,
        )
        raw_final_sha256 = sha256_bytes(raw_final)
        agent_message_bytes = audit.agent_messages[0].encode("utf-8")
        if raw_final not in {
            agent_message_bytes,
            agent_message_bytes + b"\n",
        }:
            raise CodexAuthoringContractError(
                "Codex v3 JSONL message differs from output-last-message"
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
    except Exception as error:  # The claim is consumed; preserve every failure.
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
            "schema_version": 3,
            "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
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
            "parent_path_inherited": False,
            "parent_pathext_inherited": False,
            "parent_temp_inherited": False,
            "absolute_binary_path_used": True,
            "binary_path": str(executable),
            "binary_sha256": runtime["binary"]["sha256"],
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
            "v2_retirement_claim_file_sha256": approval[
                "v2_retirement_claim_file_sha256"
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
            "binary_evidence_file_sha256": binary_evidence_file_sha256,
            "prompt_isolation_evidence_file_sha256": (
                prompt_isolation_file_sha256
            ),
            "environment_policy_sha256": environment_policy_sha256,
            "environment_instance_sha256": environment_instance_sha256,
            "command_sha256": sha256_bytes(
                canonical_json_bytes(normalized_command)
            ),
            "descendant_process_termination_evidence": (
                "cli_process_reaped; full_descendant_tree_not_independently_attested"
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
