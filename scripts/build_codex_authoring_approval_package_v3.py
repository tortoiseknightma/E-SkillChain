"""Build the create-only Codex/high v3 authoring approval package.

The v3 candidate changes only the machine runtime boundary: it binds an
absolute Codex binary and constructs PATH/TEMP/TMP instead of inheriting the
ambient desktop PATH.  It does not authorize inference.  The separate v3
approval command must retire the approved-but-uninvoked v2 authorization before
it can create a v3 owner approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from skillchain import config
from skillchain.codex_authoring_v3 import (
    CODEX_AUTHOR_CANDIDATE_ID,
    CODEX_AUTHOR_RUN_ID,
    CODEX_COMMAND_SHAPE,
    CODEX_EVIDENCE_TIER,
    CODEX_EXECUTION_TYPE,
    CODEX_FROZEN_BINARY_PATH,
    build_codex_authoring_input,
    build_codex_authoring_request,
    build_codex_output_contract,
    build_codex_runtime_dependency_snapshot_v3,
    build_codex_v3_environment_policy,
    freeze_codex_binary_identity,
    load_codex_model_access_evidence,
    render_codex_authoring_stdin,
    validate_codex_cli_output_schema,
)
from skillchain.static_authoring import (
    PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    authoring_content_json_schema,
    load_authoring_packet,
)
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
AUTHORING_ROOT = ROOT / "specs" / "authoring"
DEPLOY_ROOT = ROOT / "deploy" / "authoring" / "locks-codex-v3"

SEMANTIC_SOURCE_PATH = (
    AUTHORING_ROOT / "authoring-packet-primary-v5-candidate.json"
)
SEMANTIC_SOURCE_SHA256 = (
    "cd85f4b16974f8b2523538f5dc628d48f5492220896f92c813c28dafc48f7193"
)
ACCESS_EVIDENCE_PATH = AUTHORING_ROOT / "codex-cli-model-access-evidence-v2.json"
ACCESS_EVIDENCE_SHA256 = (
    "68c3124348e12c6083c55cda41d0a8f431e25f0dfe84b3c40f200fd3e63dd4c3"
)
PROMPT_ISOLATION_EVIDENCE_PATH = (
    AUTHORING_ROOT / "codex-cli-prompt-isolation-evidence-v1.json"
)
PROMPT_ISOLATION_EVIDENCE_SHA256 = (
    "46ea406aa2a39776e2672d79e9db0a413070c3a9c3e71e595131a43d9119adff"
)
ROLE_SELECTION_PATH = AUTHORING_ROOT / "model-role-selection-v3.json"
ROLE_SELECTION_SHA256 = (
    "ce3f09bdb5c7b38adb91f2ad4e21c2a9206a43acbb3ee15c222a39c75645297f"
)

RUNNER_PATH = ROOT / "scripts" / "run_codex_authoring_v3.py"
CONTRACT_MODULE_PATH = ROOT / "src" / "skillchain" / "codex_authoring_v3.py"
BASE_RUNNER_PATH = ROOT / "scripts" / "run_codex_authoring.py"
BASE_CONTRACT_MODULE_PATH = ROOT / "src" / "skillchain" / "codex_authoring.py"
BUILDER_PATH = Path(__file__).resolve()
APPROVAL_BUILDER_PATH = ROOT / "scripts" / "approve_codex_authoring_v3.py"

PACKET_PATH = AUTHORING_ROOT / "authoring-packet-codex-high-v3.json"
SCHEMA_PATH = AUTHORING_ROOT / "authoring-content-schema-codex-high-v3.json"
REQUEST_PATH = AUTHORING_ROOT / "authoring-request-codex-high-v3.json"
STDIN_PATH = AUTHORING_ROOT / "authoring-stdin-codex-high-v3.txt"
PROTOCOL_PATH = AUTHORING_ROOT / "authoring-codex-mediation-protocol-v3.json"
BINARY_EVIDENCE_PATH = (
    AUTHORING_ROOT / "codex-cli-absolute-binary-evidence-v3.json"
)
V2_PREFLIGHT_INCIDENT_PATH = (
    AUTHORING_ROOT / "codex-high-v2-preclaim-environment-rejection-v1.json"
)
SOURCE_MANIFEST_PATH = DEPLOY_ROOT / "source-manifest.json"
RUNTIME_PATH = DEPLOY_ROOT / "runtime-lock.json"
FREEZE_PATH = AUTHORING_ROOT / "authoring-freeze-lock-codex-high-v3.json"

OUTPUT_DIR = ROOT / "runs" / "formal-authoring" / CODEX_AUTHOR_RUN_ID
CLAIM_PATH = AUTHORING_ROOT / f"{CODEX_AUTHOR_RUN_ID}-attempt-claim.json"
RECEIPT_PATH = OUTPUT_DIR / "invocation-receipt.json"
APPROVAL_PATH = AUTHORING_ROOT / f"{CODEX_AUTHOR_RUN_ID}-owner-approval.json"

V2_RUN_ID = "llm-static-codex-primary-20260724-high-v2"
V2_FREEZE_PATH = AUTHORING_ROOT / "authoring-freeze-lock-codex-high-v2.json"
V2_FREEZE_FILE_SHA256 = (
    "63c15d4461d15472fe713f7b0ff446370124b31d4d2a912e5bf70dc1e5dfdf39"
)
V2_FREEZE_PAYLOAD_SHA256 = (
    "6aa283f7749dc58cb6461853849c702acd0b545dcf12fb0c622d83954e0d6c1a"
)
V2_APPROVAL_PATH = AUTHORING_ROOT / f"{V2_RUN_ID}-owner-approval.json"
V2_APPROVAL_FILE_SHA256 = (
    "410aadbe56513a484fc52f50455211655ed6c41cb4ef5e5f7ffd9b6acf1489ac"
)
V2_APPROVAL_PAYLOAD_SHA256 = (
    "55fb240de77e7c1f7956cb5e6694b11adb2cd996dbd11aa2b4481bd875f1e654"
)
V2_RETIREMENT_CLAIM_PATH = AUTHORING_ROOT / f"{V2_RUN_ID}-attempt-claim.json"
V2_OUTPUT_DIR = ROOT / "runs" / "formal-authoring" / V2_RUN_ID
V2_RECEIPT_PATH = V2_OUTPUT_DIR / "invocation-receipt.json"

V2_FROZEN_PATH_SHA256 = (
    "95af69bfda1e918fa71c5cf41330086da9626c7bbbf6f060e52095e87170cba6c"
)
V2_OBSERVED_PATH_SHA256 = (
    "a44b866eb86493b6991ca4ded97c26a1099c78422e2c249cd26dd1a94f15729f"
)


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed(payload: dict[str, object], field: str) -> bytes:
    return canonical_json_bytes(
        {**payload, field: sha256_bytes(canonical_json_bytes(payload))}
    )


def _binding(path: Path, content: bytes | None = None) -> dict[str, object]:
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "file_sha256": sha256_bytes(content) if content is not None else _sha(path),
    }


def _verify_v2_history() -> dict[str, object]:
    if _sha(V2_FREEZE_PATH) != V2_FREEZE_FILE_SHA256:
        raise ValueError("v2 freeze history drifted")
    freeze = json.loads(V2_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_payload_sha256") != V2_FREEZE_PAYLOAD_SHA256:
        raise ValueError("v2 freeze payload identity drifted")
    if _sha(V2_APPROVAL_PATH) != V2_APPROVAL_FILE_SHA256:
        raise ValueError("v2 owner approval history drifted")
    approval = json.loads(V2_APPROVAL_PATH.read_text(encoding="utf-8"))
    if approval.get("approval_payload_sha256") != V2_APPROVAL_PAYLOAD_SHA256:
        raise ValueError("v2 approval payload identity drifted")
    if os.path.lexists(V2_OUTPUT_DIR) or os.path.lexists(V2_RECEIPT_PATH):
        raise ValueError("v2 formal output unexpectedly exists")
    return approval


def _v2_preflight_incident_bytes() -> bytes:
    payload = {
        "schema_version": 1,
        "artifact_kind": "codex_authoring_preclaim_rejection",
        "incident_id": "codex-high-v2-environment-preclaim-20260724",
        "observed_on": "2026-07-24",
        "candidate_id": "authoring-codex-high-20260724-v2",
        "run_id": V2_RUN_ID,
        "status": "approved_but_not_invoked_runtime_preflight_unreplayable",
        "freeze_lock_file": V2_FREEZE_PATH.relative_to(ROOT).as_posix(),
        "freeze_lock_file_sha256": V2_FREEZE_FILE_SHA256,
        "freeze_payload_sha256": V2_FREEZE_PAYLOAD_SHA256,
        "owner_approval_file": V2_APPROVAL_PATH.relative_to(ROOT).as_posix(),
        "owner_approval_file_sha256": V2_APPROVAL_FILE_SHA256,
        "owner_approval_payload_sha256": V2_APPROVAL_PAYLOAD_SHA256,
        "failure_stage": "runtime_environment_validation_before_claim",
        "error_type": "CodexAuthoringContractError",
        "error_message": (
            "live Codex environment differs from the frozen value commitments"
        ),
        "frozen_parent_path_sha256": V2_FROZEN_PATH_SHA256,
        "observed_parent_path_sha256": V2_OBSERVED_PATH_SHA256,
        "sensitive_environment_values_recorded": False,
        "root_cause": (
            "the inherited PATH commitment included a Codex Desktop "
            "per-command .codex/tmp/arg0/codex-arg0* directory"
        ),
        "claim_created": False,
        "codex_exec_process_spawned": False,
        "inference_requested": False,
        "authorization_consumed_at_incident": False,
        "output_created": False,
        "receipt_created": False,
        "evidence_basis": [
            "runner control flow validates environment before claim creation",
            "runner control flow creates claim before the only subprocess launch",
            "post-failure claim output and receipt paths were absent",
        ],
        "disposition": (
            "preserve v2 bytes; require a new freeze and retire v2 before "
            "authorizing v3"
        ),
        "formal_provider_call_eligible": False,
    }
    return _signed(payload, "incident_payload_sha256")


def planned_v2_retirement_claim_bytes() -> bytes:
    payload = {
        "schema_version": 1,
        "artifact_kind": "codex_authorization_retirement_claim",
        "status": "authorization_retired_without_codex_process_launch",
        "run_id": V2_RUN_ID,
        "freeze_lock_file_sha256": V2_FREEZE_FILE_SHA256,
        "freeze_payload_sha256": V2_FREEZE_PAYLOAD_SHA256,
        "owner_approval_file_sha256": V2_APPROVAL_FILE_SHA256,
        "owner_approval_payload_sha256": V2_APPROVAL_PAYLOAD_SHA256,
        "preflight_incident_file": (
            V2_PREFLIGHT_INCIDENT_PATH.relative_to(ROOT).as_posix()
        ),
        "successor_candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "successor_run_id": CODEX_AUTHOR_RUN_ID,
        "reason": "v2 inherited PATH commitment is not replayable",
        "codex_exec_process_spawned": False,
        "inference_requested": False,
        "v2_authorization_reusable": False,
        "path_occupancy_prevents_v2_runner_launch": True,
    }
    return _signed(payload, "retirement_payload_sha256")


def _expected_files() -> dict[Path, bytes]:
    if (
        config.ASSISTANT_PROVIDER,
        config.ASSISTANT_MODEL,
        config.AUTHOR_PROVIDER,
        config.AUTHOR_MODEL,
        config.AUTHOR_REASONING_EFFORT,
        config.JUDGE_PROVIDER,
        config.JUDGE_MODEL,
        config.JUDGE_REASONING_EFFORT,
    ) != (
        "qwen",
        "qwen3-vl-flash-2026-01-22",
        "codex_internal",
        "gpt-5.6-sol",
        "high",
        "kimi",
        "kimi/kimi-k3",
        "max",
    ):
        raise ValueError("configured model-role selection drifted")
    if _sha(ROLE_SELECTION_PATH) != ROLE_SELECTION_SHA256:
        raise ValueError("model-role selection history drifted")
    _verify_v2_history()

    semantic_source = load_authoring_packet(
        SEMANTIC_SOURCE_PATH,
        expected_file_sha256=SEMANTIC_SOURCE_SHA256,
    )
    if (
        semantic_source.decoding.response_format
        != PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
        or semantic_source.compiler.compiler_version != "4.0.0"
    ):
        raise ValueError("semantic source is not the v5 content/compiler contract")
    access = load_codex_model_access_evidence(
        ACCESS_EVIDENCE_PATH,
        expected_file_sha256=ACCESS_EVIDENCE_SHA256,
    )
    if _sha(PROMPT_ISOLATION_EVIDENCE_PATH) != PROMPT_ISOLATION_EVIDENCE_SHA256:
        raise ValueError("Codex prompt-isolation evidence drifted")

    binary = freeze_codex_binary_identity(CODEX_FROZEN_BINARY_PATH)
    if (
        access.value.binary.get("bytes") != binary["bytes"]
        or access.value.binary.get("sha256") != binary["sha256"]
        or access.value.cli_version != binary["cli_version"]
    ):
        raise ValueError("absolute binary identity and access evidence disagree")
    binary_evidence = {
        "schema_version": 1,
        "artifact_kind": "codex_cli_absolute_binary_binding",
        "status": "verified_without_inference",
        "checked_on": "2026-07-24",
        "method": "absolute_regular_file_stable_size_and_sha256",
        "binary": binary,
        "model_access_evidence": _binding(ACCESS_EVIDENCE_PATH),
        "inference_request_performed": False,
        "credential_material_recorded": False,
    }
    binary_evidence_bytes = _signed(
        binary_evidence,
        "binary_evidence_payload_sha256",
    )

    packet = build_codex_authoring_input(
        semantic_source=semantic_source,
        semantic_source_packet_file_sha256=SEMANTIC_SOURCE_SHA256,
        model_access_evidence=access,
    )
    packet_bytes = packet.canonical_bytes()
    output_contract = build_codex_output_contract(
        authoring_input=packet,
        semantic_source=semantic_source,
    )
    schema_bytes = output_contract.json_schema_canonical_json.encode("utf-8")
    validate_codex_cli_output_schema(json.loads(schema_bytes))
    semantic_schema_bytes = canonical_json_bytes(
        authoring_content_json_schema(semantic_source)
    )
    if (
        output_contract.semantic_schema_sha256
        != sha256_bytes(semantic_schema_bytes)
    ):
        raise ValueError("Codex output contract lost the semantic schema")
    request = build_codex_authoring_request(
        authoring_input=packet,
        output_contract=output_contract,
    )
    request_bytes = request.canonical_bytes()
    stdin_bytes = render_codex_authoring_stdin(request)

    incident_bytes = _v2_preflight_incident_bytes()
    retirement_bytes = planned_v2_retirement_claim_bytes()
    protocol = {
        "schema_version": 1,
        "protocol_id": "authoring-codex-mediation-2026-07-24-v3",
        "status": "prospective_awaiting_final_owner_confirmation",
        "execution_type": CODEX_EXECUTION_TYPE,
        "evidence_tier": CODEX_EVIDENCE_TIER,
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "one_call_definition": "one_codex_exec_process_and_ephemeral_session",
        "input_policy": {
            "transport": "frozen_stdin_bytes",
            "working_directory": "new_empty_scratch_directory",
            "sandbox": "read-only",
            "user_config_loaded": False,
            "project_exec_rules_loaded": False,
            "session_persisted": False,
            "visible_tool_activity_allowed": False,
            "input_isolation": (
                "behaviorally_constrained_not_mechanically_proven"
            ),
            "known_limit": (
                "Codex base instructions and globally readable state cannot be "
                "mechanically excluded by the CLI flags"
            ),
            "repository_instruction_probe": _binding(
                PROMPT_ISOLATION_EVIDENCE_PATH
            ),
        },
        "runtime_policy": {
            "binary_resolution": "frozen_absolute_path_no_path_lookup",
            "parent_path_inherited": False,
            "parent_pathext_inherited": False,
            "child_path": "parent_directory_of_frozen_codex_binary",
            "child_temp": "new_external_invocation_temp_directory",
            "environment_built_from_empty_mapping": True,
            "binary_evidence": _binding(
                BINARY_EVIDENCE_PATH,
                binary_evidence_bytes,
            ),
        },
        "attempt_policy": {
            "claim_created_before_process_launch": True,
            "max_codex_exec_sessions": 1,
            "max_followups": 0,
            "max_repository_retries": 0,
            "max_repairs": 0,
            "max_fallbacks": 0,
            "invalid_or_failed_session_consumes_authorization": True,
            "backend_attempt_count": "unobservable",
            "platform_internal_retry": "unobservable",
        },
        "prior_authorization_policy": {
            "v2_approval_reusable": False,
            "v2_preflight_incident": _binding(
                V2_PREFLIGHT_INCIDENT_PATH,
                incident_bytes,
            ),
            "v2_retirement_required_before_v3_approval": True,
            "v2_retirement_claim_file": (
                V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
            ),
            "v2_retirement_claim_expected_file_sha256": sha256_bytes(
                retirement_bytes
            ),
            "v2_process_or_inference_occurred": False,
        },
        "result_policy": {
            "max_visible_agent_messages": 1,
            "max_accepted_final_outputs": 1,
            "strict_json_and_schema_validation": True,
            "trusted_compiler_validation": True,
            "human_review_required_before_bank_compile": True,
            "formal_provider_call_eligible": False,
            "formal_codex_session_eligible_field_required": True,
        },
        "comparability": {
            "llm_static_and_s1_share_surface_model_effort_and_session_budget": True,
            "token_and_cost_targets_retained_for_disclosure_only": True,
            "token_and_cost_enforcement": "unavailable_on_codex_cli",
        },
        "history_policy": {
            "qwen_v3_v4_negative_results_preserved": True,
            "qwen_v5_candidate_remains_unapproved_history": True,
            "qwen_locks_v5_v6_reusable": False,
            "qwen_claim_budget_or_run_id_reusable": False,
            "codex_v1_superseded_before_approval": True,
            "codex_v2_approved_but_not_invoked": True,
        },
    }
    protocol_bytes = _signed(protocol, "protocol_sha256")

    source_manifest_payload = build_codex_runtime_dependency_snapshot_v3(ROOT)
    source_manifest_bytes = _signed(
        source_manifest_payload,
        "source_manifest_payload_sha256",
    )
    environment_policy = build_codex_v3_environment_policy(
        binary=binary,
        parent_environment=os.environ,
    )
    runtime = {
        "schema_version": 2,
        "runtime_id": "codex-cli-author-runtime-20260724-v3",
        "status": "frozen",
        "platform": "windows-x86_64",
        "cli_version": access.value.cli_version,
        "binary": binary,
        "binary_evidence": _binding(
            BINARY_EVIDENCE_PATH,
            binary_evidence_bytes,
        ),
        "auth_mode": "chatgpt",
        "model_catalog": access.value.model_catalog.model_dump(mode="json"),
        "connectivity_preflight": access.value.connectivity,
        "inference_connectivity_verified": False,
        "command_shape": list(CODEX_COMMAND_SHAPE),
        "environment_policy": environment_policy,
        "sandbox_policy": {
            "working_directory": (
                "new_empty_system_temp_directory_outside_repository"
            ),
            "mode": "read-only",
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_project_exec_rules": True,
            "visible_tool_activity_allowed": False,
            "network_disabled": False,
            "input_isolation": (
                "behaviorally_constrained_not_mechanically_proven"
            ),
        },
        "runner": _binding(RUNNER_PATH),
        "base_runner": _binding(BASE_RUNNER_PATH),
        "contract_module": _binding(CONTRACT_MODULE_PATH),
        "base_contract_module": _binding(BASE_CONTRACT_MODULE_PATH),
        "source_manifest": _binding(
            SOURCE_MANIFEST_PATH,
            source_manifest_bytes,
        ),
        "prompt_isolation_evidence": _binding(
            PROMPT_ISOLATION_EVIDENCE_PATH
        ),
        "repository_retry_fallback_repair_followup": "forbidden",
    }
    runtime_bytes = _signed(runtime, "runtime_payload_sha256")

    paths = {
        "run_id": CODEX_AUTHOR_RUN_ID,
        "output_directory": OUTPUT_DIR.relative_to(ROOT).as_posix(),
        "claim_file": CLAIM_PATH.relative_to(ROOT).as_posix(),
        "receipt_file": RECEIPT_PATH.relative_to(ROOT).as_posix(),
        "approval_record_file": APPROVAL_PATH.relative_to(ROOT).as_posix(),
    }
    expected_partial = {
        PACKET_PATH: packet_bytes,
        SCHEMA_PATH: schema_bytes,
        REQUEST_PATH: request_bytes,
        STDIN_PATH: stdin_bytes,
        PROTOCOL_PATH: protocol_bytes,
        BINARY_EVIDENCE_PATH: binary_evidence_bytes,
        V2_PREFLIGHT_INCIDENT_PATH: incident_bytes,
        SOURCE_MANIFEST_PATH: source_manifest_bytes,
        RUNTIME_PATH: runtime_bytes,
    }
    freeze = {
        "schema_version": 1,
        "artifact_kind": "codex_authoring_freeze_candidate",
        "status": "frozen_candidate_awaiting_owner_confirmation",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "invocation_authorized": False,
        "approved_by": None,
        "formal_use": (
            "forbidden_until_an_owner_approval_record_accepts_this_exact_freeze_sha"
        ),
        "model": {
            "requested_model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "served_model": None,
            "served_revision": None,
        },
        "budget": packet.session_budget.model_dump(mode="json"),
        "authorization": {
            "owner_confirmation_required": True,
            "max_codex_exec_sessions": 1,
            "consume_on": "atomic_create_claim_before_process_launch",
            "retry_fallback_repair_followup_policy": "forbidden",
            "failure_or_invalid_output_consumes_authorization": True,
            "paths": paths,
        },
        "prior_authorization_retirement": {
            "v2_freeze": _binding(V2_FREEZE_PATH),
            "v2_owner_approval": _binding(V2_APPROVAL_PATH),
            "v2_preflight_incident": _binding(
                V2_PREFLIGHT_INCIDENT_PATH,
                incident_bytes,
            ),
            "retirement_claim_file": (
                V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
            ),
            "retirement_claim_expected_file_sha256": sha256_bytes(
                retirement_bytes
            ),
            "retirement_must_precede_v3_approval": True,
            "v2_approval_reusable": False,
        },
        "bindings": {
            "model_role_selection": _binding(ROLE_SELECTION_PATH),
            "protocol": _binding(PROTOCOL_PATH, protocol_bytes),
            "model_access_evidence": _binding(ACCESS_EVIDENCE_PATH),
            "binary_binding_evidence": _binding(
                BINARY_EVIDENCE_PATH,
                binary_evidence_bytes,
            ),
            "prompt_isolation_evidence": _binding(
                PROMPT_ISOLATION_EVIDENCE_PATH
            ),
            "semantic_source": _binding(SEMANTIC_SOURCE_PATH),
            "authoring_input": _binding(PACKET_PATH, packet_bytes),
            "output_schema": _binding(SCHEMA_PATH, schema_bytes),
            "canonical_request": _binding(REQUEST_PATH, request_bytes),
            "stdin_request": _binding(STDIN_PATH, stdin_bytes),
            "runtime_lock": _binding(RUNTIME_PATH, runtime_bytes),
            "runner": _binding(RUNNER_PATH),
            "base_runner": _binding(BASE_RUNNER_PATH),
            "contract_module": _binding(CONTRACT_MODULE_PATH),
            "base_contract_module": _binding(BASE_CONTRACT_MODULE_PATH),
            "builder": _binding(BUILDER_PATH),
            "approval_builder": _binding(APPROVAL_BUILDER_PATH),
            "source_manifest": _binding(
                SOURCE_MANIFEST_PATH,
                source_manifest_bytes,
            ),
            "v2_preflight_incident": _binding(
                V2_PREFLIGHT_INCIDENT_PATH,
                incident_bytes,
            ),
            "v2_owner_approval": _binding(V2_APPROVAL_PATH),
        },
        "evidence_boundary": {
            "formal_codex_session_eligible": (
                "decided_after_the_consumed_session"
            ),
            "formal_provider_call_eligible": False,
            "served_revision": "unavailable",
            "provider_request_id": "unavailable",
            "backend_attempt_count": "unobservable",
            "token_usage": "unavailable_or_non_authoritative",
            "cost": "unavailable_on_subscription_cli",
            "input_isolation": (
                "behaviorally_constrained_not_mechanically_proven"
            ),
            "preflight": (
                "catalog_auth_http_and_websocket_reachability_verified_by_doctor; "
                "absolute_binary_verified; no_inference_request_performed"
            ),
            "runtime_replay_scope": (
                "machine_bound_binary_and_python_dependency_closure; "
                "not_a_full_os_or_backend_snapshot"
            ),
        },
        "preserved_history": {
            "qwen_v3_v4_results_rewritten": False,
            "qwen_v5_candidate_superseded_for_author_selection": True,
            "qwen_locks_v5_v6_referenced": False,
            "codex_high_v1_candidate": (
                "superseded_before_owner_approval_after_binding_audit"
            ),
            "codex_high_v2_candidate": (
                "owner_approved_but_not_invoked_preclaim_environment_rejection"
            ),
        },
    }
    freeze_bytes = _signed(freeze, "freeze_payload_sha256")
    return {**expected_partial, FREEZE_PATH: freeze_bytes}


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = _expected_files()
    retirement_bytes = planned_v2_retirement_claim_bytes()
    if args.check:
        mismatches = [
            path.relative_to(ROOT).as_posix()
            for path, content in expected.items()
            if not path.is_file() or path.read_bytes() != content
        ]
        if os.path.lexists(V2_RETIREMENT_CLAIM_PATH) and (
            not V2_RETIREMENT_CLAIM_PATH.is_file()
            or V2_RETIREMENT_CLAIM_PATH.is_symlink()
            or V2_RETIREMENT_CLAIM_PATH.read_bytes() != retirement_bytes
        ):
            mismatches.append(
                V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
            )
        if mismatches:
            raise SystemExit(
                "Codex v3 authoring approval package drift: "
                + ", ".join(mismatches)
            )
    else:
        existing = [
            path.relative_to(ROOT).as_posix()
            for path in expected
            if os.path.lexists(path)
        ]
        existing.extend(
            path.relative_to(ROOT).as_posix()
            for path in (OUTPUT_DIR, CLAIM_PATH, RECEIPT_PATH, APPROVAL_PATH)
            if os.path.lexists(path)
        )
        if os.path.lexists(V2_RETIREMENT_CLAIM_PATH):
            existing.append(
                V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
            )
        if existing:
            raise SystemExit(
                "Codex v3 approval package is create-only; existing: "
                + ", ".join(existing)
            )
        for path, content in expected.items():
            atomic_create_file(path, content)
    freeze = json.loads(expected[FREEZE_PATH])
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "candidate_id": freeze["candidate_id"],
                "freeze_payload_sha256": freeze["freeze_payload_sha256"],
                "freeze_file_sha256": sha256_bytes(expected[FREEZE_PATH]),
                "invocation_authorized": False,
                "run_id": CODEX_AUTHOR_RUN_ID,
                "v2_retirement_pending": not os.path.lexists(
                    V2_RETIREMENT_CLAIM_PATH
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
