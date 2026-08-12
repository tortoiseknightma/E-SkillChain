"""Build the create-only Codex/high v4 authoring approval package.

v4 preserves the v3 deterministic environment and adds a pre-approved,
conditional terminal guard.  This builder never creates the guard, approval,
claim, output, or v2 retirement record and never performs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from skillchain import config
from skillchain.codex_authoring_v4 import (
    CODEX_AUTHOR_CANDIDATE_ID,
    CODEX_AUTHOR_RUN_ID,
    CODEX_COMMAND_SHAPE,
    CODEX_EVIDENCE_TIER,
    CODEX_EXECUTION_TYPE,
    CODEX_FROZEN_BINARY_PATH,
    build_codex_authoring_input,
    build_codex_authoring_request,
    build_codex_output_contract,
    build_codex_runtime_dependency_snapshot_v4,
    build_codex_v4_environment_policy,
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
from skillchain.tools.serialization import (
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)

import scripts.build_codex_authoring_approval_package_v3 as v3_builder


ROOT = Path(__file__).resolve().parents[1]
AUTHORING_ROOT = ROOT / "specs" / "authoring"
DEPLOY_ROOT = ROOT / "deploy" / "authoring" / "locks-codex-v4"

SEMANTIC_SOURCE_PATH = v3_builder.SEMANTIC_SOURCE_PATH
SEMANTIC_SOURCE_SHA256 = v3_builder.SEMANTIC_SOURCE_SHA256
ACCESS_EVIDENCE_PATH = v3_builder.ACCESS_EVIDENCE_PATH
ACCESS_EVIDENCE_SHA256 = v3_builder.ACCESS_EVIDENCE_SHA256
PROMPT_ISOLATION_EVIDENCE_PATH = v3_builder.PROMPT_ISOLATION_EVIDENCE_PATH
PROMPT_ISOLATION_EVIDENCE_SHA256 = (
    v3_builder.PROMPT_ISOLATION_EVIDENCE_SHA256
)
ROLE_SELECTION_PATH = v3_builder.ROLE_SELECTION_PATH
ROLE_SELECTION_SHA256 = v3_builder.ROLE_SELECTION_SHA256
V2_PREFLIGHT_INCIDENT_PATH = v3_builder.V2_PREFLIGHT_INCIDENT_PATH

RUNNER_PATH = ROOT / "scripts" / "run_codex_authoring_v4.py"
V3_RUNNER_PATH = ROOT / "scripts" / "run_codex_authoring_v3.py"
BASE_RUNNER_PATH = ROOT / "scripts" / "run_codex_authoring.py"
CONTRACT_MODULE_PATH = ROOT / "src" / "skillchain" / "codex_authoring_v4.py"
V3_CONTRACT_MODULE_PATH = ROOT / "src" / "skillchain" / "codex_authoring_v3.py"
BASE_CONTRACT_MODULE_PATH = ROOT / "src" / "skillchain" / "codex_authoring.py"
V3_BUILDER_PATH = ROOT / "scripts" / "build_codex_authoring_approval_package_v3.py"
BUILDER_PATH = Path(__file__).resolve()
APPROVAL_BUILDER_PATH = ROOT / "scripts" / "approve_codex_authoring_v4.py"

PACKET_PATH = AUTHORING_ROOT / "authoring-packet-codex-high-v4.json"
SCHEMA_PATH = AUTHORING_ROOT / "authoring-content-schema-codex-high-v4.json"
REQUEST_PATH = AUTHORING_ROOT / "authoring-request-codex-high-v4.json"
STDIN_PATH = AUTHORING_ROOT / "authoring-stdin-codex-high-v4.txt"
PROTOCOL_PATH = AUTHORING_ROOT / "authoring-codex-mediation-protocol-v4.json"
BINARY_EVIDENCE_PATH = (
    AUTHORING_ROOT / "codex-cli-absolute-binary-evidence-v4.json"
)
V3_AUDIT_INCIDENT_PATH = (
    AUTHORING_ROOT / "codex-high-v3-preapproval-runner-audit-rejection-v1.json"
)
SOURCE_MANIFEST_PATH = DEPLOY_ROOT / "source-manifest.json"
RUNTIME_PATH = DEPLOY_ROOT / "runtime-lock.json"
FREEZE_PATH = AUTHORING_ROOT / "authoring-freeze-lock-codex-high-v4.json"

OUTPUT_DIR = ROOT / "runs" / "formal-authoring" / CODEX_AUTHOR_RUN_ID
CLAIM_PATH = AUTHORING_ROOT / f"{CODEX_AUTHOR_RUN_ID}-attempt-claim.json"
RECEIPT_PATH = OUTPUT_DIR / "invocation-receipt.json"
TERMINAL_GUARD_PATH = (
    AUTHORING_ROOT / f"{CODEX_AUTHOR_RUN_ID}-terminal-guard.json"
)
APPROVAL_PATH = AUTHORING_ROOT / f"{CODEX_AUTHOR_RUN_ID}-owner-approval.json"

V2_RUN_ID = v3_builder.V2_RUN_ID
V2_FREEZE_PATH = v3_builder.V2_FREEZE_PATH
V2_FREEZE_FILE_SHA256 = v3_builder.V2_FREEZE_FILE_SHA256
V2_FREEZE_PAYLOAD_SHA256 = v3_builder.V2_FREEZE_PAYLOAD_SHA256
V2_APPROVAL_PATH = v3_builder.V2_APPROVAL_PATH
V2_APPROVAL_FILE_SHA256 = v3_builder.V2_APPROVAL_FILE_SHA256
V2_APPROVAL_PAYLOAD_SHA256 = v3_builder.V2_APPROVAL_PAYLOAD_SHA256
V2_RETIREMENT_CLAIM_PATH = v3_builder.V2_RETIREMENT_CLAIM_PATH
V2_OUTPUT_DIR = v3_builder.V2_OUTPUT_DIR
V2_RECEIPT_PATH = v3_builder.V2_RECEIPT_PATH

V3_CANDIDATE_ID = "authoring-codex-high-20260724-v3"
V3_RUN_ID = "llm-static-codex-primary-20260724-high-v3"
V3_FREEZE_PATH = (
    AUTHORING_ROOT / "authoring-freeze-lock-codex-high-v3.json"
)
V3_FREEZE_FILE_SHA256 = (
    "e4944a453e81ce550b626a9a76ada1a48b4916587585c9343a2185df11d44b8d"
)
V3_FREEZE_PAYLOAD_SHA256 = (
    "b48a60c0a130ecd542f05896dfdfc56ad4b01920d4b1759ccd5eed34ffbdc2e2"
)
V3_APPROVAL_PATH = AUTHORING_ROOT / f"{V3_RUN_ID}-owner-approval.json"
V3_CLAIM_PATH = AUTHORING_ROOT / f"{V3_RUN_ID}-attempt-claim.json"
V3_OUTPUT_DIR = ROOT / "runs" / "formal-authoring" / V3_RUN_ID
V3_RECEIPT_PATH = V3_OUTPUT_DIR / "invocation-receipt.json"


def _sha(path: Path) -> str:
    try:
        content = read_stable_regular_file(path, label=f"required file {path}")
    except Exception as error:
        raise ValueError(f"required regular file is missing: {path}") from error
    return hashlib.sha256(content).hexdigest()


def _signed(payload: dict[str, object], field: str) -> bytes:
    return canonical_json_bytes(
        {**payload, field: sha256_bytes(canonical_json_bytes(payload))}
    )


def _binding(path: Path, content: bytes | None = None) -> dict[str, object]:
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "file_sha256": sha256_bytes(content) if content is not None else _sha(path),
    }


def _verify_historical_freeze(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
    label: str,
) -> dict[str, object]:
    content = read_stable_regular_file(path, label=label)
    if sha256_bytes(content) != expected_file_sha256:
        raise ValueError(f"{label} file history drifted")
    freeze = json.loads(content)
    unsigned = dict(freeze)
    observed_payload_sha = unsigned.pop("freeze_payload_sha256", None)
    if (
        observed_payload_sha != expected_payload_sha256
        or observed_payload_sha != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise ValueError(f"{label} payload history drifted")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError(f"{label} bindings are missing")
    for name, item in bindings.items():
        if not isinstance(item, dict) or set(item) != {"file", "file_sha256"}:
            raise ValueError(f"{label} binding is invalid: {name}")
        relative = item.get("file")
        digest = item.get("file_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"{label} binding values are invalid: {name}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{label} binding escapes root: {name}")
        target = ROOT / relative_path
        resolved = target.resolve(strict=True)
        if (
            target.is_symlink()
            or ROOT.resolve() not in resolved.parents
            or _sha(resolved) != digest
        ):
            raise ValueError(f"{label} bound history drifted: {name}")
    return freeze


def verify_v2_v3_bound_history() -> None:
    _verify_historical_freeze(
        V2_FREEZE_PATH,
        expected_file_sha256=V2_FREEZE_FILE_SHA256,
        expected_payload_sha256=V2_FREEZE_PAYLOAD_SHA256,
        label="v2 freeze",
    )
    v2_approval = json.loads(V2_APPROVAL_PATH.read_text(encoding="utf-8"))
    unsigned_approval = dict(v2_approval)
    approval_payload_sha = unsigned_approval.pop("approval_payload_sha256", None)
    if (
        _sha(V2_APPROVAL_PATH) != V2_APPROVAL_FILE_SHA256
        or approval_payload_sha != V2_APPROVAL_PAYLOAD_SHA256
        or approval_payload_sha
        != sha256_bytes(canonical_json_bytes(unsigned_approval))
    ):
        raise ValueError("v2 owner approval history drifted")
    _verify_historical_freeze(
        V3_FREEZE_PATH,
        expected_file_sha256=V3_FREEZE_FILE_SHA256,
        expected_payload_sha256=V3_FREEZE_PAYLOAD_SHA256,
        label="v3 freeze",
    )


def _verified_v2_incident() -> tuple[bytes, dict[str, object]]:
    expected = v3_builder._v2_preflight_incident_bytes()
    if (
        _sha(V2_PREFLIGHT_INCIDENT_PATH) != sha256_bytes(expected)
        or V2_PREFLIGHT_INCIDENT_PATH.read_bytes() != expected
    ):
        raise ValueError("v2 preflight incident history drifted")
    incident = json.loads(expected)
    unsigned = dict(incident)
    observed = unsigned.pop("incident_payload_sha256", None)
    if (
        observed != sha256_bytes(canonical_json_bytes(unsigned))
        or incident.get("claim_created") is not False
        or incident.get("codex_exec_process_spawned") is not False
        or incident.get("inference_requested") is not False
    ):
        raise ValueError("v2 preflight incident is not valid non-inference evidence")
    return expected, incident


def _verify_history() -> tuple[bytes, dict[str, object]]:
    verify_v2_v3_bound_history()
    incident_bytes, incident = _verified_v2_incident()
    occupied = [
        path
        for path in (
            V3_APPROVAL_PATH,
            V3_CLAIM_PATH,
            V3_OUTPUT_DIR,
            V3_RECEIPT_PATH,
            V2_OUTPUT_DIR,
            V2_RECEIPT_PATH,
            V2_RETIREMENT_CLAIM_PATH,
        )
        if os.path.lexists(path)
    ]
    if occupied:
        raise ValueError(
            "pre-v4 history unexpectedly occupied: "
            + ", ".join(path.relative_to(ROOT).as_posix() for path in occupied)
        )
    return incident_bytes, incident


def v3_audit_incident_bytes() -> bytes:
    payload = {
        "schema_version": 1,
        "artifact_kind": "codex_authoring_preapproval_audit_rejection",
        "incident_id": "codex-high-v3-postclaim-terminal-gap-20260724",
        "observed_on": "2026-07-24",
        "candidate_id": V3_CANDIDATE_ID,
        "run_id": V3_RUN_ID,
        "status": "superseded_before_owner_approval_and_inference",
        "freeze_lock_file": V3_FREEZE_PATH.relative_to(ROOT).as_posix(),
        "freeze_lock_file_sha256": V3_FREEZE_FILE_SHA256,
        "freeze_payload_sha256": V3_FREEZE_PAYLOAD_SHA256,
        "failure_class": "claim_ambiguous_success_and_terminal_receipt_gap",
        "root_cause": (
            "claim hard-link commit could be followed by a cleanup exception "
            "that the runner treated as an uncommitted claim; post-claim "
            "initialization and canonical receipt publication also lacked an "
            "independent fail-closed terminal guard"
        ),
        "severity": "P0",
        "owner_approval_created": False,
        "claim_created": False,
        "codex_exec_process_spawned": False,
        "inference_requested": False,
        "output_created": False,
        "receipt_created": False,
        "authorization_consumed": False,
        "v3_bytes_preserved": True,
        "disposition": (
            "preserve v3 as unapproved history; require a fresh v4 namespace "
            "with a pre-claim conditional terminal guard"
        ),
        "formal_provider_call_eligible": False,
    }
    return _signed(payload, "incident_payload_sha256")


def planned_v2_retirement_claim_bytes() -> bytes:
    incident_bytes, incident = _verified_v2_incident()
    payload = {
        "schema_version": 2,
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
        "preflight_incident_file_sha256": sha256_bytes(incident_bytes),
        "preflight_incident_payload_sha256": incident[
            "incident_payload_sha256"
        ],
        "successor_candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "successor_run_id": CODEX_AUTHOR_RUN_ID,
        "reason": "v2 inherited PATH commitment is not replayable",
        "codex_exec_process_spawned": False,
        "inference_requested": False,
        "v2_authorization_reusable": False,
        "path_occupancy_prevents_v2_runner_launch": True,
    }
    return _signed(payload, "retirement_payload_sha256")


def planned_terminal_guard_bytes() -> bytes:
    payload = {
        "schema_version": 1,
        "artifact_kind": "codex_authoring_conditional_terminal_guard",
        "status": "inactive_until_exact_claim_commits",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "run_id": CODEX_AUTHOR_RUN_ID,
        "activation_claim_file": CLAIM_PATH.relative_to(ROOT).as_posix(),
        "canonical_output_directory": OUTPUT_DIR.relative_to(ROOT).as_posix(),
        "canonical_receipt_file": RECEIPT_PATH.relative_to(ROOT).as_posix(),
        "active_status": "authorization_consumed_result_rejected",
        "active_formal_codex_session_eligible": False,
        "active_process_launch_state": "unknown",
        "activation_requires_claim_to_bind_this_guard_sha256": True,
        "canonical_override_requires_full_bundle_validation": True,
        "model_retry_fallback_repair_followup_authorized": False,
        "storage_disaster_guarantee": "not_claimed",
    }
    return _signed(payload, "terminal_guard_payload_sha256")


def _expected_files() -> dict[Path, bytes]:
    if (
        config.ASSISTANT_PROVIDER,
        config.LEGACY_PORTFOLIO_ASSISTANT_MODEL,
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
    v2_incident_bytes, v2_incident = _verify_history()

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
    if output_contract.semantic_schema_sha256 != sha256_bytes(
        semantic_schema_bytes
    ):
        raise ValueError("Codex output contract lost the semantic schema")
    request = build_codex_authoring_request(
        authoring_input=packet,
        output_contract=output_contract,
    )
    request_bytes = request.canonical_bytes()
    stdin_bytes = render_codex_authoring_stdin(request)

    v3_incident_bytes = v3_audit_incident_bytes()
    retirement_bytes = planned_v2_retirement_claim_bytes()
    guard_bytes = planned_terminal_guard_bytes()
    protocol = {
        "schema_version": 1,
        "protocol_id": "authoring-codex-mediation-2026-07-24-v4",
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
            "terminal_guard_created_during_approval_before_claim": True,
            "claim_contains_256_bit_claimant_nonce": True,
            "claim_created_before_process_launch": True,
            "ambiguous_claim_commit_forbids_process_launch": True,
            "max_codex_exec_sessions": 1,
            "max_followups": 0,
            "max_repository_retries": 0,
            "max_repairs": 0,
            "max_fallbacks": 0,
            "invalid_or_failed_session_consumes_authorization": True,
            "backend_attempt_count": "unobservable",
            "platform_internal_retry": "unobservable",
        },
        "terminal_evidence_policy": {
            "conditional_terminal_guard": {
                "file": TERMINAL_GUARD_PATH.relative_to(ROOT).as_posix(),
                "expected_file_sha256": sha256_bytes(guard_bytes),
            },
            "guard_activates_only_for_exact_claim_binding_its_sha256": True,
            "canonical_bundle_overrides_guard_only_after_exact_validation": True,
            "canonical_publish_ambiguous_success_is_reread_and_validated": True,
            "whole_storage_medium_failure_guarantee": "not_claimed",
        },
        "prior_authorization_policy": {
            "v2_approval_reusable": False,
            "v2_preflight_incident": _binding(
                V2_PREFLIGHT_INCIDENT_PATH,
                v2_incident_bytes,
            ),
            "v2_preflight_incident_payload_sha256": v2_incident[
                "incident_payload_sha256"
            ],
            "v2_retirement_required_before_v4_approval": True,
            "v2_retirement_claim_file": (
                V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
            ),
            "v2_retirement_claim_expected_file_sha256": sha256_bytes(
                retirement_bytes
            ),
            "v2_process_or_inference_occurred": False,
            "v3_approval_or_inference_occurred": False,
            "v3_preapproval_audit_incident": _binding(
                V3_AUDIT_INCIDENT_PATH,
                v3_incident_bytes,
            ),
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
            "codex_v3_superseded_before_approval_or_inference": True,
        },
    }
    protocol_bytes = _signed(protocol, "protocol_sha256")

    source_manifest_payload = build_codex_runtime_dependency_snapshot_v4(ROOT)
    source_manifest_bytes = _signed(
        source_manifest_payload,
        "source_manifest_payload_sha256",
    )
    environment_policy = build_codex_v4_environment_policy(
        binary=binary,
        parent_environment=os.environ,
    )
    runtime = {
        "schema_version": 3,
        "runtime_id": "codex-cli-author-runtime-20260724-v4",
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
        "v3_runner": _binding(V3_RUNNER_PATH),
        "base_runner": _binding(BASE_RUNNER_PATH),
        "contract_module": _binding(CONTRACT_MODULE_PATH),
        "v3_contract_module": _binding(V3_CONTRACT_MODULE_PATH),
        "base_contract_module": _binding(BASE_CONTRACT_MODULE_PATH),
        "source_manifest": _binding(
            SOURCE_MANIFEST_PATH,
            source_manifest_bytes,
        ),
        "prompt_isolation_evidence": _binding(
            PROMPT_ISOLATION_EVIDENCE_PATH
        ),
        "terminal_guard_policy": {
            "created_during_owner_approval": True,
            "file": TERMINAL_GUARD_PATH.relative_to(ROOT).as_posix(),
            "expected_file_sha256": sha256_bytes(guard_bytes),
        },
        "repository_retry_fallback_repair_followup": "forbidden",
    }
    runtime_bytes = _signed(runtime, "runtime_payload_sha256")

    paths = {
        "run_id": CODEX_AUTHOR_RUN_ID,
        "output_directory": OUTPUT_DIR.relative_to(ROOT).as_posix(),
        "claim_file": CLAIM_PATH.relative_to(ROOT).as_posix(),
        "receipt_file": RECEIPT_PATH.relative_to(ROOT).as_posix(),
        "terminal_guard_file": TERMINAL_GUARD_PATH.relative_to(ROOT).as_posix(),
        "approval_record_file": APPROVAL_PATH.relative_to(ROOT).as_posix(),
    }
    expected_partial = {
        PACKET_PATH: packet_bytes,
        SCHEMA_PATH: schema_bytes,
        REQUEST_PATH: request_bytes,
        STDIN_PATH: stdin_bytes,
        PROTOCOL_PATH: protocol_bytes,
        BINARY_EVIDENCE_PATH: binary_evidence_bytes,
        V3_AUDIT_INCIDENT_PATH: v3_incident_bytes,
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
            "consume_on": "classified_exact_nonce_claim_before_process_launch",
            "retry_fallback_repair_followup_policy": "forbidden",
            "failure_or_invalid_output_consumes_authorization": True,
            "paths": paths,
        },
        "terminal_guard": {
            "created_only_after_owner_confirmation_before_v4_approval": True,
            "file": TERMINAL_GUARD_PATH.relative_to(ROOT).as_posix(),
            "expected_file_sha256": sha256_bytes(guard_bytes),
            "activation": "exact_claim_binds_guard_file_sha256",
            "canonical_override": "exact_full_bundle_validation_only",
        },
        "prior_authorization_retirement": {
            "v2_freeze": _binding(V2_FREEZE_PATH),
            "v2_owner_approval": _binding(V2_APPROVAL_PATH),
            "v2_preflight_incident": _binding(
                V2_PREFLIGHT_INCIDENT_PATH,
                v2_incident_bytes,
            ),
            "v2_preflight_incident_payload_sha256": v2_incident[
                "incident_payload_sha256"
            ],
            "retirement_claim_file": (
                V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
            ),
            "retirement_claim_expected_file_sha256": sha256_bytes(
                retirement_bytes
            ),
            "retirement_must_precede_v4_approval": True,
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
            "v3_runner": _binding(V3_RUNNER_PATH),
            "base_runner": _binding(BASE_RUNNER_PATH),
            "contract_module": _binding(CONTRACT_MODULE_PATH),
            "v3_contract_module": _binding(V3_CONTRACT_MODULE_PATH),
            "base_contract_module": _binding(BASE_CONTRACT_MODULE_PATH),
            "v3_builder": _binding(V3_BUILDER_PATH),
            "builder": _binding(BUILDER_PATH),
            "approval_builder": _binding(APPROVAL_BUILDER_PATH),
            "source_manifest": _binding(
                SOURCE_MANIFEST_PATH,
                source_manifest_bytes,
            ),
            "v2_preflight_incident": _binding(
                V2_PREFLIGHT_INCIDENT_PATH,
                v2_incident_bytes,
            ),
            "v2_owner_approval": _binding(V2_APPROVAL_PATH),
            "v3_preapproval_audit_incident": _binding(
                V3_AUDIT_INCIDENT_PATH,
                v3_incident_bytes,
            ),
            "v3_freeze": _binding(V3_FREEZE_PATH),
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
            "terminal_guard_scope": (
                "process-level exceptions and ambiguous create/publish effects; "
                "not total storage failure, power loss, or privileged tampering"
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
            "codex_high_v3_candidate": (
                "superseded_before_owner_approval_or_inference_after_p0_audit"
            ),
        },
    }
    freeze_bytes = _signed(freeze, "freeze_payload_sha256")
    return {**expected_partial, FREEZE_PATH: freeze_bytes}


def _authority_paths() -> tuple[Path, ...]:
    return (
        APPROVAL_PATH,
        TERMINAL_GUARD_PATH,
        CLAIM_PATH,
        OUTPUT_DIR,
        RECEIPT_PATH,
        V2_RETIREMENT_CLAIM_PATH,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--check-candidate", action="store_true")
    args = parser.parse_args()
    expected = _expected_files()
    if args.check or args.check_candidate:
        mismatches = [
            path.relative_to(ROOT).as_posix()
            for path, content in expected.items()
            if (
                not path.is_file()
                or path.is_symlink()
                or path.read_bytes() != content
            )
        ]
        occupied = [
            path.relative_to(ROOT).as_posix()
            for path in _authority_paths()
            if os.path.lexists(path)
        ]
        if mismatches or occupied:
            details = []
            if mismatches:
                details.append("drift=" + ",".join(mismatches))
            if occupied:
                details.append("authority_paths_present=" + ",".join(occupied))
            raise SystemExit(
                "Codex v4 pre-approval candidate check failed: "
                + "; ".join(details)
            )
    else:
        existing = [
            path.relative_to(ROOT).as_posix()
            for path in (*expected, *_authority_paths())
            if os.path.lexists(path)
        ]
        if existing:
            raise SystemExit(
                "Codex v4 approval package is create-only; existing: "
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
                "terminal_guard_created": False,
                "v2_retirement_pending": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
