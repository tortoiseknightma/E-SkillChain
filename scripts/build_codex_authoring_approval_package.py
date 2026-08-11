"""Build the create-only Codex/high authoring approval package.

The generated freeze is immutable but does not authorize inference.  A separate
owner-approval record, checked by ``run_codex_authoring.py``, is required before
the one create-only attempt claim can be consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from skillchain import config
from skillchain.codex_authoring import (
    CODEX_AUTHOR_RUN_ID,
    CODEX_COMMAND_SHAPE,
    CODEX_ENV_ALLOWLIST,
    CODEX_EVIDENCE_TIER,
    CODEX_EXECUTION_TYPE,
    build_codex_authoring_input,
    build_codex_authoring_request,
    build_codex_output_contract,
    build_codex_runtime_dependency_snapshot,
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
DEPLOY_ROOT = ROOT / "deploy" / "authoring" / "locks-codex-v2"

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
RUNNER_PATH = ROOT / "scripts" / "run_codex_authoring.py"
CONTRACT_MODULE_PATH = ROOT / "src" / "skillchain" / "codex_authoring.py"
BUILDER_PATH = Path(__file__).resolve()
APPROVAL_BUILDER_PATH = ROOT / "scripts" / "approve_codex_authoring.py"

ROLE_SELECTION_PATH = AUTHORING_ROOT / "model-role-selection-v3.json"
PACKET_PATH = AUTHORING_ROOT / "authoring-packet-codex-high-v2.json"
SCHEMA_PATH = AUTHORING_ROOT / "authoring-content-schema-codex-high-v2.json"
REQUEST_PATH = AUTHORING_ROOT / "authoring-request-codex-high-v2.json"
STDIN_PATH = AUTHORING_ROOT / "authoring-stdin-codex-high-v2.txt"
PROTOCOL_PATH = AUTHORING_ROOT / "authoring-codex-mediation-protocol-v2.json"
SOURCE_MANIFEST_PATH = DEPLOY_ROOT / "source-manifest.json"
RUNTIME_PATH = DEPLOY_ROOT / "runtime-lock.json"
FREEZE_PATH = AUTHORING_ROOT / "authoring-freeze-lock-codex-high-v2.json"

OUTPUT_DIR = ROOT / "runs" / "formal-authoring" / CODEX_AUTHOR_RUN_ID
CLAIM_PATH = AUTHORING_ROOT / f"{CODEX_AUTHOR_RUN_ID}-attempt-claim.json"
RECEIPT_PATH = OUTPUT_DIR / "invocation-receipt.json"
APPROVAL_PATH = AUTHORING_ROOT / f"{CODEX_AUTHOR_RUN_ID}-owner-approval.json"


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


def _environment_value_commitments() -> dict[str, str]:
    return {
        name: sha256_bytes(value.encode("utf-8"))
        for name, value in sorted(os.environ.items())
        if name in CODEX_ENV_ALLOWLIST
    }


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
    projected_schema = json.loads(schema_bytes)
    validate_codex_cli_output_schema(projected_schema)
    semantic_schema_bytes = canonical_json_bytes(
        authoring_content_json_schema(semantic_source)
    )
    if (
        output_contract.semantic_schema_sha256
        != sha256_bytes(semantic_schema_bytes)
    ):
        raise ValueError("Codex output contract lost the trusted semantic schema")
    request = build_codex_authoring_request(
        authoring_input=packet,
        output_contract=output_contract,
    )
    request_bytes = request.canonical_bytes()
    stdin_bytes = render_codex_authoring_stdin(request)

    role_selection = {
        "schema_version": 3,
        "artifact_kind": "model_role_selection",
        "status": "owner_selected_models",
        "selected_on": "2026-07-24",
        "assistant": {
            "provider": "qwen",
            "transport": "dashscope_openai_compatible_api",
            "model": "qwen3-vl-flash-2026-01-22",
            "revision": "2026-01-22",
            "thinking": False,
            "formal_runtime_lock_status": "pending",
        },
        "author": {
            "provider": "codex_internal",
            "transport": "codex_cli",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "execution_type": CODEX_EXECUTION_TYPE,
            "evidence_tier": CODEX_EVIDENCE_TIER,
            "formal_provider_call_eligible": False,
            "formal_session_authorization_status": (
                "awaiting_final_owner_confirmation"
            ),
        },
        "offline_judge": {
            "provider": "kimi",
            "transport": "dashscope_openai_compatible_api",
            "model": "kimi/kimi-k3",
            "reasoning_effort": "max",
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": None,
            "runner_status": "not_implemented",
            "visual_transport_status": (
                "blocked_until_audited_public_url_asset_binding_exists"
            ),
        },
        "feedback_evaluator": {
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "isolation_requirement": (
                "separate_family_prompt_cache_and_artifact_namespace_from_final_judge"
            ),
        },
        "label_synthesis": {
            "vision": {
                "configuration_namespace": "LABEL_VISION_SYNTH",
                "provider": config.LABEL_VISION_SYNTH_PROVIDER,
                "model": config.LABEL_VISION_SYNTH_MODEL,
                "revision_status": "moving_alias_not_formal_replay_eligible",
            },
            "text_review": {
                "configuration_namespace": "LABEL_TEXT_REVIEW",
                "provider": config.LABEL_TEXT_REVIEW_PROVIDER,
                "model": config.LABEL_TEXT_REVIEW_MODEL,
                "revision_status": "moving_alias_not_formal_replay_eligible",
            },
            "coupled_to_assistant": False,
            "eligible_for_feedback_or_final_judging": False,
        },
    }
    role_bytes = _signed(role_selection, "selection_sha256")

    protocol = {
        "schema_version": 1,
        "protocol_id": "authoring-codex-mediation-2026-07-24-v2",
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
        },
    }
    protocol_bytes = _signed(protocol, "protocol_sha256")

    source_manifest_payload = build_codex_runtime_dependency_snapshot(ROOT)
    source_manifest_bytes = _signed(
        source_manifest_payload,
        "source_manifest_payload_sha256",
    )
    runtime = {
        "schema_version": 1,
        "runtime_id": "codex-cli-author-runtime-20260724-v2",
        "status": "frozen",
        "platform": "windows-x86_64",
        "cli_version": access.value.cli_version,
        "binary": access.value.binary,
        "auth_mode": "chatgpt",
        "model_catalog": access.value.model_catalog.model_dump(mode="json"),
        "connectivity_preflight": access.value.connectivity,
            "inference_connectivity_verified": False,
        "command_shape": list(CODEX_COMMAND_SHAPE),
        "environment_policy": {
            "policy_version": "codex-author-env-allowlist-v1",
            "allowed_names": list(CODEX_ENV_ALLOWLIST),
            "all_other_environment_variables_removed": True,
            "present_value_sha256": _environment_value_commitments(),
            "credential_values_recorded": False,
        },
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
        "contract_module": _binding(CONTRACT_MODULE_PATH),
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
        ROLE_SELECTION_PATH: role_bytes,
        PACKET_PATH: packet_bytes,
        SCHEMA_PATH: schema_bytes,
        REQUEST_PATH: request_bytes,
        STDIN_PATH: stdin_bytes,
        PROTOCOL_PATH: protocol_bytes,
        SOURCE_MANIFEST_PATH: source_manifest_bytes,
        RUNTIME_PATH: runtime_bytes,
    }
    freeze = {
        "schema_version": 1,
        "artifact_kind": "codex_authoring_freeze_candidate",
        "status": "frozen_candidate_awaiting_owner_confirmation",
        "candidate_id": "authoring-codex-high-20260724-v2",
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
        "bindings": {
            "model_role_selection": _binding(
                ROLE_SELECTION_PATH, role_bytes
            ),
            "protocol": _binding(PROTOCOL_PATH, protocol_bytes),
            "model_access_evidence": _binding(ACCESS_EVIDENCE_PATH),
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
            "contract_module": _binding(CONTRACT_MODULE_PATH),
            "builder": _binding(BUILDER_PATH),
            "approval_builder": _binding(APPROVAL_BUILDER_PATH),
            "source_manifest": _binding(
                SOURCE_MANIFEST_PATH,
                source_manifest_bytes,
            ),
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
                "no_inference_request_performed"
            ),
        },
        "preserved_history": {
            "qwen_v3_v4_results_rewritten": False,
            "qwen_v5_candidate_superseded_for_author_selection": True,
            "qwen_locks_v5_v6_referenced": False,
            "codex_high_v1_candidate": (
                "superseded_before_owner_approval_after_binding_audit"
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
    if args.check:
        mismatches = [
            path.relative_to(ROOT).as_posix()
            for path, content in expected.items()
            if not path.is_file() or path.read_bytes() != content
        ]
        if mismatches:
            raise SystemExit(
                "Codex authoring approval package drift: " + ", ".join(mismatches)
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
        if existing:
            raise SystemExit(
                "Codex authoring approval package is create-only; existing: "
                + ", ".join(existing)
            )
        for path, content in expected.items():
            atomic_create_file(path, content)
    freeze = json.loads(expected[FREEZE_PATH])
    freeze_file_sha256 = sha256_bytes(expected[FREEZE_PATH])
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "candidate_id": freeze["candidate_id"],
                "freeze_payload_sha256": freeze["freeze_payload_sha256"],
                "freeze_file_sha256": freeze_file_sha256,
                "invocation_authorized": False,
                "run_id": CODEX_AUTHOR_RUN_ID,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
