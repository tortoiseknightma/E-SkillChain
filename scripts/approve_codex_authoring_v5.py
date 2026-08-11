"""Issue one-session authority for the frozen Codex/high v5 candidate.

The project owner's standing directive authorizes additional independent
Codex/high runs until C1 closes.  This command records that directive against
one exact v5 freeze.  It performs no model inference and creates the
conditional terminal guard before the owner-approval record.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from skillchain.codex_authoring_v5 import (
    CODEX_AUTHOR_CANDIDATE_ID,
    CODEX_AUTHOR_RUN_ID,
)
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)

from scripts.build_codex_authoring_approval_package_v5 import (
    APPROVAL_PATH,
    CLAIM_PATH,
    FREEZE_PATH,
    OUTPUT_DIR,
    RECEIPT_PATH,
    ROOT,
    TERMINAL_GUARD_PATH,
    V2_RETIREMENT_CLAIM_PATH,
    V3_AUDIT_INCIDENT_PATH,
    V4_RECEIPT_PATH,
    _expected_files,
    planned_terminal_guard_bytes,
)


def _self_hash(payload: dict[str, Any], field: str, label: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    if observed != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError(f"{label} self-hash drifted")


def _read_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    content = read_stable_regular_file(path, label=label)
    parsed = parse_canonical_json(content, label=label)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain an object")
    return content, parsed


def _freeze_binding(
    freeze: dict[str, Any],
    name: str,
) -> tuple[Path, bytes, dict[str, Any] | None]:
    bindings = freeze.get("bindings")
    item = bindings.get(name) if isinstance(bindings, dict) else None
    if not isinstance(item, dict) or set(item) != {"file", "file_sha256"}:
        raise ValueError(f"Codex v5 freeze binding is invalid: {name}")
    relative = item.get("file")
    expected_sha256 = item.get("file_sha256")
    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"Codex v5 freeze binding values are invalid: {name}")
    path = (ROOT / relative).resolve(strict=True)
    if ROOT.resolve() not in path.parents:
        raise ValueError(f"Codex v5 freeze binding escapes root: {name}")
    content = read_stable_regular_file(path, label=f"Codex v5 binding {name}")
    if sha256_bytes(content) != expected_sha256:
        raise ValueError(f"Codex v5 freeze binding drifted: {name}")
    parsed = parse_canonical_json(content, label=f"Codex v5 binding {name}")
    return path, content, parsed if isinstance(parsed, dict) else None


def _create_or_verify(path: Path, content: bytes, label: str) -> str | None:
    creation_error: BaseException | None = None
    try:
        atomic_create_file(path, content)
    except BaseException as error:
        creation_error = error
    try:
        exact = (
            not path.is_symlink()
            and path.is_file()
            and read_stable_regular_file(path, label=label) == content
        )
    except BaseException:
        exact = False
    if not exact:
        if creation_error is not None:
            raise creation_error
        raise ValueError(f"{label} create returned without exact durable bytes")
    if creation_error is None:
        return None
    return f"{type(creation_error).__name__}: {creation_error}"


def _require_absent_or_exact(path: Path, content: bytes, label: str) -> None:
    if not os.path.lexists(path):
        return
    if (
        path.is_symlink()
        or not path.is_file()
        or read_stable_regular_file(path, label=label) != content
    ):
        raise ValueError(f"{label} path is occupied by different bytes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-lock-file-sha256", required=True)
    parser.add_argument("--freeze-payload-sha256", required=True)
    parser.add_argument("--accepted-by", required=True)
    parser.add_argument(
        "--accept-one-codex-exec-session",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-gpt-5-6-sol-high",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-platform-mediated-evidence-boundary",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-zero-retry-repair-fallback-followup",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-v4-consumed-compiler-rejection",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-v5-prompt-contract-repair",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-conditional-terminal-guard",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-standing-owner-directive",
        action="store_true",
        required=True,
    )
    args = parser.parse_args()
    if args.accepted_by != "project-owner":
        raise SystemExit("Codex v5 owner approval must be accepted by project-owner")

    freeze_content, freeze = _read_object(FREEZE_PATH, "Codex v5 freeze lock")
    _self_hash(freeze, "freeze_payload_sha256", "Codex v5 freeze lock")
    expected = _expected_files()
    if freeze_content != expected[FREEZE_PATH]:
        raise ValueError("Codex v5 freeze differs from the live exact candidate")
    if (
        sha256_bytes(freeze_content) != args.freeze_lock_file_sha256
        or freeze.get("freeze_payload_sha256") != args.freeze_payload_sha256
        or freeze.get("status") != "frozen_candidate_awaiting_owner_confirmation"
        or freeze.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or freeze.get("invocation_authorized") is not False
    ):
        raise ValueError("Codex v5 freeze identity or state differs from approval")
    for path, content in expected.items():
        if read_stable_regular_file(path, label=f"Codex v5 candidate {path}") != content:
            raise ValueError(f"Codex v5 candidate artifact drifted: {path}")

    authorization = freeze.get("authorization")
    paths = authorization.get("paths") if isinstance(authorization, dict) else None
    expected_paths = {
        "run_id": CODEX_AUTHOR_RUN_ID,
        "output_directory": OUTPUT_DIR.relative_to(ROOT).as_posix(),
        "claim_file": CLAIM_PATH.relative_to(ROOT).as_posix(),
        "receipt_file": RECEIPT_PATH.relative_to(ROOT).as_posix(),
        "terminal_guard_file": TERMINAL_GUARD_PATH.relative_to(ROOT).as_posix(),
        "approval_record_file": APPROVAL_PATH.relative_to(ROOT).as_posix(),
    }
    if paths != expected_paths:
        raise ValueError("Codex v5 authorization paths are invalid")
    if any(os.path.lexists(path) for path in (CLAIM_PATH, OUTPUT_DIR, RECEIPT_PATH)):
        raise ValueError("a Codex v5 execution path is already occupied")

    _, retirement_bytes, retirement = _freeze_binding(freeze, "v2_retirement")
    _, v3_incident_bytes, v3_incident = _freeze_binding(
        freeze,
        "v3_preapproval_audit_incident",
    )
    v4_receipt_path, v4_receipt_bytes, v4_receipt = _freeze_binding(
        freeze,
        "v4_canonical_receipt",
    )
    if retirement is None or v3_incident is None or v4_receipt is None:
        raise ValueError("Codex v5 predecessor evidence must contain objects")
    _self_hash(
        retirement,
        "retirement_payload_sha256",
        "Codex v2 retirement",
    )
    _self_hash(
        v3_incident,
        "incident_payload_sha256",
        "Codex v3 incident",
    )
    _self_hash(
        v4_receipt,
        "receipt_payload_sha256",
        "Codex v4 receipt",
    )
    if (
        v4_receipt_path != V4_RECEIPT_PATH.resolve(strict=True)
        or v4_receipt.get("status")
        != "codex_session_completed_draft_rejected"
        or v4_receipt.get("authorization_consumed") is not True
        or v4_receipt.get("failure_stage") != "trusted_compiler"
        or v4_receipt.get("formal_codex_session_eligible") is not False
        or retirement.get("v2_authorization_reusable") is not False
        or v3_incident.get("inference_requested") is not False
    ):
        raise ValueError("Codex v5 predecessor evidence state drifted")

    guard_bytes = planned_terminal_guard_bytes()
    guard_file_sha256 = sha256_bytes(guard_bytes)
    guard = parse_canonical_json(guard_bytes, label="Codex v5 terminal guard")
    if not isinstance(guard, dict):
        raise ValueError("Codex v5 terminal guard must contain an object")
    guard_policy = freeze.get("terminal_guard")
    if (
        not isinstance(guard_policy, dict)
        or guard_policy.get("file")
        != TERMINAL_GUARD_PATH.relative_to(ROOT).as_posix()
        or guard_policy.get("expected_file_sha256") != guard_file_sha256
    ):
        raise ValueError("Codex v5 freeze does not bind the terminal guard")

    _, v2_incident_bytes, v2_incident = _freeze_binding(
        freeze,
        "v2_preflight_incident",
    )
    if v2_incident is None:
        raise ValueError("Codex v2 incident must contain an object")
    approval_payload = {
        "schema_version": 1,
        "status": "owner_approved_for_one_codex_author_session",
        "approved_by": "project-owner",
        "freeze_lock_file": FREEZE_PATH.relative_to(ROOT).as_posix(),
        "freeze_lock_file_sha256": args.freeze_lock_file_sha256,
        "freeze_payload_sha256": args.freeze_payload_sha256,
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
        "transactional_terminal_guard_v5_accepted": True,
        "prompt_contract_repair_v5_accepted": True,
        "standing_owner_repeat_authorization_accepted": True,
        "v2_authorization_retired_without_inference": True,
        "v2_retirement_claim_file": (
            V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
        ),
        "v2_retirement_claim_file_sha256": sha256_bytes(retirement_bytes),
        "v2_retirement_claim_payload_sha256": retirement[
            "retirement_payload_sha256"
        ],
        "v2_preflight_incident_file_sha256": sha256_bytes(v2_incident_bytes),
        "v2_preflight_incident_payload_sha256": v2_incident[
            "incident_payload_sha256"
        ],
        "v3_preapproval_audit_incident_file_sha256": sha256_bytes(
            v3_incident_bytes
        ),
        "v3_preapproval_audit_incident_payload_sha256": v3_incident[
            "incident_payload_sha256"
        ],
        "v4_canonical_receipt_file": (
            v4_receipt_path.relative_to(ROOT).as_posix()
        ),
        "v4_canonical_receipt_file_sha256": sha256_bytes(v4_receipt_bytes),
        "v4_canonical_receipt_payload_sha256": v4_receipt[
            "receipt_payload_sha256"
        ],
        "v4_authorization_consumed": True,
        "v4_result_rewritten_or_reclassified": False,
        "terminal_guard_file": TERMINAL_GUARD_PATH.relative_to(ROOT).as_posix(),
        "terminal_guard_file_sha256": guard_file_sha256,
        "terminal_guard_payload_sha256": guard[
            "terminal_guard_payload_sha256"
        ],
    }
    approval_bytes = canonical_json_bytes(
        {
            **approval_payload,
            "approval_payload_sha256": sha256_bytes(
                canonical_json_bytes(approval_payload)
            ),
        }
    )

    _require_absent_or_exact(
        TERMINAL_GUARD_PATH,
        guard_bytes,
        "Codex v5 conditional terminal guard",
    )
    _require_absent_or_exact(
        APPROVAL_PATH,
        approval_bytes,
        "Codex v5 owner approval",
    )
    guard_warning = _create_or_verify(
        TERMINAL_GUARD_PATH,
        guard_bytes,
        "Codex v5 conditional terminal guard",
    )
    approval_warning = _create_or_verify(
        APPROVAL_PATH,
        approval_bytes,
        "Codex v5 owner approval",
    )
    approval = json.loads(approval_bytes)
    print(
        json.dumps(
            {
                "status": approval["status"],
                "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
                "run_id": CODEX_AUTHOR_RUN_ID,
                "approval_record": APPROVAL_PATH.relative_to(ROOT).as_posix(),
                "approval_record_file_sha256": sha256_bytes(approval_bytes),
                "approval_payload_sha256": approval[
                    "approval_payload_sha256"
                ],
                "terminal_guard_file_sha256": guard_file_sha256,
                "v4_receipt_file_sha256": sha256_bytes(v4_receipt_bytes),
                "create_recovery_warnings": {
                    "terminal_guard": guard_warning,
                    "owner_approval": approval_warning,
                },
                "codex_process_launched": False,
                "inference_requested": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
