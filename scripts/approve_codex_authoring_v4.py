"""Issue exact owner authority for the frozen Codex/high v4 candidate.

This command performs no model inference.  It creates, in fail-closed order:
the conditional terminal guard, the v2 non-inference retirement tombstone, and
the v4 one-session owner approval.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from skillchain.codex_authoring_v4 import (
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

from scripts.build_codex_authoring_approval_package_v4 import (
    APPROVAL_PATH,
    CLAIM_PATH,
    FREEZE_PATH,
    OUTPUT_DIR,
    RECEIPT_PATH,
    ROOT,
    TERMINAL_GUARD_PATH,
    V2_APPROVAL_FILE_SHA256,
    V2_APPROVAL_PATH,
    V2_APPROVAL_PAYLOAD_SHA256,
    V2_FREEZE_FILE_SHA256,
    V2_FREEZE_PATH,
    V2_FREEZE_PAYLOAD_SHA256,
    V2_OUTPUT_DIR,
    V2_PREFLIGHT_INCIDENT_PATH,
    V2_RECEIPT_PATH,
    V2_RETIREMENT_CLAIM_PATH,
    V2_RUN_ID,
    V3_APPROVAL_PATH,
    V3_AUDIT_INCIDENT_PATH,
    V3_CLAIM_PATH,
    V3_OUTPUT_DIR,
    V3_RECEIPT_PATH,
    planned_terminal_guard_bytes,
    planned_v2_retirement_claim_bytes,
    verify_v2_v3_bound_history,
)


def _bound_object(
    path: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    content = read_stable_regular_file(path, label=label)
    if sha256_bytes(content) != expected_sha256:
        raise ValueError(f"{label} file SHA-256 drifted")
    parsed = parse_canonical_json(content, label=label)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain an object")
    return parsed


def _self_hash(
    payload: dict[str, Any],
    field: str,
    label: str,
) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    if observed != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError(f"{label} self-hash drifted")


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
    kind = type(creation_error).__name__
    try:
        message = str(creation_error)
    except BaseException:
        message = "<error stringification failed>"
    return f"{kind}: {message}"


def _require_absent_or_exact(path: Path, content: bytes, label: str) -> None:
    if not os.path.lexists(path):
        return
    if (
        path.is_symlink()
        or not path.is_file()
        or read_stable_regular_file(path, label=label) != content
    ):
        raise ValueError(f"{label} path is occupied by different bytes")


def _freeze_binding(freeze: dict[str, Any], name: str) -> tuple[Path, str]:
    bindings = freeze.get("bindings")
    item = bindings.get(name) if isinstance(bindings, dict) else None
    if not isinstance(item, dict) or set(item) != {"file", "file_sha256"}:
        raise ValueError(f"Codex v4 freeze binding is invalid: {name}")
    relative = item.get("file")
    digest = item.get("file_sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError(f"Codex v4 freeze binding values are invalid: {name}")
    unresolved = ROOT / relative
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ValueError(f"Codex v4 freeze binding is not a regular file: {name}")
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Codex v4 freeze binding escapes root: {name}") from error
    return path, digest


def _validate_incidents_before_retirement(
    freeze: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    v2_path, v2_digest = _freeze_binding(freeze, "v2_preflight_incident")
    if v2_path != V2_PREFLIGHT_INCIDENT_PATH.resolve(strict=True):
        raise ValueError("v2 preflight incident path differs from the freeze")
    v2 = _bound_object(v2_path, v2_digest, "Codex v2 preflight incident")
    _self_hash(v2, "incident_payload_sha256", "Codex v2 preflight incident")
    if (
        v2.get("claim_created") is not False
        or v2.get("codex_exec_process_spawned") is not False
        or v2.get("inference_requested") is not False
        or v2.get("authorization_consumed_at_incident") is not False
    ):
        raise ValueError("v2 incident does not prove a pre-claim non-inference failure")

    v3_path, v3_digest = _freeze_binding(
        freeze,
        "v3_preapproval_audit_incident",
    )
    if v3_path != V3_AUDIT_INCIDENT_PATH.resolve(strict=True):
        raise ValueError("v3 audit incident path differs from the freeze")
    v3 = _bound_object(v3_path, v3_digest, "Codex v3 preapproval audit incident")
    _self_hash(v3, "incident_payload_sha256", "Codex v3 audit incident")
    if (
        v3.get("status") != "superseded_before_owner_approval_and_inference"
        or v3.get("owner_approval_created") is not False
        or v3.get("claim_created") is not False
        or v3.get("codex_exec_process_spawned") is not False
        or v3.get("inference_requested") is not False
    ):
        raise ValueError("v3 audit incident does not prove unapproved non-inference")
    return v2, v3


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
        "--accept-v2-non-inference-retirement",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-v3-preapproval-supersession",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-conditional-terminal-guard",
        action="store_true",
        required=True,
    )
    args = parser.parse_args()
    if args.accepted_by != "project-owner":
        raise SystemExit("Codex v4 owner approval must be accepted by project-owner")

    freeze_content = read_stable_regular_file(
        FREEZE_PATH,
        label="Codex v4 freeze lock",
    )
    if sha256_bytes(freeze_content) != args.freeze_lock_file_sha256:
        raise ValueError("Codex v4 freeze file SHA-256 differs from confirmation")
    freeze = parse_canonical_json(freeze_content, label="Codex v4 freeze lock")
    if not isinstance(freeze, dict):
        raise ValueError("Codex v4 freeze must contain an object")
    _self_hash(freeze, "freeze_payload_sha256", "Codex v4 freeze")
    if (
        freeze.get("freeze_payload_sha256") != args.freeze_payload_sha256
        or freeze.get("status") != "frozen_candidate_awaiting_owner_confirmation"
        or freeze.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or freeze.get("invocation_authorized") is not False
    ):
        raise ValueError("Codex v4 freeze identity or state differs from confirmation")

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
        raise ValueError("Codex v4 authorization paths are invalid")

    if (
        sha256_bytes(read_stable_regular_file(V2_FREEZE_PATH, label="v2 freeze"))
        != V2_FREEZE_FILE_SHA256
        or sha256_bytes(
            read_stable_regular_file(V2_APPROVAL_PATH, label="v2 approval")
        )
        != V2_APPROVAL_FILE_SHA256
    ):
        raise ValueError("Codex v2 freeze or approval history drifted")
    v2_freeze = _bound_object(V2_FREEZE_PATH, V2_FREEZE_FILE_SHA256, "v2 freeze")
    v2_approval = _bound_object(
        V2_APPROVAL_PATH,
        V2_APPROVAL_FILE_SHA256,
        "v2 approval",
    )
    if (
        v2_freeze.get("freeze_payload_sha256") != V2_FREEZE_PAYLOAD_SHA256
        or v2_approval.get("approval_payload_sha256")
        != V2_APPROVAL_PAYLOAD_SHA256
        or os.path.lexists(V2_OUTPUT_DIR)
        or os.path.lexists(V2_RECEIPT_PATH)
    ):
        raise ValueError("Codex v2 history is not the approved-uninvoked state")

    verify_v2_v3_bound_history()
    v2_incident, v3_incident = _validate_incidents_before_retirement(freeze)
    if any(
        os.path.lexists(path)
        for path in (
            V3_APPROVAL_PATH,
            V3_CLAIM_PATH,
            V3_OUTPUT_DIR,
            V3_RECEIPT_PATH,
            CLAIM_PATH,
            OUTPUT_DIR,
            RECEIPT_PATH,
        )
    ):
        raise ValueError("a v3/v4 authority or execution path is already occupied")

    guard_bytes = planned_terminal_guard_bytes()
    guard_file_sha256 = sha256_bytes(guard_bytes)
    guard_policy = freeze.get("terminal_guard")
    if (
        not isinstance(guard_policy, dict)
        or guard_policy.get("file")
        != TERMINAL_GUARD_PATH.relative_to(ROOT).as_posix()
        or guard_policy.get("expected_file_sha256") != guard_file_sha256
    ):
        raise ValueError("Codex v4 freeze does not bind the exact terminal guard")

    retirement_bytes = planned_v2_retirement_claim_bytes()
    retirement_file_sha256 = sha256_bytes(retirement_bytes)
    retirement_policy = freeze.get("prior_authorization_retirement")
    if (
        not isinstance(retirement_policy, dict)
        or retirement_policy.get("retirement_claim_file")
        != V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
        or retirement_policy.get("retirement_claim_expected_file_sha256")
        != retirement_file_sha256
        or retirement_policy.get("retirement_must_precede_v4_approval") is not True
        or retirement_policy.get("v2_approval_reusable") is not False
    ):
        raise ValueError("Codex v4 freeze does not retire the v2 authorization")
    retirement = parse_canonical_json(
        retirement_bytes,
        label="Codex v2 non-inference retirement claim",
    )
    guard = parse_canonical_json(
        guard_bytes,
        label="Codex v4 conditional terminal guard",
    )
    if not isinstance(retirement, dict) or not isinstance(guard, dict):
        raise ValueError("planned guard or retirement must contain an object")

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
        "transactional_terminal_guard_v4_accepted": True,
        "v2_authorization_retired_without_inference": True,
        "v2_retirement_claim_file": (
            V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
        ),
        "v2_retirement_claim_file_sha256": retirement_file_sha256,
        "v2_retirement_claim_payload_sha256": retirement[
            "retirement_payload_sha256"
        ],
        "v2_preflight_incident_file_sha256": sha256_bytes(
            read_stable_regular_file(
                V2_PREFLIGHT_INCIDENT_PATH,
                label="Codex v2 preflight incident",
            )
        ),
        "v2_preflight_incident_payload_sha256": v2_incident[
            "incident_payload_sha256"
        ],
        "v3_preapproval_audit_incident_file_sha256": sha256_bytes(
            read_stable_regular_file(
                V3_AUDIT_INCIDENT_PATH,
                label="Codex v3 preapproval audit incident",
            )
        ),
        "v3_preapproval_audit_incident_payload_sha256": v3_incident[
            "incident_payload_sha256"
        ],
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

    # The guard must exist before the claim can ever be accepted.  The v2
    # retirement must exist before the v4 approval becomes valid.
    _require_absent_or_exact(
        TERMINAL_GUARD_PATH,
        guard_bytes,
        "Codex v4 conditional terminal guard",
    )
    _require_absent_or_exact(
        V2_RETIREMENT_CLAIM_PATH,
        retirement_bytes,
        "Codex v2 non-inference retirement claim",
    )
    _require_absent_or_exact(
        APPROVAL_PATH,
        approval_bytes,
        "Codex v4 owner approval",
    )
    guard_creation_warning = _create_or_verify(
        TERMINAL_GUARD_PATH,
        guard_bytes,
        "Codex v4 conditional terminal guard",
    )
    retirement_creation_warning = _create_or_verify(
        V2_RETIREMENT_CLAIM_PATH,
        retirement_bytes,
        "Codex v2 non-inference retirement claim",
    )
    approval_creation_warning = _create_or_verify(
        APPROVAL_PATH,
        approval_bytes,
        "Codex v4 owner approval",
    )

    print(
        json.dumps(
            {
                "status": "owner_approved_for_one_codex_author_session",
                "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
                "run_id": CODEX_AUTHOR_RUN_ID,
                "approval_record": APPROVAL_PATH.relative_to(ROOT).as_posix(),
                "approval_record_file_sha256": sha256_bytes(approval_bytes),
                "approval_payload_sha256": json.loads(approval_bytes)[
                    "approval_payload_sha256"
                ],
                "terminal_guard_file_sha256": guard_file_sha256,
                "retired_v2_claim_file_sha256": retirement_file_sha256,
                "create_recovery_warnings": {
                    "terminal_guard": guard_creation_warning,
                    "v2_retirement": retirement_creation_warning,
                    "owner_approval": approval_creation_warning,
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
