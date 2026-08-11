"""Retire the uninvoked v2 authority and create the exact v3 owner approval.

This command never invokes Codex.  It first occupies the frozen v2 claim path
with a deterministic non-inference retirement tombstone, preventing the v2
approval and a future v3 approval from being live at the same time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.build_codex_authoring_approval_package_v3 import (
    APPROVAL_PATH,
    FREEZE_PATH,
    ROOT,
    V2_APPROVAL_FILE_SHA256,
    V2_APPROVAL_PATH,
    V2_APPROVAL_PAYLOAD_SHA256,
    V2_FREEZE_FILE_SHA256,
    V2_FREEZE_PATH,
    V2_FREEZE_PAYLOAD_SHA256,
    V2_OUTPUT_DIR,
    V2_PREFLIGHT_INCIDENT_PATH,
    V2_RECEIPT_PATH,
    V2_RUN_ID,
    V2_RETIREMENT_CLAIM_PATH,
    planned_v2_retirement_claim_bytes,
)
from skillchain.codex_authoring_v3 import (
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


def _bound_object(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    content = read_stable_regular_file(path, label=label)
    if sha256_bytes(content) != expected_sha256:
        raise ValueError(f"{label} file digest drifted")
    value = parse_canonical_json(content, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _create_or_verify(path: Path, content: bytes, label: str) -> None:
    if os.path.lexists(path):
        if (
            not path.is_file()
            or path.is_symlink()
            or read_stable_regular_file(path, label=label) != content
        ):
            raise ValueError(f"{label} conflicts with the approved bytes")
        return
    atomic_create_file(path, content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-freeze-file-sha256", required=True)
    parser.add_argument("--accept-freeze-payload-sha256", required=True)
    parser.add_argument(
        "--accept-platform-mediated-non-provider-attested",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-one-session-and-permanent-consumption",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-input-isolation-limit",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-s1-comparability-envelope",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-deterministic-environment-v3",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--accept-v2-non-inference-retirement",
        action="store_true",
        required=True,
    )
    args = parser.parse_args()

    freeze_bytes = read_stable_regular_file(
        FREEZE_PATH,
        label="Codex v3 freeze lock",
    )
    freeze_file_sha256 = sha256_bytes(freeze_bytes)
    freeze = parse_canonical_json(freeze_bytes, label="Codex v3 freeze lock")
    if not isinstance(freeze, dict):
        raise ValueError("Codex v3 freeze lock must contain an object")
    unsigned_freeze = dict(freeze)
    freeze_payload_sha256 = unsigned_freeze.pop("freeze_payload_sha256", None)
    if (
        freeze_file_sha256 != args.accept_freeze_file_sha256
        or freeze_payload_sha256 != args.accept_freeze_payload_sha256
        or freeze_payload_sha256
        != sha256_bytes(canonical_json_bytes(unsigned_freeze))
        or freeze.get("status")
        != "frozen_candidate_awaiting_owner_confirmation"
        or freeze.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or freeze.get("invocation_authorized") is not False
    ):
        raise ValueError("accepted Codex v3 freeze identity or status does not match")

    authorization = freeze.get("authorization")
    paths = authorization.get("paths") if isinstance(authorization, dict) else None
    if (
        not isinstance(paths, dict)
        or paths.get("run_id") != CODEX_AUTHOR_RUN_ID
        or paths.get("approval_record_file")
        != APPROVAL_PATH.relative_to(ROOT).as_posix()
    ):
        raise ValueError("Codex v3 authorization paths are invalid")

    v2_freeze = _bound_object(
        V2_FREEZE_PATH,
        V2_FREEZE_FILE_SHA256,
        "Codex v2 freeze history",
    )
    v2_approval = _bound_object(
        V2_APPROVAL_PATH,
        V2_APPROVAL_FILE_SHA256,
        "Codex v2 owner approval history",
    )
    if (
        v2_freeze.get("freeze_payload_sha256")
        != V2_FREEZE_PAYLOAD_SHA256
        or v2_approval.get("approval_payload_sha256")
        != V2_APPROVAL_PAYLOAD_SHA256
        or v2_approval.get("status")
        != "owner_approved_for_one_codex_author_session"
    ):
        raise ValueError("Codex v2 approval history is invalid")
    if (
        os.path.lexists(V2_OUTPUT_DIR)
        or os.path.lexists(V2_RECEIPT_PATH)
        or not V2_PREFLIGHT_INCIDENT_PATH.is_file()
        or V2_PREFLIGHT_INCIDENT_PATH.is_symlink()
    ):
        raise ValueError("Codex v2 preclaim history is incomplete or conflicts")

    retirement_bytes = planned_v2_retirement_claim_bytes()
    retirement_file_sha256 = sha256_bytes(retirement_bytes)
    retirement_policy = freeze.get("prior_authorization_retirement")
    if (
        not isinstance(retirement_policy, dict)
        or retirement_policy.get("retirement_claim_file")
        != V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
        or retirement_policy.get("retirement_claim_expected_file_sha256")
        != retirement_file_sha256
        or retirement_policy.get("retirement_must_precede_v3_approval") is not True
        or retirement_policy.get("v2_approval_reusable") is not False
    ):
        raise ValueError("Codex v3 freeze does not retire the v2 authorization")
    _create_or_verify(
        V2_RETIREMENT_CLAIM_PATH,
        retirement_bytes,
        "Codex v2 non-inference retirement claim",
    )
    retirement = parse_canonical_json(
        retirement_bytes,
        label="Codex v2 non-inference retirement claim",
    )
    if not isinstance(retirement, dict):
        raise ValueError("Codex v2 retirement claim must contain an object")
    if (
        retirement.get("status")
        != "authorization_retired_without_codex_process_launch"
        or retirement.get("run_id") != V2_RUN_ID
        or retirement.get("codex_exec_process_spawned") is not False
        or retirement.get("inference_requested") is not False
        or retirement.get("v2_authorization_reusable") is not False
    ):
        raise ValueError("Codex v2 retirement claim is not a non-inference tombstone")

    payload = {
        "schema_version": 1,
        "status": "owner_approved_for_one_codex_author_session",
        "approved_by": "project-owner",
        "freeze_lock_file": FREEZE_PATH.relative_to(ROOT).as_posix(),
        "freeze_lock_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": freeze_payload_sha256,
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
        "v2_retirement_claim_file_sha256": retirement_file_sha256,
        "v2_retirement_claim_payload_sha256": retirement[
            "retirement_payload_sha256"
        ],
    }
    approval_payload_sha256 = sha256_bytes(canonical_json_bytes(payload))
    approval_bytes = canonical_json_bytes(
        {
            **payload,
            "approval_payload_sha256": approval_payload_sha256,
        }
    )
    _create_or_verify(APPROVAL_PATH, approval_bytes, "Codex v3 owner approval")
    print(
        json.dumps(
            {
                "approval_file": APPROVAL_PATH.relative_to(ROOT).as_posix(),
                "approval_file_sha256": sha256_bytes(approval_bytes),
                "approval_payload_sha256": approval_payload_sha256,
                "retired_v2_claim_file": (
                    V2_RETIREMENT_CLAIM_PATH.relative_to(ROOT).as_posix()
                ),
                "retired_v2_claim_file_sha256": retirement_file_sha256,
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
