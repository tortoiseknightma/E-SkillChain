"""Create the exact owner-approval record for the frozen Codex Author candidate.

This command records a decision that the project owner has already made.  It
never invokes Codex and cannot create the attempt claim, receipt, or run output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.codex_authoring import CODEX_AUTHOR_RUN_ID
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = (
    ROOT / "specs" / "authoring" / "authoring-freeze-lock-codex-high-v2.json"
)


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
    args = parser.parse_args()

    freeze_bytes = read_stable_regular_file(
        FREEZE_PATH,
        label="Codex freeze lock",
    )
    freeze_file_sha256 = sha256_bytes(freeze_bytes)
    freeze = parse_canonical_json(freeze_bytes, label="Codex freeze lock")
    if not isinstance(freeze, dict):
        raise ValueError("Codex freeze lock must contain an object")
    unsigned_freeze = dict(freeze)
    freeze_payload_sha256 = unsigned_freeze.pop(
        "freeze_payload_sha256",
        None,
    )
    if (
        freeze_file_sha256 != args.accept_freeze_file_sha256
        or freeze_payload_sha256 != args.accept_freeze_payload_sha256
        or freeze_payload_sha256
        != sha256_bytes(canonical_json_bytes(unsigned_freeze))
        or freeze.get("status")
        != "frozen_candidate_awaiting_owner_confirmation"
        or freeze.get("candidate_id")
        != "authoring-codex-high-20260724-v2"
        or freeze.get("invocation_authorized") is not False
    ):
        raise ValueError("accepted Codex freeze identity or status does not match")
    authorization = freeze.get("authorization")
    paths = authorization.get("paths") if isinstance(authorization, dict) else None
    if not isinstance(paths, dict) or paths.get("run_id") != CODEX_AUTHOR_RUN_ID:
        raise ValueError("Codex freeze authorization paths are invalid")
    approval_path_value = paths.get("approval_record_file")
    if not isinstance(approval_path_value, str):
        raise ValueError("Codex freeze approval path is invalid")
    approval_path = (ROOT / approval_path_value).resolve()
    if ROOT.resolve() not in approval_path.parents:
        raise ValueError("Codex freeze approval path escapes the repository")

    payload = {
        "schema_version": 1,
        "status": "owner_approved_for_one_codex_author_session",
        "approved_by": "project-owner",
        "freeze_lock_file": FREEZE_PATH.relative_to(ROOT).as_posix(),
        "freeze_lock_file_sha256": freeze_file_sha256,
        "freeze_payload_sha256": freeze_payload_sha256,
        "candidate_id": freeze["candidate_id"],
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
    approval_payload_sha256 = sha256_bytes(canonical_json_bytes(payload))
    approval_bytes = canonical_json_bytes(
        {
            **payload,
            "approval_payload_sha256": approval_payload_sha256,
        }
    )
    atomic_create_file(approval_path, approval_bytes)
    print(
        json.dumps(
            {
                "approval_file": approval_path.relative_to(ROOT).as_posix(),
                "approval_file_sha256": sha256_bytes(approval_bytes),
                "approval_payload_sha256": approval_payload_sha256,
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
