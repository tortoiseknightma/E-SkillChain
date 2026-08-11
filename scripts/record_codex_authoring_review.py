"""Record an unchanged human acceptance for the immutable Codex v5 draft.

This command does not modify the canonical run directory and does not compile
a Skill Bank.  It writes one external, create-only review receipt whose hashes
bind the canonical invocation receipt and pre-review draft.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
v5_runner = importlib.import_module("scripts.run_codex_authoring_v5")

from skillchain.codex_authoring_review import (  # noqa: E402
    CodexAuthoringHumanReviewReceipt,
    build_codex_authoring_human_review_receipt,
)
from skillchain.codex_authoring_v5 import (  # noqa: E402
    CODEX_AUTHOR_CANDIDATE_ID,
    CODEX_AUTHOR_RUN_ID,
)
from skillchain.static_authoring import (  # noqa: E402
    AuthoringDraftBundle,
    ReviewChecklist,
    build_human_review,
)
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)

RUN_ROOT = ROOT / "runs" / "formal-authoring" / CODEX_AUTHOR_RUN_ID
CLAIM_PATH = ROOT / "specs" / "authoring" / f"{CODEX_AUTHOR_RUN_ID}-attempt-claim.json"
GUARD_PATH = ROOT / "specs" / "authoring" / f"{CODEX_AUTHOR_RUN_ID}-terminal-guard.json"
OUTPUT_PATH = ROOT / "specs" / "authoring" / f"{CODEX_AUTHOR_RUN_ID}-human-review.json"


def _read_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    content = read_stable_regular_file(path, label=label)
    parsed = parse_canonical_json(content, label=label)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain an object")
    return content, parsed


def _real_directory(path: Path, label: str) -> None:
    metadata = path.lstat()
    reparse = int(getattr(metadata, "st_file_attributes", 0)) & int(
        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or reparse:
        raise ValueError(f"{label} must be a real directory")


def _create_or_verify(path: Path, content: bytes) -> bool:
    if os.path.lexists(path):
        if (
            path.is_symlink()
            or read_stable_regular_file(path, label="Codex human-review receipt")
            != content
        ):
            raise FileExistsError(
                f"create-only human-review target is occupied by different bytes: {path}"
            )
        return False
    _real_directory(path.parent, "human-review receipt parent")
    atomic_create_file(path, content)
    observed = read_stable_regular_file(path, label="Codex human-review receipt")
    if observed != content:
        raise ValueError("human-review receipt changed during create-only publication")
    return True


def record_codex_authoring_review(
    *,
    expected_invocation_receipt_file_sha256: str,
    expected_pre_review_file_sha256: str,
    reviewer_id: str,
    review_minutes: int,
    change_reason: str,
    checklist: ReviewChecklist,
    output_path: Path = OUTPUT_PATH,
) -> tuple[CodexAuthoringHumanReviewReceipt, bool]:
    _real_directory(RUN_ROOT, "Codex v5 canonical run")
    canonical_run_root = RUN_ROOT.resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if output_path == canonical_run_root or canonical_run_root in output_path.parents:
        raise ValueError("human review must remain outside the canonical run directory")

    receipt_path = RUN_ROOT / "invocation-receipt.json"
    receipt_bytes, receipt = _read_object(
        receipt_path, "Codex v5 canonical invocation receipt"
    )
    if sha256_bytes(receipt_bytes) != expected_invocation_receipt_file_sha256:
        raise ValueError("Codex v5 invocation-receipt file SHA-256 mismatch")
    receipt_unsigned = dict(receipt)
    receipt_payload_sha256 = receipt_unsigned.pop("receipt_payload_sha256", None)
    if receipt_payload_sha256 != sha256_bytes(canonical_json_bytes(receipt_unsigned)):
        raise ValueError("Codex v5 invocation receipt self-hash mismatch")
    guard_sha256 = receipt.get("terminal_guard_file_sha256")
    if not isinstance(
        guard_sha256, str
    ) or not v5_runner.validate_canonical_bundle_from_disk(
        RUN_ROOT,
        claim_path=CLAIM_PATH,
        guard_path=GUARD_PATH,
        guard_file_sha256=guard_sha256,
    ):
        raise ValueError("Codex v5 canonical bundle no longer replays")
    if (
        receipt.get("candidate_id") != CODEX_AUTHOR_CANDIDATE_ID
        or receipt.get("run_id") != CODEX_AUTHOR_RUN_ID
        or receipt.get("status") != "codex_session_completed_draft_ready_for_review"
        or receipt.get("formal_codex_session_eligible") is not True
        or receipt.get("canonical_bundle_complete") is not True
    ):
        raise ValueError("Codex v5 invocation is not eligible for human review")

    draft_path = RUN_ROOT / "pre-review-draft.json"
    draft_bytes, draft_raw = _read_object(draft_path, "Codex v5 pre-review draft")
    draft_file_sha256 = sha256_bytes(draft_bytes)
    canonical_files = receipt.get("canonical_bundle_files")
    if (
        draft_file_sha256 != expected_pre_review_file_sha256
        or receipt.get("pre_review_draft_sha256") != draft_file_sha256
        or not isinstance(canonical_files, dict)
        or canonical_files.get("pre-review-draft.json") != draft_file_sha256
    ):
        raise ValueError("Codex v5 pre-review draft SHA-256 binding mismatch")
    draft = AuthoringDraftBundle.model_validate(draft_raw, strict=True)
    if draft.canonical_bytes() != draft_bytes:
        raise ValueError("Codex v5 pre-review draft is not canonical")

    request_path = RUN_ROOT / "authoring-request.json"
    request_bytes, request = _read_object(
        request_path, "Codex v5 canonical authoring request"
    )
    if receipt.get("canonical_request_file_sha256") != sha256_bytes(request_bytes):
        raise ValueError("Codex v5 authoring request SHA-256 binding mismatch")
    authoring_input = request.get("authoring_input")
    session_budget = (
        authoring_input.get("session_budget")
        if isinstance(authoring_input, dict)
        else None
    )
    max_review_minutes = (
        session_budget.get("max_human_review_minutes")
        if isinstance(session_budget, dict)
        else None
    )
    if (
        not isinstance(authoring_input, dict)
        or authoring_input.get("input_sha256") != draft.authoring_input_sha256
        or not isinstance(max_review_minutes, int)
        or max_review_minutes <= 0
        or review_minutes > max_review_minutes
    ):
        raise ValueError("Codex v5 human-review budget or input binding mismatch")

    review = build_human_review(
        pre_review=draft,
        post_review=draft,
        reviewer_id=reviewer_id,
        review_minutes=review_minutes,
        change_reason=change_reason,
        checklist=checklist,
    )
    canonical_receipt_relative = receipt_path.relative_to(ROOT).as_posix()
    pre_review_relative = draft_path.relative_to(ROOT).as_posix()
    review_receipt = build_codex_authoring_human_review_receipt(
        candidate_id=CODEX_AUTHOR_CANDIDATE_ID,
        run_id=CODEX_AUTHOR_RUN_ID,
        canonical_invocation_receipt_file=canonical_receipt_relative,
        canonical_invocation_receipt_file_sha256=sha256_bytes(receipt_bytes),
        canonical_invocation_receipt_payload_sha256=receipt_payload_sha256,
        pre_review_file=pre_review_relative,
        pre_review_file_sha256=draft_file_sha256,
        pre_review_bundle_sha256=draft.bundle_sha256,
        authoring_input_sha256=draft.authoring_input_sha256,
        max_human_review_minutes=max_review_minutes,
        human_review=review,
    )
    content = review_receipt.canonical_bytes()
    created = _create_or_verify(output_path, content)
    reloaded_bytes, reloaded_raw = _read_object(
        output_path, "Codex human-review receipt"
    )
    reloaded = CodexAuthoringHumanReviewReceipt.model_validate(
        reloaded_raw, strict=True
    )
    if reloaded_bytes != content or reloaded != review_receipt:
        raise ValueError("Codex human-review receipt failed independent reload")
    return review_receipt, created


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-invocation-receipt-file-sha256", required=True)
    parser.add_argument("--expected-pre-review-file-sha256", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--review-minutes", required=True, type=int)
    parser.add_argument("--change-reason", required=True)
    parser.add_argument(
        "--decision",
        required=True,
        choices=("ACCEPT_UNCHANGED",),
    )
    parser.add_argument(
        "--confirm-no-private-inputs", action="store_true", required=True
    )
    parser.add_argument("--confirm-safety-checked", action="store_true", required=True)
    parser.add_argument("--confirm-schema-valid", action="store_true", required=True)
    parser.add_argument(
        "--confirm-source-citations-checked", action="store_true", required=True
    )
    parser.add_argument(
        "--confirm-tool-permissions-checked", action="store_true", required=True
    )
    parser.add_argument(
        "--confirm-edit-scope-checked", action="store_true", required=True
    )
    arguments = parser.parse_args()
    checklist = ReviewChecklist(
        no_private_inputs=arguments.confirm_no_private_inputs,
        safety_checked=arguments.confirm_safety_checked,
        schema_valid=arguments.confirm_schema_valid,
        source_citations_checked=arguments.confirm_source_citations_checked,
        tool_permissions_checked=arguments.confirm_tool_permissions_checked,
        edit_scope_checked=arguments.confirm_edit_scope_checked,
    )
    receipt, created = record_codex_authoring_review(
        expected_invocation_receipt_file_sha256=(
            arguments.expected_invocation_receipt_file_sha256
        ),
        expected_pre_review_file_sha256=arguments.expected_pre_review_file_sha256,
        reviewer_id=arguments.reviewer_id,
        review_minutes=arguments.review_minutes,
        change_reason=arguments.change_reason,
        checklist=checklist,
    )
    print(
        json.dumps(
            {
                "created": created,
                "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
                "run_id": receipt.run_id,
                "decision": receipt.decision,
                "reviewer_id": receipt.human_review.reviewer_id,
                "review_minutes": receipt.human_review.review_minutes,
                "changed": receipt.human_review.changed,
                "pre_review_file_sha256": receipt.pre_review_file_sha256,
                "human_review_sha256": receipt.human_review_sha256,
                "receipt_sha256": receipt.receipt_sha256,
                "receipt_file_sha256": sha256_bytes(receipt.canonical_bytes()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
