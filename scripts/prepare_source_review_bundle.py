"""Prepare evidence-bound owner proposals or materialize an explicitly confirmed ledger."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.source_lock import RequiredSourceLock  # noqa: E402
from skillchain.data.source_review import (  # noqa: E402
    SourceReviewProposal,
    load_source_review_policy,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    sha256_bytes,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--source-lock-root",
        type=Path,
        default=REPOSITORY_ROOT / "specs/data_sources/c2/source-locks",
    )
    prepare.add_argument(
        "--license-dossier",
        type=Path,
        default=REPOSITORY_ROOT
        / "specs/data_sources/license-evidence-dossier-v1.json",
    )
    prepare.add_argument(
        "--proposal-plan",
        type=Path,
        default=REPOSITORY_ROOT
        / "specs/data_sources/owner-review-proposal-plan-v1.json",
    )
    prepare.add_argument(
        "--policy",
        type=Path,
        default=REPOSITORY_ROOT
        / "specs/data_sources/mvp-source-review-policy-v2.json",
    )
    prepare.add_argument(
        "--portfolio",
        type=Path,
        default=REPOSITORY_ROOT
        / "specs/data_sources/ecommerce-mvp-source-portfolio-v1.json",
    )
    prepare.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "specs/data_sources/c2/source-review",
    )

    materialize = commands.add_parser("materialize-owner-ledger")
    materialize.add_argument("--proposals", type=Path, required=True)
    materialize.add_argument("--proposal-sha256", required=True)
    materialize.add_argument("--reviewer-id", required=True)
    materialize.add_argument("--reviewed-at", required=True)
    materialize.add_argument("--owner-instruction", required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    if arguments.command == "prepare":
        return _prepare(arguments)
    return _materialize(arguments)


def _prepare(arguments: argparse.Namespace) -> int:
    portfolio_bytes = arguments.portfolio.read_bytes()
    policy_bytes = arguments.policy.read_bytes()
    policy = load_source_review_policy(
        arguments.policy,
        expected_policy_file_sha256=sha256_bytes(policy_bytes),
        expected_portfolio_sha256=sha256_bytes(portfolio_bytes),
    )
    required_ids = tuple(
        item.source_id for item in policy.requirements if item.required
    )
    lock_manifest_bytes = (
        arguments.source_lock_root / "required-source-lock-manifest.json"
    ).read_bytes()
    lock_manifest = parse_canonical_json(
        lock_manifest_bytes, label="required source lock manifest"
    )
    lock_rows = lock_manifest["sources"]
    if tuple(item["source_id"] for item in lock_rows) != required_ids:
        raise ValueError("source lock manifest differs from required policy sources")

    dossier_bytes = arguments.license_dossier.read_bytes()
    dossier = json.loads(dossier_bytes)
    proposal_plan_bytes = arguments.proposal_plan.read_bytes()
    proposal_plan = json.loads(proposal_plan_bytes)
    dossier_rows = dossier["sources"]
    proposal_rows = proposal_plan["sources"]
    if tuple(item["source_id"] for item in dossier_rows) != required_ids:
        raise ValueError("license dossier differs from required policy sources")
    if tuple(item["source_id"] for item in proposal_rows) != required_ids:
        raise ValueError("proposal plan differs from required policy sources")

    arguments.output.mkdir(parents=True, exist_ok=False)
    evidence_root = arguments.output / "license-evidence"
    evidence_root.mkdir()
    proposals: list[SourceReviewProposal] = []
    evidence_manifest: list[dict[str, Any]] = []
    requirement_by_id = {
        item.source_id: item for item in policy.requirements if item.required
    }
    for lock_row, dossier_row, proposal_row in zip(
        lock_rows, dossier_rows, proposal_rows, strict=True
    ):
        source_id = lock_row["source_id"]
        lock_path = arguments.source_lock_root / lock_row["lock_path"]
        lock_bytes = lock_path.read_bytes()
        if sha256_bytes(lock_bytes) != lock_row["source_lock_sha256"]:
            raise ValueError(f"{source_id}: source lock manifest digest mismatch")
        parse_canonical_json(lock_bytes, label=f"{source_id} source lock")
        lock = RequiredSourceLock.model_validate_json(lock_bytes, strict=True)
        if lock.source_revision != lock_row["source_revision"]:
            raise ValueError(f"{source_id}: lock revision mismatch")

        evidence_payload = {
            "assessed_at": dossier["assessed_at"],
            "assessment_boundary": dossier["assessment_boundary"],
            "dossier_id": dossier["dossier_id"],
            "schema_version": 1,
            **dossier_row,
        }
        evidence_bytes = canonical_json_bytes(evidence_payload)
        evidence_name = f"{source_id}.license-evidence.json"
        (evidence_root / evidence_name).write_bytes(evidence_bytes)
        evidence_sha256 = sha256_bytes(evidence_bytes)
        evidence_manifest.append(
            {
                "evidence_path": f"license-evidence/{evidence_name}",
                "license_evidence_sha256": evidence_sha256,
                "license_id": dossier_row["license_id"],
                "source_id": source_id,
            }
        )

        requirement = requirement_by_id[source_id]
        if tuple(proposal_row["purposes"]) != requirement.purposes:
            raise ValueError(f"{source_id}: proposal purposes differ from policy")
        proposal = SourceReviewProposal(
            source_id=source_id,
            source_revision=lock.source_revision,
            source_lock_sha256=lock_row["source_lock_sha256"],
            license_id=dossier_row["license_id"],
            license_evidence_sha256=evidence_sha256,
            proposed_decision=proposal_row["proposed_decision"],
            prepared_by=proposal_plan["prepared_by"],
            prepared_at=_datetime(proposal_plan["prepared_at"]),
            purposes=tuple(proposal_row["purposes"]),
            permissions=proposal_row["permissions"],
            pii_status=proposal_row["pii_status"],
            redaction_policy_sha256=proposal_row["redaction_policy_sha256"],
            rationale=proposal_row["rationale"],
        )
        proposals.append(proposal)

    proposal_bytes = canonical_jsonl_bytes(
        [item.model_dump(mode="json") for item in proposals]
    )
    proposal_path = arguments.output / "owner-review-proposals.jsonl"
    proposal_path.write_bytes(proposal_bytes)
    bundle_manifest = {
        "evidence": evidence_manifest,
        "license_dossier_sha256": sha256_bytes(dossier_bytes),
        "owner_confirmation_required": True,
        "owner_ledger_created": False,
        "policy_id": policy.policy_id,
        "policy_sha256": sha256_bytes(policy_bytes),
        "portfolio_sha256": sha256_bytes(portfolio_bytes),
        "proposal_plan_sha256": sha256_bytes(proposal_plan_bytes),
        "proposal_sha256": sha256_bytes(proposal_bytes),
        "schema_version": 1,
        "source_lock_manifest_sha256": sha256_bytes(lock_manifest_bytes),
        "status": "prepared_for_owner_confirmation",
    }
    manifest_bytes = canonical_json_bytes(bundle_manifest)
    manifest_path = arguments.output / "source-review-bundle-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    print(
        json.dumps(
            {
                "approved_proposals": [
                    item.source_id
                    for item in proposals
                    if item.proposed_decision == "approved"
                ],
                "deferred_proposals": [
                    item.source_id
                    for item in proposals
                    if item.proposed_decision == "deferred"
                ],
                "manifest_sha256": sha256_bytes(manifest_bytes),
                "owner_confirmation_required": True,
                "proposal_sha256": sha256_bytes(proposal_bytes),
                "status": "prepared",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _materialize(arguments: argparse.Namespace) -> int:
    proposal_bytes = arguments.proposals.read_bytes()
    if sha256_bytes(proposal_bytes) != arguments.proposal_sha256:
        raise ValueError("proposal external digest mismatch")
    rows = parse_canonical_jsonl(proposal_bytes, label="owner review proposals")
    proposals = tuple(
        SourceReviewProposal.model_validate_json(
            canonical_json_bytes(row), strict=True
        )
        for row in rows
    )
    source_ids = tuple(item.source_id for item in proposals)
    if not source_ids or source_ids != tuple(sorted(set(source_ids))):
        raise ValueError("owner proposals must be sorted, unique, and nonempty")
    reviewer_id = arguments.reviewer_id.strip()
    if not reviewer_id or reviewer_id != arguments.reviewer_id:
        raise ValueError("reviewer ID must be canonical nonblank text")
    owner_instruction = arguments.owner_instruction.strip()
    if (
        not owner_instruction
        or owner_instruction != arguments.owner_instruction
    ):
        raise ValueError("owner instruction must be canonical nonblank text")
    if arguments.output.resolve() == arguments.receipt_output.resolve():
        raise ValueError("ledger and signature receipt paths must differ")
    if arguments.output.parent.resolve() != arguments.receipt_output.parent.resolve():
        raise ValueError("ledger and signature receipt must share one output directory")
    reviewed_at = _datetime(arguments.reviewed_at)
    if any(reviewed_at < proposal.prepared_at for proposal in proposals):
        raise ValueError("owner review cannot predate proposal preparation")
    records = [
        proposal.owner_record(
            reviewer_id=reviewer_id,
            reviewed_at=reviewed_at,
        ).model_dump(mode="json")
        for proposal in proposals
    ]
    ledger_bytes = canonical_jsonl_bytes(records)
    signed_root = arguments.output.parent
    signed_root.mkdir(parents=True, exist_ok=False)
    with arguments.output.open("xb") as handle:
        handle.write(ledger_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    ledger_sha256 = sha256_bytes(ledger_bytes)
    receipt = {
        "authority_basis": "explicit_project_owner_instruction",
        "decision_summary": {
            "approved": [
                item["source_id"]
                for item in records
                if item["decision"] == "approved"
            ],
            "deferred": [
                item["source_id"]
                for item in records
                if item["decision"] == "deferred"
            ],
            "rejected": [
                item["source_id"]
                for item in records
                if item["decision"] == "rejected"
            ],
        },
        "ledger_path": _repository_relative(arguments.output),
        "ledger_sha256": ledger_sha256,
        "owner_instruction": owner_instruction,
        "proposal_path": _repository_relative(arguments.proposals),
        "proposal_sha256": arguments.proposal_sha256,
        "record_count": len(records),
        "reviewed_at": reviewed_at.isoformat().replace("+00:00", "Z"),
        "reviewer_id": reviewer_id,
        "risk_boundary": (
            "Project-owner approval records an internal risk decision; it does "
            "not replace missing upstream license evidence or grant "
            "redistribution rights. Remote processing and public-demo rights "
            "are limited to the permissions explicitly present in each source "
            "record."
        ),
        "schema_version": 1,
        "status": "owner-signed",
    }
    receipt_bytes = canonical_json_bytes(receipt)
    arguments.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.receipt_output.open("xb") as handle:
        handle.write(receipt_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "ledger_path": str(arguments.output),
                "ledger_sha256": ledger_sha256,
                "record_count": len(records),
                "receipt_path": str(arguments.receipt_output),
                "receipt_sha256": sha256_bytes(receipt_bytes),
                "status": "owner-ledger-materialized",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("signed artifacts must be inside the repository") from error


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
