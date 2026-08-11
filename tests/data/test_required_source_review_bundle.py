from __future__ import annotations

from pathlib import Path

from skillchain.data.source_lock import RequiredSourceLock
from skillchain.data.source_review import (
    SourceReviewProposal,
    load_source_review_policy,
    load_verified_source_review_ledger,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    sha256_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_required_source_review_bundle_binds_ten_real_locks_and_evidence() -> None:
    lock_root = REPO_ROOT / "specs/data_sources/c2/source-locks"
    review_root = REPO_ROOT / "specs/data_sources/c2/source-review"
    lock_manifest_bytes = (
        lock_root / "required-source-lock-manifest.json"
    ).read_bytes()
    lock_manifest = parse_canonical_json(
        lock_manifest_bytes, label="source lock manifest"
    )
    review_manifest = parse_canonical_json(
        (review_root / "source-review-bundle-manifest.json").read_bytes(),
        label="source review bundle manifest",
    )
    assert review_manifest["source_lock_manifest_sha256"] == sha256_bytes(
        lock_manifest_bytes
    )
    assert review_manifest["owner_confirmation_required"] is True
    assert review_manifest["owner_ledger_created"] is False
    assert len(lock_manifest["sources"]) == 10

    source_ids = []
    for row in lock_manifest["sources"]:
        lock_bytes = (lock_root / row["lock_path"]).read_bytes()
        assert sha256_bytes(lock_bytes) == row["source_lock_sha256"]
        parse_canonical_json(lock_bytes, label=f"{row['source_id']} source lock")
        lock = RequiredSourceLock.model_validate_json(lock_bytes, strict=True)
        assert lock.source_id == row["source_id"]
        source_ids.append(lock.source_id)

    proposal_bytes = (review_root / "owner-review-proposals.jsonl").read_bytes()
    assert sha256_bytes(proposal_bytes) == review_manifest["proposal_sha256"]
    proposal_rows = parse_canonical_jsonl(
        proposal_bytes, label="owner review proposals"
    )
    proposals = tuple(
        SourceReviewProposal.model_validate_json(
            canonical_json_bytes(row), strict=True
        )
        for row in proposal_rows
    )
    assert [item.source_id for item in proposals] == source_ids
    assert all(item.owner_confirmation_required for item in proposals)

    evidence_by_source = {
        item["source_id"]: item for item in review_manifest["evidence"]
    }
    assert sorted(evidence_by_source) == source_ids
    for proposal in proposals:
        evidence = evidence_by_source[proposal.source_id]
        evidence_bytes = (review_root / evidence["evidence_path"]).read_bytes()
        parse_canonical_json(
            evidence_bytes,
            label=f"{proposal.source_id} license evidence",
        )
        assert sha256_bytes(evidence_bytes) == proposal.license_evidence_sha256
        assert evidence["license_id"] == proposal.license_id


def test_conservative_proposals_do_not_claim_all_required_sources_approved() -> None:
    review_root = REPO_ROOT / "specs/data_sources/c2/source-review"
    rows = parse_canonical_jsonl(
        (review_root / "owner-review-proposals.jsonl").read_bytes(),
        label="owner review proposals",
    )
    decisions = {
        row["source_id"]: row["proposed_decision"] for row in rows
    }
    assert {
        source_id
        for source_id, decision in decisions.items()
        if decision == "approved"
    } == {"abo", "crosswoz", "durecdial_2_0", "fashioniq", "rpc"}
    assert {
        source_id
        for source_id, decision in decisions.items()
        if decision == "deferred"
    } == {
        "isia_food500",
        "muge",
        "wikimedia_zhwiki",
        "wildreceipt",
        "xiachufang",
    }


def test_owner_signed_v2_approves_all_sources_without_erasing_risk_evidence() -> None:
    review_root = REPO_ROOT / "specs/data_sources/c2/source-review-v2"
    signed_root = review_root / "signed"
    proposal_bytes = (review_root / "owner-review-proposals.jsonl").read_bytes()
    manifest_bytes = (
        review_root / "source-review-bundle-manifest.json"
    ).read_bytes()
    manifest = parse_canonical_json(
        manifest_bytes, label="v2 source review bundle manifest"
    )
    assert manifest["proposal_sha256"] == sha256_bytes(proposal_bytes)

    proposal_rows = parse_canonical_jsonl(
        proposal_bytes, label="v2 owner review proposals"
    )
    proposals = tuple(
        SourceReviewProposal.model_validate_json(
            canonical_json_bytes(row), strict=True
        )
        for row in proposal_rows
    )
    assert len(proposals) == 10
    assert {item.proposed_decision for item in proposals} == {"approved"}
    outbound_sources = {
        item.source_id
        for item in proposals
        if item.permissions.remote_embedding_allowed
        and item.permissions.public_demo_allowed
    }
    assert outbound_sources == {"abo", "fashioniq", "isia_food500", "rpc"}
    assert all(not item.permissions.redistribution_allowed for item in proposals)

    receipt_path = signed_root / "owner-ledger-signature-receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = parse_canonical_json(
        receipt_bytes, label="owner ledger signature receipt"
    )
    ledger_path = REPO_ROOT / receipt["ledger_path"]
    ledger_bytes = ledger_path.read_bytes()
    assert receipt["status"] == "owner-signed"
    assert receipt["reviewer_id"] == "project-owner"
    assert receipt["proposal_sha256"] == sha256_bytes(proposal_bytes)
    assert receipt["ledger_sha256"] == sha256_bytes(ledger_bytes)
    assert receipt["decision_summary"]["approved"] == [
        item.source_id for item in proposals
    ]
    assert receipt["decision_summary"]["deferred"] == []
    assert receipt["decision_summary"]["rejected"] == []

    policy_path = REPO_ROOT / "specs/data_sources/mvp-source-review-policy-v2.json"
    portfolio_path = (
        REPO_ROOT / "specs/data_sources/ecommerce-mvp-source-portfolio-v1.json"
    )
    policy_bytes = policy_path.read_bytes()
    policy = load_source_review_policy(
        policy_path,
        expected_policy_file_sha256=sha256_bytes(policy_bytes),
        expected_portfolio_sha256=sha256_bytes(portfolio_path.read_bytes()),
    )
    verified = load_verified_source_review_ledger(
        ledger_path,
        policy,
        policy_file_sha256=sha256_bytes(policy_bytes),
        expected_ledger_file_sha256=receipt["ledger_sha256"],
    )
    assert sorted(verified.approved) == [
        item.source_id for item in proposals
    ]

    noassertion = {
        record.source_id
        for record in verified.records
        if record.license_id == "NOASSERTION"
    }
    assert noassertion == {
        "isia_food500",
        "muge",
        "wildreceipt",
        "xiachufang",
    }
    assert "does not replace missing upstream license evidence" in receipt[
        "risk_boundary"
    ]
    assert "limited to the permissions explicitly present" in receipt["risk_boundary"]
