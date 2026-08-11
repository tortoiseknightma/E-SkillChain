from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from skillchain.data.source_review import (
    SourceReviewProposal,
    SourceReviewError,
    load_source_review_ledger,
    load_source_review_policy,
    load_verified_source_review_ledger,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]


def _policy(path: Path, *, portfolio_sha256: str) -> tuple[object, str]:
    content = canonical_json_bytes(
        {
            "policy_id": "test-policy-v1",
            "portfolio_sha256": portfolio_sha256,
            "requirements": [
                {
                    "pii_review": "not_applicable",
                    "purposes": ["product_gallery"],
                    "required": True,
                    "required_permissions": [
                        "download_allowed",
                        "local_embedding_allowed",
                        "local_research_allowed",
                    ],
                    "source_id": "abo",
                },
                {
                    "pii_review": "required",
                    "purposes": ["capability_gold"],
                    "required": True,
                    "required_permissions": [
                        "download_allowed",
                        "local_research_allowed",
                    ],
                    "source_id": "wildreceipt",
                },
            ],
            "schema_version": 1,
        }
    )
    path.write_bytes(content)
    digest = sha256_bytes(content)
    return (
        load_source_review_policy(
            path,
            expected_policy_file_sha256=digest,
            expected_portfolio_sha256=portfolio_sha256,
        ),
        digest,
    )


def _record(source_id: str, *, pii_status: str, purposes: list[str]) -> dict:
    return {
        "decision": "approved",
        "license_evidence_sha256": "2" * 64,
        "license_id": "research-license",
        "notes": None,
        "permissions": {
            "download_allowed": True,
            "local_embedding_allowed": source_id == "abo",
            "local_research_allowed": True,
            "public_demo_allowed": False,
            "redistribution_allowed": False,
            "remote_embedding_allowed": False,
        },
        "pii_status": pii_status,
        "purposes": purposes,
        "redaction_policy_sha256": None,
        "reviewed_at": "2026-07-23T00:00:00Z",
        "reviewer_id": "owner",
        "schema_version": 1,
        "source_id": source_id,
        "source_lock_sha256": "1" * 64,
        "source_revision": "fixture-revision",
    }


def test_verified_source_review_requires_all_human_gates(tmp_path: Path):
    policy, policy_sha256 = _policy(
        tmp_path / "policy.json",
        portfolio_sha256="3" * 64,
    )
    records = [
        _record("abo", pii_status="not_applicable", purposes=["product_gallery"]),
        _record(
            "wildreceipt",
            pii_status="restricted",
            purposes=["capability_gold"],
        ),
    ]
    content = b"".join(canonical_json_bytes(record) for record in records)
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_bytes(content)

    verified = load_verified_source_review_ledger(
        ledger_path,
        policy,
        policy_file_sha256=policy_sha256,
        expected_ledger_file_sha256=sha256_bytes(content),
    )
    assert sorted(verified.approved) == ["abo", "wildreceipt"]


def test_source_review_fails_closed_on_missing_or_fake_pii_approval(tmp_path: Path):
    policy, policy_sha256 = _policy(
        tmp_path / "policy.json",
        portfolio_sha256="3" * 64,
    )
    only_abo = canonical_json_bytes(
        _record("abo", pii_status="not_applicable", purposes=["product_gallery"])
    )
    ledger_path = tmp_path / "missing.jsonl"
    ledger_path.write_bytes(only_abo)
    with pytest.raises(SourceReviewError, match="wildreceipt:missing"):
        load_verified_source_review_ledger(
            ledger_path,
            policy,
            policy_file_sha256=policy_sha256,
            expected_ledger_file_sha256=sha256_bytes(only_abo),
        )

    records = [
        _record("abo", pii_status="not_applicable", purposes=["product_gallery"]),
        _record(
            "wildreceipt",
            pii_status="not_applicable",
            purposes=["capability_gold"],
        ),
    ]
    content = b"".join(canonical_json_bytes(record) for record in records)
    ledger_path = tmp_path / "fake-pii.jsonl"
    ledger_path.write_bytes(content)
    with pytest.raises(SourceReviewError, match="wildreceipt:pii_review"):
        load_verified_source_review_ledger(
            ledger_path,
            policy,
            policy_file_sha256=policy_sha256,
            expected_ledger_file_sha256=sha256_bytes(content),
        )


def test_incomplete_owner_ledger_is_inspectable_but_not_verified(
    tmp_path: Path,
) -> None:
    policy, policy_sha256 = _policy(
        tmp_path / "policy.json",
        portfolio_sha256="3" * 64,
    )
    deferred = _record(
        "wildreceipt",
        pii_status="restricted",
        purposes=["capability_gold"],
    )
    deferred["decision"] = "deferred"
    deferred["permissions"] = {
        key: False for key in deferred["permissions"]
    }
    records = [
        _record("abo", pii_status="not_applicable", purposes=["product_gallery"]),
        deferred,
    ]
    content = b"".join(canonical_json_bytes(record) for record in records)
    ledger_path = tmp_path / "incomplete-ledger.jsonl"
    ledger_path.write_bytes(content)

    loaded = load_source_review_ledger(
        ledger_path,
        policy,
        policy_file_sha256=policy_sha256,
        expected_ledger_file_sha256=sha256_bytes(content),
    )
    assert loaded.blockers == ("wildreceipt:decision=deferred",)
    assert sorted(loaded.approved) == ["abo"]
    with pytest.raises(SourceReviewError, match="decision=deferred"):
        load_verified_source_review_ledger(
            ledger_path,
            policy,
            policy_file_sha256=policy_sha256,
            expected_ledger_file_sha256=sha256_bytes(content),
        )


def test_proposal_requires_owner_confirmation_before_record_materialization() -> None:
    proposal = SourceReviewProposal(
        source_id="wildreceipt",
        source_revision="fixture-revision",
        source_lock_sha256="1" * 64,
        license_id="NOASSERTION",
        license_evidence_sha256="2" * 64,
        proposed_decision="deferred",
        prepared_by="evidence-preparer",
        prepared_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
        purposes=("capability_gold",),
        permissions={
            "download_allowed": False,
            "local_embedding_allowed": False,
            "local_research_allowed": False,
            "public_demo_allowed": False,
            "redistribution_allowed": False,
            "remote_embedding_allowed": False,
        },
        pii_status="restricted",
        rationale="License and PII gates remain unresolved.",
    )
    assert proposal.owner_confirmation_required is True
    record = proposal.owner_record(
        reviewer_id="project-owner",
        reviewed_at=datetime(2026, 7, 25, 1, tzinfo=timezone.utc),
    )
    assert record.decision == "deferred"
    assert record.reviewer_id == "project-owner"


def test_adapter_approval_binds_exact_source_and_license_bytes(tmp_path: Path):
    policy, policy_sha256 = _policy(
        tmp_path / "policy.json",
        portfolio_sha256="3" * 64,
    )
    records = [
        _record("abo", pii_status="not_applicable", purposes=["product_gallery"]),
        _record(
            "wildreceipt",
            pii_status="restricted",
            purposes=["capability_gold"],
        ),
    ]
    content = b"".join(canonical_json_bytes(record) for record in records)
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_bytes(content)
    verified = load_verified_source_review_ledger(
        ledger_path,
        policy,
        policy_file_sha256=policy_sha256,
        expected_ledger_file_sha256=sha256_bytes(content),
    )

    approval = verified.require_approval(
        "abo",
        source_revision="fixture-revision",
        source_lock_sha256="1" * 64,
        license_id="research-license",
        license_evidence_sha256="2" * 64,
        purposes=("product_gallery",),
        permissions=("local_embedding_allowed",),
    )
    assert approval.source_id == "abo"

    with pytest.raises(SourceReviewError, match="source_lock_sha256"):
        verified.require_approval(
            "abo",
            source_revision="fixture-revision",
            source_lock_sha256="f" * 64,
            license_id="research-license",
            license_evidence_sha256="2" * 64,
        )

    with pytest.raises(SourceReviewError, match="public_demo_allowed"):
        verified.require_approval(
            "abo",
            source_revision="fixture-revision",
            source_lock_sha256="1" * 64,
            license_id="research-license",
            license_evidence_sha256="2" * 64,
            permissions=("public_demo_allowed",),
        )


def test_v2_policy_matches_current_mvp_source_strategy() -> None:
    portfolio_path = (
        REPO_ROOT / "specs/data_sources/ecommerce-mvp-source-portfolio-v1.json"
    )
    policy_path = REPO_ROOT / "specs/data_sources/mvp-source-review-policy-v2.json"
    portfolio_sha256 = sha256_bytes(portfolio_path.read_bytes())
    policy = load_source_review_policy(
        policy_path,
        expected_policy_file_sha256=sha256_bytes(policy_path.read_bytes()),
        expected_portfolio_sha256=portfolio_sha256,
    )

    requirements = {item.source_id: item for item in policy.requirements}
    assert portfolio_sha256 == (
        "6f2a0c98a2889aa6980e3f2308cfd6a076e5bdd60e0e7e99acc77f8db74b501a"
    )
    assert policy.policy_id == "ecommerce-mvp-source-review-v2"
    assert {
        source_id
        for source_id, requirement in requirements.items()
        if requirement.required
    } == {
        "abo",
        "crosswoz",
        "durecdial_2_0",
        "fashioniq",
        "isia_food500",
        "muge",
        "rpc",
        "wikimedia_zhwiki",
        "wildreceipt",
        "xiachufang",
    }
    assert {
        source_id
        for source_id, requirement in requirements.items()
        if not requirement.required
    } == {
        "cord",
        "inaturalist",
        "sroie",
        "wikimedia_documents",
    }
    assert "jddc_2_0" not in requirements
    assert "codex_mock_trajectories" not in requirements
    assert requirements["durecdial_2_0"].required is True
    assert requirements["durecdial_2_0"].purposes == ("interaction_pattern",)
    assert requirements["durecdial_2_0"].pii_review == "required"
    assert requirements["crosswoz"].required is True
    assert requirements["crosswoz"].purposes == ("interaction_pattern",)
    assert requirements["crosswoz"].pii_review == "required"
    assert requirements["muge"].purposes == ("language_style",)
    assert requirements["xiachufang"].pii_review == "required"


def test_v2_policy_rejects_retired_source_ledger_record(tmp_path: Path) -> None:
    portfolio_path = (
        REPO_ROOT / "specs/data_sources/ecommerce-mvp-source-portfolio-v1.json"
    )
    policy_path = REPO_ROOT / "specs/data_sources/mvp-source-review-policy-v2.json"
    policy_bytes = policy_path.read_bytes()
    policy = load_source_review_policy(
        policy_path,
        expected_policy_file_sha256=sha256_bytes(policy_bytes),
        expected_portfolio_sha256=sha256_bytes(portfolio_path.read_bytes()),
    )
    record = _record(
        "jddc_2_0",
        pii_status="reviewed_no_pii",
        purposes=["language_style"],
    )
    content = canonical_json_bytes(record)
    ledger_path = tmp_path / "retired-source-ledger.jsonl"
    ledger_path.write_bytes(content)

    with pytest.raises(SourceReviewError, match="outside the policy: jddc_2_0"):
        load_verified_source_review_ledger(
            ledger_path,
            policy,
            policy_file_sha256=sha256_bytes(policy_bytes),
            expected_ledger_file_sha256=sha256_bytes(content),
        )
