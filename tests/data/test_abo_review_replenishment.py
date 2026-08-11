from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from scripts.render_abo_catalog_review import _review_storage_key
from skillchain.data.abo import (
    ABOImageManifestRecord,
    ABOImageReviewDecision,
    ABOProvenanceError,
)
from skillchain.data.abo_original_review import (
    DECISION_BUNDLE_ID,
    PACKET_ID,
    verify_abo_original_review_packet,
)
from skillchain.data.abo_review_replenishment import (
    prepare_abo_review_decision_carry_forward,
    verify_abo_review_decision_carry_forward,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)

_STATUS = "human_review_complete_pending_owner_scope_and_near_duplicate_audit"


def _packet_row(
    *,
    item_id: str,
    image_id: str,
    image_role: str,
    metadata_path: str,
    content: bytes,
) -> tuple[dict[str, object], bytes]:
    source_sha256 = sha256_bytes(content)
    uri = (
        "https://amazon-berkeley-objects.s3.amazonaws.com/"
        f"images/original/{metadata_path}"
    )
    image = ABOImageManifestRecord(
        image_id=image_id,
        path=metadata_path,
        official_image_uri=uri,
        official_image_etag=f'"etag-{image_id}"',
        source_image_sha256=source_sha256,
    )
    return (
        {
            "height": 20,
            "image_id": image_id,
            "image_record_sha256": sha256_bytes(
                canonical_json_bytes(image.model_dump(mode="json"))
            ),
            "image_role": image_role,
            "item_id": item_id,
            "listing_record_sha256": sha256_bytes(item_id.encode()),
            "local_path": f"images/original/{metadata_path}",
            "metadata_path": metadata_path,
            "official_image_etag": image.official_image_etag,
            "official_image_uri": uri,
            "packet_status": "human_decision_required",
            "parent_small_image_sha256": "0" * 64,
            "schema_version": 1,
            "source_image_sha256": source_sha256,
            "width": 30,
        },
        content,
    )


def _write_packet(
    root: Path,
    rows_with_content: list[tuple[dict[str, object], bytes]],
) -> str:
    root.mkdir()
    rows = [row for row, _ in rows_with_content]
    packet_bytes = canonical_jsonl_bytes(rows)
    (root / "review-packet.jsonl").write_bytes(packet_bytes)
    files = [
        {
            "bytes": len(packet_bytes),
            "path": "review-packet.jsonl",
            "sha256": sha256_bytes(packet_bytes),
        }
    ]
    written_paths: set[str] = set()
    for row, content in rows_with_content:
        relative = str(row["local_path"])
        if relative in written_paths:
            continue
        path = root / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written_paths.add(relative)
        files.append(
            {
                "bytes": len(content),
                "path": relative,
                "sha256": sha256_bytes(content),
            }
        )
    manifest = {
        "candidate_image_count": len(rows),
        "candidate_pair_count": len(rows) // 2,
        "files": sorted(files, key=lambda row: row["path"]),
        "formal_use_allowed": False,
        "human_decision_required": True,
        "packet_id": PACKET_ID,
        "raw_mutation_performed": False,
        "schema_version": 1,
        "status": "pending_owner_scope_and_human_catalog_photo_review",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    return sha256_bytes(manifest_bytes)


def _write_human_review(
    root: Path,
    *,
    packet_manifest_sha256: str,
    packet_rows: list[dict[str, object]],
) -> str:
    root.mkdir()
    decisions = [
        ABOImageReviewDecision(
            item_id=str(row["item_id"]),
            image_id=str(row["image_id"]),
            listing_record_sha256=str(row["listing_record_sha256"]),
            image_record_sha256=str(row["image_record_sha256"]),
            source_image_sha256=str(row["source_image_sha256"]),
            decision="approve_catalog_product_photo",
            reviewer_kind="human",
            reviewer_id="owner_1",
            reviewed_at="2026-07-25T12:00:00Z",
        )
        for row in packet_rows
    ]
    ledger_bytes = canonical_jsonl_bytes(
        row.model_dump(mode="json") for row in decisions
    )
    (root / "review-ledger.jsonl").write_bytes(ledger_bytes)
    counts = Counter(row.decision for row in decisions)
    summary = {
        "approved_pair_candidates": len(packet_rows) // 2,
        "decision_counts": dict(sorted(counts.items())),
        "formal_use_allowed": False,
        "human_decision_count": len(decisions),
        "packet_manifest_sha256": packet_manifest_sha256,
        "rejected_pair_candidates": 0,
        "reviewer_id": "owner_1",
        "schema_version": 1,
        "status": _STATUS,
        "total_pair_candidates": len(packet_rows) // 2,
    }
    summary_bytes = canonical_json_bytes(summary)
    (root / "summary.json").write_bytes(summary_bytes)
    manifest = {
        "bundle_id": DECISION_BUNDLE_ID,
        "files": [
            {
                "bytes": len(ledger_bytes),
                "path": "review-ledger.jsonl",
                "sha256": sha256_bytes(ledger_bytes),
            },
            {
                "bytes": len(summary_bytes),
                "path": "summary.json",
                "sha256": sha256_bytes(summary_bytes),
            },
        ],
        "formal_use_allowed": False,
        "packet_manifest_sha256": packet_manifest_sha256,
        "schema_version": 1,
        "status": _STATUS,
    }
    manifest["files"].sort(key=lambda row: row["path"])
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    return sha256_bytes(manifest_bytes)


def _fixture_rows() -> tuple[
    list[tuple[dict[str, object], bytes]],
    list[tuple[dict[str, object], bytes]],
]:
    old = [
        _packet_row(
            item_id="ITEM_OLD",
            image_id="OLD_MAIN",
            image_role="main",
            metadata_path="00/00000001.jpg",
            content=b"old-main",
        ),
        _packet_row(
            item_id="ITEM_OLD",
            image_id="OLD_OTHER",
            image_role="other",
            metadata_path="00/00000002.jpg",
            content=b"old-other",
        ),
    ]
    new = [
        _packet_row(
            item_id="ITEM_NEW",
            image_id="NEW_MAIN",
            image_role="main",
            metadata_path="00/00000003.jpg",
            content=b"new-main",
        ),
        _packet_row(
            item_id="ITEM_NEW",
            image_id="NEW_OTHER",
            image_role="other",
            metadata_path="00/00000004.jpg",
            content=b"new-other",
        ),
    ]
    return old, [*old, *new]


def test_review_storage_isolated_by_carry_manifest() -> None:
    target = "1" * 64
    carry_a = "2" * 64
    carry_b = "3" * 64

    assert _review_storage_key(target, None) == f"abo-catalog-photo-review:{target}"
    assert _review_storage_key(target, carry_a) != _review_storage_key(
        target,
        carry_b,
    )


def _rewrite_carry_bundle(
    root: Path,
    *,
    decisions: list[dict[str, object]],
) -> str:
    decision_bytes = canonical_jsonl_bytes(decisions)
    (root / "carried-decisions.jsonl").write_bytes(decision_bytes)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = [
        {
            "bytes": len(decision_bytes),
            "path": "carried-decisions.jsonl",
            "sha256": sha256_bytes(decision_bytes),
        }
    ]
    manifest["carried_decision_count"] = len(decisions)
    manifest["remaining_decision_count"] = manifest[
        "target_total_decision_count"
    ] - len(decisions)
    manifest.pop("manifest_self_sha256")
    manifest["manifest_self_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    return sha256_bytes(manifest_bytes)


def test_carry_forward_only_carries_exact_prior_human_decisions(
    tmp_path: Path,
) -> None:
    old_rows, target_rows = _fixture_rows()
    source_root = tmp_path / "source"
    source_sha256 = _write_packet(source_root, old_rows)
    human_root = tmp_path / "human"
    human_sha256 = _write_human_review(
        human_root,
        packet_manifest_sha256=source_sha256,
        packet_rows=[row for row, _ in old_rows],
    )
    target_root = tmp_path / "target"
    target_sha256 = _write_packet(target_root, target_rows)
    output = tmp_path / "carry"

    result = prepare_abo_review_decision_carry_forward(
        target_packet_root=target_root,
        expected_target_packet_manifest_sha256=target_sha256,
        source_packet_root=source_root,
        expected_source_packet_manifest_sha256=source_sha256,
        source_human_review_root=human_root,
        expected_source_human_review_manifest_sha256=human_sha256,
        output_dir=output,
    )
    manifest, decisions = verify_abo_review_decision_carry_forward(
        output,
        expected_manifest_sha256=result["manifest_sha256"],
        target_packet_root=target_root,
        expected_target_packet_manifest_sha256=target_sha256,
        source_packet_root=source_root,
        expected_source_packet_manifest_sha256=source_sha256,
        source_human_review_root=human_root,
        expected_source_human_review_manifest_sha256=human_sha256,
    )

    assert result["carried_decisions"] == 2
    assert result["remaining_decisions"] == 2
    assert {row.item_id for row in decisions} == {"ITEM_OLD"}
    assert manifest["formal_use_allowed"] is False
    assert manifest["manifest_self_sha256"]

    with pytest.raises(FileExistsError):
        prepare_abo_review_decision_carry_forward(
            target_packet_root=target_root,
            expected_target_packet_manifest_sha256=target_sha256,
            source_packet_root=source_root,
            expected_source_packet_manifest_sha256=source_sha256,
            source_human_review_root=human_root,
            expected_source_human_review_manifest_sha256=human_sha256,
            output_dir=output,
        )


def test_carry_verifier_rejects_self_rehashed_forgery_and_omission(
    tmp_path: Path,
) -> None:
    old_rows, target_rows = _fixture_rows()
    source_root = tmp_path / "source"
    source_sha256 = _write_packet(source_root, old_rows)
    human_root = tmp_path / "human"
    human_sha256 = _write_human_review(
        human_root,
        packet_manifest_sha256=source_sha256,
        packet_rows=[row for row, _ in old_rows],
    )
    target_root = tmp_path / "target"
    target_sha256 = _write_packet(target_root, target_rows)
    output = tmp_path / "carry"
    prepare_abo_review_decision_carry_forward(
        target_packet_root=target_root,
        expected_target_packet_manifest_sha256=target_sha256,
        source_packet_root=source_root,
        expected_source_packet_manifest_sha256=source_sha256,
        source_human_review_root=human_root,
        expected_source_human_review_manifest_sha256=human_sha256,
        output_dir=output,
    )
    original = [
        json.loads(line)
        for line in (output / "carried-decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    forged = [dict(row) for row in original]
    forged[0]["decision"] = "reject_uncertain"
    forged_manifest_sha256 = _rewrite_carry_bundle(
        output,
        decisions=forged,
    )

    with pytest.raises(ABOProvenanceError, match="authoritative source subset"):
        verify_abo_review_decision_carry_forward(
            output,
            expected_manifest_sha256=forged_manifest_sha256,
            target_packet_root=target_root,
            expected_target_packet_manifest_sha256=target_sha256,
            source_packet_root=source_root,
            expected_source_packet_manifest_sha256=source_sha256,
            source_human_review_root=human_root,
            expected_source_human_review_manifest_sha256=human_sha256,
        )

    omitted_manifest_sha256 = _rewrite_carry_bundle(
        output,
        decisions=original[1:],
    )
    with pytest.raises(ABOProvenanceError, match="authoritative source subset"):
        verify_abo_review_decision_carry_forward(
            output,
            expected_manifest_sha256=omitted_manifest_sha256,
            target_packet_root=target_root,
            expected_target_packet_manifest_sha256=target_sha256,
            source_packet_root=source_root,
            expected_source_packet_manifest_sha256=source_sha256,
            source_human_review_root=human_root,
            expected_source_human_review_manifest_sha256=human_sha256,
        )


def test_carry_verifier_rejects_wrong_external_source_digest(
    tmp_path: Path,
) -> None:
    old_rows, target_rows = _fixture_rows()
    source_root = tmp_path / "source"
    source_sha256 = _write_packet(source_root, old_rows)
    human_root = tmp_path / "human"
    human_sha256 = _write_human_review(
        human_root,
        packet_manifest_sha256=source_sha256,
        packet_rows=[row for row, _ in old_rows],
    )
    target_root = tmp_path / "target"
    target_sha256 = _write_packet(target_root, target_rows)
    output = tmp_path / "carry"
    result = prepare_abo_review_decision_carry_forward(
        target_packet_root=target_root,
        expected_target_packet_manifest_sha256=target_sha256,
        source_packet_root=source_root,
        expected_source_packet_manifest_sha256=source_sha256,
        source_human_review_root=human_root,
        expected_source_human_review_manifest_sha256=human_sha256,
        output_dir=output,
    )

    with pytest.raises(ABOProvenanceError, match="manifest identity"):
        verify_abo_review_decision_carry_forward(
            output,
            expected_manifest_sha256=result["manifest_sha256"],
            target_packet_root=target_root,
            expected_target_packet_manifest_sha256=target_sha256,
            source_packet_root=source_root,
            expected_source_packet_manifest_sha256="0" * 64,
            source_human_review_root=human_root,
            expected_source_human_review_manifest_sha256=human_sha256,
        )


def test_original_packet_verifier_rejects_symlink_in_tree(
    tmp_path: Path,
) -> None:
    old_rows, _ = _fixture_rows()
    packet_root = tmp_path / "packet"
    packet_sha256 = _write_packet(packet_root, old_rows)
    link = packet_root / "linked-image.jpg"
    try:
        link.symlink_to(packet_root / str(old_rows[0][0]["local_path"]))
    except OSError:
        pytest.skip("local environment does not permit symlink creation")

    with pytest.raises(ABOProvenanceError, match="links or junctions"):
        verify_abo_original_review_packet(
            packet_root,
            expected_manifest_sha256=packet_sha256,
        )


def test_carry_forward_rejects_changed_content_under_same_identity(
    tmp_path: Path,
) -> None:
    old_rows, target_rows = _fixture_rows()
    source_root = tmp_path / "source"
    source_sha256 = _write_packet(source_root, old_rows)
    human_root = tmp_path / "human"
    human_sha256 = _write_human_review(
        human_root,
        packet_manifest_sha256=source_sha256,
        packet_rows=[row for row, _ in old_rows],
    )
    changed, content = _packet_row(
        item_id="ITEM_OLD",
        image_id="OLD_MAIN",
        image_role="main",
        metadata_path="00/00000001.jpg",
        content=b"changed-main",
    )
    target_rows[0] = (changed, content)
    target_root = tmp_path / "target"
    target_sha256 = _write_packet(target_root, target_rows)

    with pytest.raises(ABOProvenanceError, match="changed content"):
        prepare_abo_review_decision_carry_forward(
            target_packet_root=target_root,
            expected_target_packet_manifest_sha256=target_sha256,
            source_packet_root=source_root,
            expected_source_packet_manifest_sha256=source_sha256,
            source_human_review_root=human_root,
            expected_source_human_review_manifest_sha256=human_sha256,
            output_dir=tmp_path / "carry",
        )


def test_tracked_replenishment_receipt_is_canonical_and_self_hashed() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[2]
        / "specs"
        / "data_sources"
        / "c2"
        / "adapters"
        / "abo-original-review-replenishment-v2.receipt.json"
    )
    content = receipt_path.read_bytes()
    receipt = json.loads(content)

    assert content == canonical_json_bytes(receipt)
    assert (
        sha256_bytes(content)
        == "5cbd8b1db07367ae5b9c7a7fa59ffa21ce1bdccbd0b4e49ad6935f23698562ba"
    )
    unsigned = dict(receipt)
    receipt_self_sha256 = unsigned.pop("receipt_self_sha256")
    assert receipt_self_sha256 == sha256_bytes(canonical_json_bytes(unsigned))
    assert receipt["status"] == "pending_human_review_of_new_images"
    assert receipt["formal_use_allowed"] is False
    assert receipt["carry_forward"]["carried_decision_count"] == 100
    assert receipt["carry_forward"]["remaining_decision_count"] == 20


def test_tracked_v2_finalization_receipt_closes_only_capacity_gate() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[2]
        / "specs"
        / "data_sources"
        / "c2"
        / "adapters"
        / "abo-original-review-v2-finalization.receipt.json"
    )
    content = receipt_path.read_bytes()
    receipt = json.loads(content)

    assert content == canonical_json_bytes(receipt)
    assert (
        sha256_bytes(content)
        == "10796ef09b6911c29b88863fea07bb19af7b3632d1019c2c5b6f66c5ebdcb06c"
    )
    unsigned = dict(receipt)
    receipt_self_sha256 = unsigned.pop("receipt_self_sha256")
    assert receipt_self_sha256 == sha256_bytes(canonical_json_bytes(unsigned))
    assert receipt["capacity_gate"] == {
        "minimum_retained_pair_count": 35,
        "passed": True,
    }
    assert receipt["audit"]["approved_pair_count"] == 44
    assert receipt["audit"]["retained_pair_count"] == 41
    assert receipt["formal_use_allowed"] is False
    assert receipt["status"] == (
        "exact_match_capacity_gate_closed_pending_exact_scope_approval"
    )
    assert "owner_exact_original_scope_approval" in receipt["remaining_gates"]

    audit_path = receipt_path.with_name(
        "abo-original-pair-leakage-audit-v2.receipt.json"
    )
    audit_content = audit_path.read_bytes()
    assert sha256_bytes(audit_content) == receipt["audit"]["file_sha256"]
    audit = json.loads(audit_content)
    audit_unsigned = dict(audit)
    audit_self_sha256 = audit_unsigned.pop("receipt_self_sha256")
    assert audit_self_sha256 == receipt["audit"]["receipt_self_sha256"]
    assert audit_self_sha256 == sha256_bytes(
        canonical_json_bytes(audit_unsigned)
    )
    assert audit["formal_use_allowed"] is False
    assert audit["retained_pair_count"] == receipt["audit"]["retained_pair_count"]
    assert audit["excluded_pair_ids"] == receipt["audit"]["excluded_pair_ids"]
