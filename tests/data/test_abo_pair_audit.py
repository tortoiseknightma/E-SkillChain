from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from skillchain.data.abo import (
    ABOImageManifestRecord,
    ABOProvenanceError,
)
from skillchain.data.abo_original_review import (
    finalize_abo_original_review_decisions,
)
from skillchain.data.abo_pair_audit import (
    _maximum_independent_pair_ids,
    audit_abo_original_pairs,
)
from skillchain.data.asset_catalog import NearDuplicatePolicy
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    sha256_bytes,
)


def _pattern_bytes(pattern: str) -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    pixels = image.load()
    assert pixels is not None
    for y in range(64):
        for x in range(64):
            black = (
                (pattern == "diagonal" and x < y)
                or (pattern == "checker" and (x // 8 + y // 8) % 2 == 0)
                or (pattern == "vertical" and x < 32)
                or (pattern == "horizontal" and y < 32)
                or (pattern == "frame" and (x < 8 or x >= 56 or y < 8 or y >= 56))
                or (pattern == "cross" and (28 <= x < 36 or 28 <= y < 36))
            )
            if black:
                pixels[x, y] = (0, 0, 0)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _publish_packet_and_human_review(tmp_path: Path):
    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    image_specs = {
        "a-main": ("A_MAIN", "main", "diagonal", "images/original/a-main.png"),
        "shared": ("SHARED", "other", "checker", "images/original/shared.png"),
        "b-main": ("B_MAIN", "main", "vertical", "images/original/b-main.png"),
        "c-main": ("C_MAIN", "main", "frame", "images/original/c-main.png"),
        "c-other": ("C_OTHER", "other", "frame", "images/original/c-other.png"),
        "d-main": ("D_MAIN", "main", "cross", "images/original/d-main.png"),
        "d-other": (
            "D_OTHER",
            "other",
            "horizontal",
            "images/original/d-other.png",
        ),
    }
    contents: dict[str, bytes] = {}
    for key, (_image_id, _role, pattern, local_path) in image_specs.items():
        content = _pattern_bytes(pattern)
        contents[key] = content
        path = packet_root / local_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    pair_specs = {
        "ITEM-A": ("a-main", "shared"),
        "ITEM-B": ("b-main", "shared"),
        "ITEM-C": ("c-main", "c-other"),
        "ITEM-D": ("d-main", "d-other"),
    }
    rows = []
    for item_id, keys in pair_specs.items():
        listing_sha = sha256_bytes(canonical_json_bytes({"item_id": item_id}))
        for expected_role, key in zip(("main", "other"), keys, strict=True):
            image_id, role, _pattern, local_path = image_specs[key]
            assert role == expected_role
            relative = local_path.removeprefix("images/original/")
            source_sha = sha256_bytes(contents[key])
            record = ABOImageManifestRecord(
                image_id=image_id,
                path=relative,
                official_image_uri=(
                    "https://amazon-berkeley-objects.s3.amazonaws.com/"
                    f"images/original/{relative}"
                ),
                official_image_etag=f'"{key}-etag"',
                source_image_sha256=source_sha,
            )
            rows.append(
                {
                    "height": 64,
                    "image_id": image_id,
                    "image_record_sha256": sha256_bytes(
                        canonical_json_bytes(record.model_dump(mode="json"))
                    ),
                    "image_role": role,
                    "item_id": item_id,
                    "listing_record_sha256": listing_sha,
                    "local_path": local_path,
                    "metadata_path": relative,
                    "official_image_etag": record.official_image_etag,
                    "official_image_uri": record.official_image_uri,
                    "packet_status": "human_decision_required",
                    "parent_small_image_sha256": "0" * 64,
                    "schema_version": 1,
                    "source_image_sha256": source_sha,
                    "width": 64,
                }
            )
    rows.sort(key=lambda row: (row["item_id"], row["image_id"]))
    packet_bytes = canonical_jsonl_bytes(rows)
    (packet_root / "review-packet.jsonl").write_bytes(packet_bytes)
    files = [
        {
            "bytes": len(packet_bytes),
            "path": "review-packet.jsonl",
            "sha256": sha256_bytes(packet_bytes),
        }
    ]
    for _key, (_image_id, _role, _pattern, local_path) in image_specs.items():
        content = (packet_root / local_path).read_bytes()
        files.append(
            {
                "bytes": len(content),
                "path": local_path,
                "sha256": sha256_bytes(content),
            }
        )
    manifest = {
        "acquisition_policy_version": "abo-targeted-original-http-acquisition-v1",
        "candidate_image_count": len(rows),
        "candidate_pair_count": len(pair_specs),
        "files": sorted(files, key=lambda row: row["path"]),
        "formal_use_allowed": False,
        "human_decision_required": True,
        "packet_id": "abo-targeted-original-review-v1",
        "parent_compact_manifest_sha256": "1" * 64,
        "raw_mutation_performed": True,
        "schema_version": 1,
        "status": "pending_owner_scope_and_human_catalog_photo_review",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (packet_root / "manifest.json").write_bytes(manifest_bytes)
    packet_sha = sha256_bytes(manifest_bytes)

    decisions = [
        {
            "decision": "approve_catalog_product_photo",
            "image_id": row["image_id"],
            "image_record_sha256": row["image_record_sha256"],
            "item_id": row["item_id"],
            "listing_record_sha256": row["listing_record_sha256"],
            "review_policy_version": "abo-catalog-photo-human-review-v1",
            "reviewed_at": "2026-07-25T12:00:00Z",
            "reviewer_id": "owner_1",
            "reviewer_kind": "human",
            "schema_version": 1,
            "source_image_sha256": row["source_image_sha256"],
        }
        for row in rows
    ]
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_bytes(
        b"".join(
            (json.dumps(row, separators=(",", ":")) + "\n").encode()
            for row in decisions
        )
    )
    human_root = tmp_path / "human"
    finalized = finalize_abo_original_review_decisions(
        packet_root=packet_root,
        expected_packet_manifest_sha256=packet_sha,
        decisions_path=decisions_path,
        output_dir=human_root,
    )
    return packet_root, packet_sha, human_root, finalized["manifest_sha256"]


def test_pair_audit_publishes_canonical_exact_mis_receipt(tmp_path: Path) -> None:
    packet_root, packet_sha, human_root, human_sha = _publish_packet_and_human_review(
        tmp_path
    )
    output = tmp_path / "pair-audit.json"

    result = audit_abo_original_pairs(
        packet_root=packet_root,
        expected_packet_manifest_sha256=packet_sha,
        human_review_root=human_root,
        expected_human_review_manifest_sha256=human_sha,
        output_path=output,
        near_duplicate_policy=NearDuplicatePolicy(
            max_phash_hamming_distance=0,
        ),
    )

    receipt_bytes = output.read_bytes()
    receipt = parse_canonical_json(receipt_bytes, label="test pair audit")
    assert isinstance(receipt, dict)
    assert result["receipt_sha256"] == sha256_bytes(receipt_bytes)
    assert receipt["approved_pair_count"] == 4
    assert receipt["unique_image_count"] == 7
    assert receipt["within_pair_conflict_count"] == 1
    assert receipt["within_pair_conflicts"][0]["pair_id"] == "ITEM-C"
    assert receipt["cross_pair_conflict_count"] == 1
    assert receipt["cross_pair_conflicts"][0]["left_pair_id"] == "ITEM-A"
    assert receipt["cross_pair_conflicts"][0]["right_pair_id"] == "ITEM-B"
    assert receipt["retained_pair_ids"] == ["ITEM-A", "ITEM-D"]
    assert receipt["excluded_pair_ids"] == ["ITEM-B", "ITEM-C"]
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_self_sha256"
    }
    assert receipt["receipt_self_sha256"] == sha256_bytes(
        canonical_json_bytes(unsigned)
    )
    with pytest.raises(FileExistsError, match="already exists"):
        audit_abo_original_pairs(
            packet_root=packet_root,
            expected_packet_manifest_sha256=packet_sha,
            human_review_root=human_root,
            expected_human_review_manifest_sha256=human_sha,
            output_path=output,
        )


def test_exact_mis_is_deterministic_and_lexicographically_smallest() -> None:
    edges = {("A", "B"), ("B", "C"), ("D", "E")}

    result = _maximum_independent_pair_ids(
        ("E", "D", "C", "B", "A", "F"),
        edges,
    )

    assert result == ("A", "C", "D", "F")


def test_pair_audit_rejects_external_digest_drift(tmp_path: Path) -> None:
    packet_root, packet_sha, human_root, human_sha = _publish_packet_and_human_review(
        tmp_path
    )

    with pytest.raises(ABOProvenanceError, match="digest mismatch"):
        audit_abo_original_pairs(
            packet_root=packet_root,
            expected_packet_manifest_sha256=packet_sha,
            human_review_root=human_root,
            expected_human_review_manifest_sha256="f" * 64,
            output_path=tmp_path / "pair-audit.json",
        )
    assert not (tmp_path / "pair-audit.json").exists()
    assert len(human_sha) == 64


def test_pair_audit_rejects_packet_byte_drift(tmp_path: Path) -> None:
    packet_root, packet_sha, human_root, human_sha = _publish_packet_and_human_review(
        tmp_path
    )
    image = packet_root / "images" / "original" / "a-main.png"
    image.write_bytes(_pattern_bytes("cross"))

    with pytest.raises((ABOProvenanceError, ValueError), match="drift|differs|SHA"):
        audit_abo_original_pairs(
            packet_root=packet_root,
            expected_packet_manifest_sha256=packet_sha,
            human_review_root=human_root,
            expected_human_review_manifest_sha256=human_sha,
            output_path=tmp_path / "pair-audit.json",
        )


def test_pair_audit_rejects_symlink_output(tmp_path: Path) -> None:
    packet_root, packet_sha, human_root, human_sha = _publish_packet_and_human_review(
        tmp_path
    )
    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "pair-audit.json"
    try:
        output.symlink_to(existing)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(FileExistsError, match="already exists"):
        audit_abo_original_pairs(
            packet_root=packet_root,
            expected_packet_manifest_sha256=packet_sha,
            human_review_root=human_root,
            expected_human_review_manifest_sha256=human_sha,
            output_path=output,
        )
    assert os.path.lexists(output)
