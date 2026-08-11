"""Acquire exact ABO originals for a compact-archive review selection.

The compact packet is useful for deterministic identity-only selection, but
``abo-images-small.tar`` contains 256-pixel previews.  This module resolves the
same selected image identities to the official per-image ``images/original``
objects, records their HTTP and content identities, and publishes a
self-contained human-review packet without treating the new scope as approved.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image

from skillchain.data.abo import (
    ABOImageManifestRecord,
    ABOImageReviewDecision,
    ABOListingRecord,
    ABOProvenanceError,
)
from skillchain.data.abo_archive_review import verify_abo_archive_review_packet
from skillchain.data.source_lock import stable_file_digest
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)
from skillchain.tools.serialization import (
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
)

PACKET_ID = "abo-targeted-original-review-v1"
DECISION_BUNDLE_ID = "abo-targeted-original-human-review-v1"
ACQUISITION_POLICY_VERSION = "abo-targeted-original-http-acquisition-v1"
REPLENISHMENT_POLICY_VERSION = "abo-original-replenishment-carry-bytes-v1"
_OFFICIAL_ROOT = "https://amazon-berkeley-objects.s3.amazonaws.com/images/original"
_SMALL_MEMBER = re.compile(r"images/small/([0-9a-f]{2}/[0-9a-f]{8}\.(?:jpg|png))")
_METADATA_PATH = re.compile(r"[0-9a-f]{2}/[0-9a-f]{8}\.(?:jpg|png)")
_MAX_IMAGE_BYTES = 50 * 1024 * 1024


def prepare_abo_original_review_packet(
    *,
    compact_packet_root: Path | str,
    expected_compact_manifest_sha256: str,
    raw_root: Path | str,
    output_dir: Path | str,
    maximum_workers: int = 8,
) -> dict[str, Any]:
    """Download selected official originals and publish a review-only packet."""

    if not 1 <= maximum_workers <= 32:
        raise ValueError("maximum_workers must be between 1 and 32")
    compact_packet_root = Path(compact_packet_root).resolve(strict=True)
    raw_root = Path(raw_root).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"ABO original review output already exists: {output_dir}"
        )
    raw_images_dir = raw_root / "images"
    if os.path.lexists(raw_images_dir):
        raise FileExistsError(
            "targeted original acquisition requires an absent raw images directory"
        )
    compact_manifest = verify_abo_archive_review_packet(
        compact_packet_root,
        expected_manifest_sha256=expected_compact_manifest_sha256,
    )
    compact_rows = parse_canonical_jsonl(
        (compact_packet_root / "review-packet.jsonl").read_bytes(),
        label="ABO compact review packet",
    )
    listing_records = _load_listing_records(raw_root, compact_rows)
    targets = _original_targets(compact_rows)
    output_staging = new_staging_directory(output_dir)
    raw_staging = new_staging_directory(raw_images_dir)
    try:
        downloaded = _download_targets(
            targets,
            output_staging / "images" / "original",
            maximum_workers=maximum_workers,
        )
        packet_rows = _build_packet_rows(
            compact_rows,
            listing_records=listing_records,
            downloaded=downloaded,
        )
        receipt_rows = [downloaded[path] for path in sorted(downloaded)]
        packet_bytes = canonical_jsonl_bytes(packet_rows)
        receipt_bytes = canonical_jsonl_bytes(receipt_rows)
        (output_staging / "review-packet.jsonl").write_bytes(packet_bytes)
        (output_staging / "download-receipt.jsonl").write_bytes(receipt_bytes)
        files = [
            {
                "bytes": len(packet_bytes),
                "path": "review-packet.jsonl",
                "sha256": sha256_bytes(packet_bytes),
            },
            {
                "bytes": len(receipt_bytes),
                "path": "download-receipt.jsonl",
                "sha256": sha256_bytes(receipt_bytes),
            },
        ]
        for relative, receipt in sorted(downloaded.items()):
            files.append(
                {
                    "bytes": receipt["bytes"],
                    "path": f"images/original/{relative}",
                    "sha256": receipt["source_image_sha256"],
                }
            )
        manifest = {
            "acquisition_policy_version": ACQUISITION_POLICY_VERSION,
            "candidate_image_count": len(packet_rows),
            "candidate_pair_count": compact_manifest["candidate_pair_count"],
            "files": sorted(files, key=lambda row: row["path"]),
            "formal_use_allowed": False,
            "human_decision_required": True,
            "packet_id": PACKET_ID,
            "parent_compact_manifest_sha256": expected_compact_manifest_sha256,
            "raw_mutation_performed": True,
            "schema_version": 1,
            "status": "pending_owner_scope_and_human_catalog_photo_review",
        }
        manifest_bytes = canonical_json_bytes(manifest)
        (output_staging / "manifest.json").write_bytes(manifest_bytes)

        raw_original_root = raw_staging / "original"
        raw_original_root.mkdir()
        for relative in sorted(downloaded):
            source = output_staging / "images" / "original" / Path(*relative.split("/"))
            target = raw_original_root / Path(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        _verify_raw_copy(raw_original_root, downloaded)
        atomic_publish_new_directory(raw_staging, raw_images_dir)
        atomic_publish_new_directory(output_staging, output_dir)
    except BaseException:
        for staging in (output_staging, raw_staging):
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "candidate_images": len(packet_rows),
        "candidate_pairs": manifest["candidate_pair_count"],
        "manifest_path": str(output_dir / "manifest.json"),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "output_dir": str(output_dir),
        "status": manifest["status"],
    }


def prepare_abo_original_replenishment_packet(
    *,
    compact_packet_root: Path | str,
    expected_compact_manifest_sha256: str,
    raw_root: Path | str,
    previous_packet_root: Path | str,
    expected_previous_packet_manifest_sha256: str,
    output_dir: Path | str,
    maximum_workers: int = 8,
) -> dict[str, Any]:
    """Publish an expanded packet while reusing verified original bytes.

    Reuse is limited to official image objects whose metadata path, image ID,
    content digest, byte length, ETag, and prior packet file descriptor all
    agree.  This function carries no human decision; new identities always
    remain ``human_decision_required``.
    """

    if not 1 <= maximum_workers <= 32:
        raise ValueError("maximum_workers must be between 1 and 32")
    compact_packet_root = Path(compact_packet_root).absolute()
    raw_root = Path(raw_root).absolute()
    previous_packet_root = Path(previous_packet_root).absolute()
    output_dir = Path(output_dir).absolute()
    for root, label in (
        (compact_packet_root, "ABO compact review packet"),
        (raw_root, "ABO raw root"),
        (previous_packet_root, "ABO previous original review packet"),
    ):
        _reject_links_in_tree(root, label)
    _require_real_directory(output_dir.parent, "ABO replenishment output parent")
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"ABO original replenishment output already exists: {output_dir}"
        )

    compact_manifest = verify_abo_archive_review_packet(
        compact_packet_root,
        expected_manifest_sha256=expected_compact_manifest_sha256,
    )
    verify_abo_original_review_packet(
        previous_packet_root,
        expected_manifest_sha256=expected_previous_packet_manifest_sha256,
    )
    compact_rows = parse_canonical_jsonl(
        read_stable_regular_file(
            compact_packet_root / "review-packet.jsonl",
            label="ABO compact replenishment packet",
            max_bytes=4 * 1024 * 1024,
        ),
        label="ABO compact replenishment packet",
    )
    previous_rows = parse_canonical_jsonl(
        read_stable_regular_file(
            previous_packet_root / "review-packet.jsonl",
            label="ABO previous original packet",
            max_bytes=4 * 1024 * 1024,
        ),
        label="ABO previous original packet",
    )
    previous_receipts = _load_reusable_receipts(
        previous_packet_root,
        previous_rows,
    )
    listing_records = _load_listing_records(raw_root, compact_rows)
    targets = _original_targets(compact_rows)
    reusable = {
        relative: previous_receipts[relative]
        for relative, target in targets.items()
        if relative in previous_receipts
        and previous_receipts[relative]["image_id"] == target["image_id"]
        and previous_receipts[relative]["official_image_uri"]
        == target["official_image_uri"]
    }
    new_targets = {
        relative: target
        for relative, target in targets.items()
        if relative not in reusable
    }

    staging = new_staging_directory(output_dir)
    try:
        image_root = staging / "images" / "original"
        image_root.mkdir(parents=True)
        for relative, receipt in sorted(reusable.items()):
            source = (
                previous_packet_root
                / "images"
                / "original"
                / Path(*relative.split("/"))
            )
            destination = image_root / Path(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            digest, size = stable_file_digest(
                destination,
                label=f"ABO reused original {relative}",
            )
            if digest != receipt["source_image_sha256"] or size != receipt["bytes"]:
                raise ABOProvenanceError(
                    "ABO reused original differs after packet copy"
                )
        downloaded = _download_targets(
            new_targets,
            image_root,
            maximum_workers=maximum_workers,
        )
        all_receipts = {**reusable, **downloaded}
        packet_rows = _build_packet_rows(
            compact_rows,
            listing_records=listing_records,
            downloaded=all_receipts,
        )
        _verify_reused_row_identities(
            previous_rows,
            packet_rows,
        )
        receipt_rows = [all_receipts[path] for path in sorted(all_receipts)]
        packet_bytes = canonical_jsonl_bytes(packet_rows)
        receipt_bytes = canonical_jsonl_bytes(receipt_rows)
        (staging / "review-packet.jsonl").write_bytes(packet_bytes)
        (staging / "download-receipt.jsonl").write_bytes(receipt_bytes)
        files = [
            {
                "bytes": len(packet_bytes),
                "path": "review-packet.jsonl",
                "sha256": sha256_bytes(packet_bytes),
            },
            {
                "bytes": len(receipt_bytes),
                "path": "download-receipt.jsonl",
                "sha256": sha256_bytes(receipt_bytes),
            },
        ]
        for relative, receipt in sorted(all_receipts.items()):
            files.append(
                {
                    "bytes": receipt["bytes"],
                    "path": f"images/original/{relative}",
                    "sha256": receipt["source_image_sha256"],
                }
            )
        reused_rows = sum(
            1 for row in compact_rows if _compact_metadata_path(row) in reusable
        )
        manifest = {
            "acquisition_policy_version": ACQUISITION_POLICY_VERSION,
            "candidate_image_count": len(packet_rows),
            "candidate_pair_count": compact_manifest["candidate_pair_count"],
            "downloaded_unique_image_count": len(downloaded),
            "files": sorted(files, key=lambda row: row["path"]),
            "formal_use_allowed": False,
            "human_decision_required": True,
            "packet_id": PACKET_ID,
            "parent_compact_manifest_sha256": expected_compact_manifest_sha256,
            "parent_original_manifest_sha256": (
                expected_previous_packet_manifest_sha256
            ),
            "raw_mutation_performed": False,
            "replenishment_policy_version": REPLENISHMENT_POLICY_VERSION,
            "reused_candidate_image_count": reused_rows,
            "reused_unique_image_count": len(reusable),
            "schema_version": 1,
            "status": "pending_owner_scope_and_human_catalog_photo_review",
        }
        manifest["manifest_self_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        manifest_bytes = canonical_json_bytes(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        _reject_links_in_tree(compact_packet_root, "ABO compact review packet")
        _reject_links_in_tree(
            previous_packet_root,
            "ABO previous original review packet",
        )
        verify_abo_archive_review_packet(
            compact_packet_root,
            expected_manifest_sha256=expected_compact_manifest_sha256,
        )
        verify_abo_original_review_packet(
            previous_packet_root,
            expected_manifest_sha256=expected_previous_packet_manifest_sha256,
        )
        atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "candidate_images": len(packet_rows),
        "candidate_pairs": manifest["candidate_pair_count"],
        "downloaded_unique_images": len(downloaded),
        "manifest_path": str(output_dir / "manifest.json"),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "output_dir": str(output_dir),
        "reused_candidate_images": reused_rows,
        "reused_unique_images": len(reusable),
        "status": manifest["status"],
    }


def verify_abo_original_review_packet(
    root: Path | str,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify a published original-resolution review packet."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None:
        raise ValueError("ABO original review digest must be lowercase SHA-256")
    root = Path(root).absolute()
    _reject_links_in_tree(root, "ABO original review packet")
    manifest_bytes = read_stable_regular_file(
        root / "manifest.json",
        label="ABO original review manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        raise ABOProvenanceError("ABO original review manifest digest mismatch")
    manifest = parse_canonical_json(
        manifest_bytes,
        label="ABO original review manifest",
    )
    if isinstance(manifest, dict) and "manifest_self_sha256" in manifest:
        manifest_without_self_hash = dict(manifest)
        manifest_self_sha256 = manifest_without_self_hash.pop("manifest_self_sha256")
        if not isinstance(
            manifest_self_sha256, str
        ) or manifest_self_sha256 != sha256_bytes(
            canonical_json_bytes(manifest_without_self_hash)
        ):
            raise ABOProvenanceError(
                "ABO original review manifest self digest mismatch"
            )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("packet_id") != PACKET_ID
        or manifest.get("formal_use_allowed") is not False
        or manifest.get("human_decision_required") is not True
        or manifest.get("status")
        != "pending_owner_scope_and_human_catalog_photo_review"
    ):
        raise ABOProvenanceError("ABO original review manifest is invalid")
    expected_paths = {"manifest.json"}
    for row in manifest.get("files", []):
        if not isinstance(row, dict):
            raise ABOProvenanceError("ABO original review file row is invalid")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            raise ABOProvenanceError("ABO original review path is invalid")
        digest, size = stable_file_digest(
            root / Path(*relative.split("/")),
            label=f"ABO original review {relative}",
        )
        if digest != row.get("sha256") or size != row.get("bytes"):
            raise ABOProvenanceError(f"ABO original review payload drifted: {relative}")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ABOProvenanceError("ABO original review file set drifted")
    rows = parse_canonical_jsonl(
        (root / "review-packet.jsonl").read_bytes(),
        label="ABO original review packet",
    )
    roles: dict[str, set[str]] = {}
    for row in rows:
        roles.setdefault(row["item_id"], set()).add(row["image_role"])
        record = ABOImageManifestRecord.model_validate(
            {
                "image_id": row["image_id"],
                "path": row["metadata_path"],
                "official_image_uri": row["official_image_uri"],
                "official_image_etag": row["official_image_etag"],
                "source_image_sha256": row["source_image_sha256"],
            }
        )
        if (
            sha256_bytes(canonical_json_bytes(record.model_dump(mode="json")))
            != row["image_record_sha256"]
        ):
            raise ABOProvenanceError("ABO original image record digest differs")
    if (
        len(rows) != manifest.get("candidate_image_count")
        or len(roles) != manifest.get("candidate_pair_count")
        or any(value != {"main", "other"} for value in roles.values())
    ):
        raise ABOProvenanceError("ABO original review pair coverage differs")
    return manifest


def finalize_abo_original_review_decisions(
    *,
    packet_root: Path | str,
    expected_packet_manifest_sha256: str,
    decisions_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Validate complete human decisions and publish a canonical ledger."""

    packet_root = Path(packet_root).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"ABO human review output already exists: {output_dir}")
    packet_manifest = verify_abo_original_review_packet(
        packet_root,
        expected_manifest_sha256=expected_packet_manifest_sha256,
    )
    packet_rows = parse_canonical_jsonl(
        (packet_root / "review-packet.jsonl").read_bytes(),
        label="ABO original review packet",
    )
    decision_snapshot = read_stable_regular_file(
        Path(decisions_path),
        label="ABO exported human decisions",
        max_bytes=4 * 1024 * 1024,
    )
    decisions = _parse_exported_decisions(decision_snapshot)
    expected = {(row["item_id"], row["image_id"]): row for row in packet_rows}
    actual = {(row.item_id, row.image_id): row for row in decisions}
    if len(actual) != len(decisions):
        raise ABOProvenanceError("ABO human decisions contain duplicate identities")
    if set(actual) != set(expected):
        raise ABOProvenanceError("ABO human decisions do not exactly cover the packet")
    for key, decision in actual.items():
        packet = expected[key]
        if (
            decision.listing_record_sha256 != packet["listing_record_sha256"]
            or decision.image_record_sha256 != packet["image_record_sha256"]
            or decision.source_image_sha256 != packet["source_image_sha256"]
        ):
            raise ABOProvenanceError(
                "ABO human decision identity does not match the original packet"
            )
    reviewers = {row.reviewer_id for row in decisions}
    if len(reviewers) != 1:
        raise ABOProvenanceError("ABO human review must name one accountable reviewer")
    ordered = tuple(sorted(decisions, key=lambda row: (row.item_id, row.image_id)))
    decision_counts = Counter(row.decision for row in ordered)
    roles_by_item: dict[str, dict[str, ABOImageReviewDecision]] = {}
    for packet in packet_rows:
        roles_by_item.setdefault(packet["item_id"], {})[packet["image_role"]] = actual[
            (packet["item_id"], packet["image_id"])
        ]
    approved_pairs = sum(
        1
        for roles in roles_by_item.values()
        if all(
            roles[role].decision == "approve_catalog_product_photo"
            for role in ("main", "other")
        )
    )
    ledger_bytes = canonical_jsonl_bytes(row.model_dump(mode="json") for row in ordered)
    summary = {
        "approved_pair_candidates": approved_pairs,
        "decision_counts": dict(sorted(decision_counts.items())),
        "formal_use_allowed": False,
        "human_decision_count": len(ordered),
        "packet_manifest_sha256": expected_packet_manifest_sha256,
        "rejected_pair_candidates": len(roles_by_item) - approved_pairs,
        "reviewer_id": next(iter(reviewers)),
        "schema_version": 1,
        "status": (
            "human_review_complete_pending_owner_scope_and_near_duplicate_audit"
        ),
        "total_pair_candidates": len(roles_by_item),
    }
    summary_bytes = canonical_json_bytes(summary)
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
        "packet_manifest_sha256": expected_packet_manifest_sha256,
        "schema_version": 1,
        "status": summary["status"],
    }
    manifest["files"].sort(key=lambda row: row["path"])
    manifest_bytes = canonical_json_bytes(manifest)
    staging = new_staging_directory(output_dir)
    try:
        (staging / "review-ledger.jsonl").write_bytes(ledger_bytes)
        (staging / "summary.json").write_bytes(summary_bytes)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    if (
        read_stable_regular_file(
            Path(decisions_path),
            label="ABO exported human decisions",
            max_bytes=4 * 1024 * 1024,
        )
        != decision_snapshot
    ):
        raise ABOProvenanceError("ABO exported human decisions changed during publish")
    return {
        **summary,
        "manifest_path": str(output_dir / "manifest.json"),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "output_dir": str(output_dir),
        "packet_candidate_images": packet_manifest["candidate_image_count"],
    }


def _parse_exported_decisions(content: bytes) -> tuple[ABOImageReviewDecision, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ABOProvenanceError("ABO exported decisions must be UTF-8") from error
    if not text or not text.endswith("\n"):
        raise ABOProvenanceError(
            "ABO exported decisions must be non-empty newline-terminated JSONL"
        )
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ABOProvenanceError("ABO exported decisions contain a blank row")
        try:
            raw = json.loads(line, object_pairs_hook=_unique_json_object)
            records.append(ABOImageReviewDecision.model_validate(raw, strict=True))
        except Exception as error:
            raise ABOProvenanceError(
                f"ABO exported decision row {line_number} is invalid"
            ) from error
    return tuple(records)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _original_targets(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    for row in rows:
        match = _SMALL_MEMBER.fullmatch(row["archive_member"])
        if match is None:
            raise ABOProvenanceError("compact review row lacks a canonical small image")
        relative = match.group(1)
        value = {
            "image_id": row["image_id"],
            "official_image_uri": f"{_OFFICIAL_ROOT}/{relative}",
        }
        previous = targets.setdefault(relative, value)
        if previous != value:
            raise ABOProvenanceError("ABO original path has contradictory identities")
    return targets


def _download_targets(
    targets: dict[str, dict[str, str]],
    destination: Path,
    *,
    maximum_workers: int,
) -> dict[str, dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                relative,
                target,
                destination,
            ): relative
            for relative, target in targets.items()
        }
        for future in as_completed(futures):
            relative, receipt = future.result()
            downloaded[relative] = receipt
    return downloaded


def _download_one(
    relative: str,
    target: dict[str, str],
    destination: Path,
) -> tuple[str, dict[str, Any]]:
    request = Request(
        target["official_image_uri"],
        headers={"User-Agent": "ECommerceSkillChain-ABO-review/1"},
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - exact official URI
        if response.status != 200:
            raise ABOProvenanceError(
                f"ABO original image returned HTTP {response.status}"
            )
        content = response.read(_MAX_IMAGE_BYTES + 1)
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        content_type = response.headers.get("Content-Type")
    if not content or len(content) > _MAX_IMAGE_BYTES or not etag:
        raise ABOProvenanceError("ABO original image response identity is invalid")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except Exception as error:
        raise ABOProvenanceError("ABO original image is not decodable") from error
    path = destination / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return relative, {
        "bytes": len(content),
        "content_type": content_type,
        "height": height,
        "image_id": target["image_id"],
        "last_modified": last_modified,
        "metadata_path": relative,
        "official_image_etag": etag,
        "official_image_uri": target["official_image_uri"],
        "schema_version": 1,
        "source_image_sha256": hashlib.sha256(content).hexdigest(),
        "width": width,
    }


def _load_listing_records(
    raw_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, ABOListingRecord]:
    locators: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["listing_member"], row["listing_line_number"])
        locators[key] = row
    found: dict[str, ABOListingRecord] = {}
    by_member: dict[str, dict[int, dict[str, Any]]] = {}
    for (member, line), row in locators.items():
        by_member.setdefault(member, {})[line] = row
    with tarfile.open(raw_root / "archives" / "abo-listings.tar", mode="r:") as archive:
        for member_name, wanted in by_member.items():
            extracted = archive.extractfile(archive.getmember(member_name))
            if extracted is None:
                raise ABOProvenanceError("ABO listing shard cannot be read")
            import gzip

            with gzip.GzipFile(fileobj=extracted) as decompressed:
                for line_number, raw_line in enumerate(decompressed, start=1):
                    if line_number not in wanted:
                        continue
                    expected = wanted[line_number]
                    if sha256_bytes(raw_line) != expected["listing_raw_sha256"]:
                        raise ABOProvenanceError("ABO selected listing bytes drifted")
                    value = json.loads(raw_line)
                    record = ABOListingRecord.model_validate(
                        {
                            "item_id": value["item_id"],
                            "main_image_id": value["main_image_id"],
                            "other_image_id": sorted(set(value["other_image_id"])),
                        }
                    )
                    if record.item_id != expected["item_id"]:
                        raise ABOProvenanceError(
                            "ABO selected listing identity differs"
                        )
                    found[record.item_id] = record
    if len(found) != len({row["item_id"] for row in rows}):
        raise ABOProvenanceError("ABO selected listings are incomplete")
    return found


def _build_packet_rows(
    compact_rows: list[dict[str, Any]],
    *,
    listing_records: dict[str, ABOListingRecord],
    downloaded: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    packet: list[dict[str, Any]] = []
    for compact in compact_rows:
        match = _SMALL_MEMBER.fullmatch(compact["archive_member"])
        assert match is not None
        metadata_path = match.group(1)
        receipt = downloaded[metadata_path]
        image_record = ABOImageManifestRecord.model_validate(
            {
                "image_id": compact["image_id"],
                "path": metadata_path,
                "official_image_uri": receipt["official_image_uri"],
                "official_image_etag": receipt["official_image_etag"],
                "source_image_sha256": receipt["source_image_sha256"],
            }
        )
        listing_record = listing_records[compact["item_id"]]
        packet.append(
            {
                "height": receipt["height"],
                "image_id": compact["image_id"],
                "image_record_sha256": sha256_bytes(
                    canonical_json_bytes(image_record.model_dump(mode="json"))
                ),
                "image_role": compact["image_role"],
                "item_id": compact["item_id"],
                "listing_record_sha256": sha256_bytes(
                    canonical_json_bytes(listing_record.model_dump(mode="json"))
                ),
                "local_path": f"images/original/{metadata_path}",
                "metadata_path": metadata_path,
                "official_image_etag": receipt["official_image_etag"],
                "official_image_uri": receipt["official_image_uri"],
                "packet_status": "human_decision_required",
                "parent_small_image_sha256": compact["source_image_sha256"],
                "schema_version": 1,
                "source_image_sha256": receipt["source_image_sha256"],
                "width": receipt["width"],
            }
        )
    packet.sort(key=lambda row: (row["item_id"], row["image_role"]))
    return packet


def _verify_raw_copy(
    raw_original_root: Path,
    downloaded: dict[str, dict[str, Any]],
) -> None:
    for relative, receipt in downloaded.items():
        digest, size = stable_file_digest(
            raw_original_root / Path(*relative.split("/")),
            label=f"ABO raw original {relative}",
        )
        if digest != receipt["source_image_sha256"] or size != receipt["bytes"]:
            raise ABOProvenanceError("ABO raw original copy differs from download")


def _compact_metadata_path(row: dict[str, Any]) -> str:
    member = row.get("archive_member")
    if not isinstance(member, str):
        raise ABOProvenanceError("ABO compact row lacks archive_member")
    match = _SMALL_MEMBER.fullmatch(member)
    if match is None:
        raise ABOProvenanceError("ABO compact row has a non-canonical image path")
    return match.group(1)


def _load_reusable_receipts(
    previous_packet_root: Path,
    previous_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    content = read_stable_regular_file(
        previous_packet_root / "download-receipt.jsonl",
        label="ABO previous download receipt",
        max_bytes=4 * 1024 * 1024,
    )
    parsed = parse_canonical_jsonl(content, label="ABO previous download receipt")
    rows_by_path: dict[str, list[dict[str, Any]]] = {}
    for row in previous_rows:
        if not isinstance(row, dict):
            raise ABOProvenanceError("ABO previous packet row is invalid")
        metadata_path = row.get("metadata_path")
        local_path = row.get("local_path")
        if (
            not isinstance(metadata_path, str)
            or _METADATA_PATH.fullmatch(metadata_path) is None
            or local_path != f"images/original/{metadata_path}"
        ):
            raise ABOProvenanceError(
                "ABO previous packet has a non-canonical original path"
            )
        rows_by_path.setdefault(metadata_path, []).append(row)

    receipts: dict[str, dict[str, Any]] = {}
    for receipt in parsed:
        if not isinstance(receipt, dict):
            raise ABOProvenanceError("ABO previous receipt row is invalid")
        relative = receipt.get("metadata_path")
        if (
            not isinstance(relative, str)
            or _METADATA_PATH.fullmatch(relative) is None
            or relative in receipts
            or not isinstance(receipt.get("bytes"), int)
            or receipt["bytes"] <= 0
            or not isinstance(receipt.get("width"), int)
            or receipt["width"] <= 0
            or not isinstance(receipt.get("height"), int)
            or receipt["height"] <= 0
            or not isinstance(receipt.get("image_id"), str)
            or not isinstance(receipt.get("official_image_etag"), str)
            or not receipt["official_image_etag"]
            or not isinstance(receipt.get("source_image_sha256"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                receipt["source_image_sha256"],
            )
            is None
            or receipt.get("official_image_uri") != f"{_OFFICIAL_ROOT}/{relative}"
        ):
            raise ABOProvenanceError("ABO previous receipt identity is invalid")
        matching_rows = rows_by_path.get(relative)
        if not matching_rows:
            raise ABOProvenanceError(
                "ABO previous receipt is not referenced by its packet"
            )
        for row in matching_rows:
            if (
                row.get("image_id") != receipt["image_id"]
                or row.get("official_image_uri") != receipt["official_image_uri"]
                or row.get("official_image_etag") != receipt["official_image_etag"]
                or row.get("source_image_sha256") != receipt["source_image_sha256"]
                or row.get("width") != receipt["width"]
                or row.get("height") != receipt["height"]
            ):
                raise ABOProvenanceError(
                    "ABO previous receipt differs from packet identity"
                )
        image_path = (
            previous_packet_root / "images" / "original" / Path(*relative.split("/"))
        )
        digest, size = stable_file_digest(
            image_path,
            label=f"ABO previous original {relative}",
        )
        if digest != receipt["source_image_sha256"] or size != receipt["bytes"]:
            raise ABOProvenanceError("ABO previous original bytes differ from receipt")
        receipts[relative] = receipt
    if set(receipts) != set(rows_by_path):
        raise ABOProvenanceError(
            "ABO previous receipt does not cover every packet image"
        )
    return receipts


def _verify_reused_row_identities(
    previous_rows: list[dict[str, Any]],
    expanded_rows: list[dict[str, Any]],
) -> None:
    expanded_by_identity = {
        (row["item_id"], row["image_id"]): row for row in expanded_rows
    }
    fields = (
        "image_role",
        "listing_record_sha256",
        "image_record_sha256",
        "source_image_sha256",
        "metadata_path",
    )
    for previous in previous_rows:
        expanded = expanded_by_identity.get((previous["item_id"], previous["image_id"]))
        if expanded is None:
            continue
        if any(previous.get(field) != expanded.get(field) for field in fields):
            raise ABOProvenanceError(
                "ABO expanded packet changed a reusable reviewed identity"
            )


def _require_real_directory(path: Path, label: str) -> None:
    try:
        snapshot = path.lstat()
    except OSError as error:
        raise ABOProvenanceError(f"{label} cannot be inspected") from error
    is_junction = getattr(path, "is_junction", None)
    if (
        stat.S_ISLNK(snapshot.st_mode)
        or (is_junction is not None and is_junction())
        or not stat.S_ISDIR(snapshot.st_mode)
    ):
        raise ABOProvenanceError(f"{label} must be a real non-symlink directory")


def _reject_links_in_tree(root: Path, label: str) -> None:
    _require_real_directory(root, f"{label} root")
    for path in root.rglob("*"):
        try:
            snapshot = path.lstat()
        except OSError as error:
            raise ABOProvenanceError(f"{label} path cannot be inspected") from error
        is_junction = getattr(path, "is_junction", None)
        if stat.S_ISLNK(snapshot.st_mode) or (
            is_junction is not None and is_junction()
        ):
            raise ABOProvenanceError(f"{label} must not contain links or junctions")
        if not stat.S_ISREG(snapshot.st_mode) and not stat.S_ISDIR(snapshot.st_mode):
            raise ABOProvenanceError(f"{label} contains a non-file entry")
