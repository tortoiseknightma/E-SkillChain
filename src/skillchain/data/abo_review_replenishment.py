"""Carry verified ABO human decisions into an expanded review packet.

Only byte- and record-identical target rows inherit a prior decision.  The
result is a review aid, not a formal-use approval, and every unmatched target
row remains for a human to review.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any

from skillchain.data.abo import ABOImageReviewDecision, ABOProvenanceError
from skillchain.data.abo_original_review import verify_abo_original_review_packet
from skillchain.data.abo_pair_audit import (
    load_verified_abo_human_review_decisions,
)
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

CARRY_FORWARD_BUNDLE_ID = "abo-review-decision-carry-forward-v1"
CARRY_FORWARD_STATUS = "carried_decisions_pending_new_human_review"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTITY_FIELDS = (
    "listing_record_sha256",
    "image_record_sha256",
    "source_image_sha256",
)


def prepare_abo_review_decision_carry_forward(
    *,
    target_packet_root: Path | str,
    expected_target_packet_manifest_sha256: str,
    source_packet_root: Path | str,
    expected_source_packet_manifest_sha256: str,
    source_human_review_root: Path | str,
    expected_source_human_review_manifest_sha256: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Publish immutable prior decisions that exactly match a larger packet."""

    for digest, label in (
        (
            expected_target_packet_manifest_sha256,
            "ABO target packet manifest",
        ),
        (
            expected_source_packet_manifest_sha256,
            "ABO source packet manifest",
        ),
        (
            expected_source_human_review_manifest_sha256,
            "ABO source human-review manifest",
        ),
    ):
        _validate_digest(digest, label)
    target_packet_root = Path(target_packet_root).absolute()
    source_packet_root = Path(source_packet_root).absolute()
    source_human_review_root = Path(source_human_review_root).absolute()
    output_dir = Path(output_dir).absolute()
    for root, label in (
        (target_packet_root, "ABO target packet"),
        (source_packet_root, "ABO source packet"),
        (source_human_review_root, "ABO source human review"),
    ):
        _reject_links_in_tree(root, label)
    _require_real_directory(output_dir.parent, "ABO carry-forward output parent")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"ABO carry-forward output already exists: {output_dir}")

    verify_abo_original_review_packet(
        target_packet_root,
        expected_manifest_sha256=expected_target_packet_manifest_sha256,
    )
    verify_abo_original_review_packet(
        source_packet_root,
        expected_manifest_sha256=expected_source_packet_manifest_sha256,
    )
    target_rows = _load_packet_rows(target_packet_root)
    source_decisions = load_verified_abo_human_review_decisions(
        packet_root=source_packet_root,
        expected_packet_manifest_sha256=expected_source_packet_manifest_sha256,
        human_review_root=source_human_review_root,
        expected_human_review_manifest_sha256=(
            expected_source_human_review_manifest_sha256
        ),
    )
    decisions_by_identity = {
        (decision.item_id, decision.image_id): decision for decision in source_decisions
    }
    carried: list[ABOImageReviewDecision] = []
    for row in target_rows:
        decision = decisions_by_identity.get((row["item_id"], row["image_id"]))
        if decision is None:
            continue
        if any(getattr(decision, field) != row[field] for field in _IDENTITY_FIELDS):
            raise ABOProvenanceError(
                "ABO target row reuses a reviewed identity with changed content"
            )
        carried.append(decision)
    carried.sort(key=lambda row: (row.item_id, row.image_id))
    if not carried:
        raise ABOProvenanceError(
            "ABO carry-forward found no byte-identical prior decisions"
        )
    if len(carried) >= len(target_rows):
        raise ABOProvenanceError(
            "ABO carry-forward target has no new image decisions to review"
        )

    decision_bytes = canonical_jsonl_bytes(
        decision.model_dump(mode="json") for decision in carried
    )
    unsigned_manifest = {
        "bundle_id": CARRY_FORWARD_BUNDLE_ID,
        "carried_decision_count": len(carried),
        "files": [
            {
                "bytes": len(decision_bytes),
                "path": "carried-decisions.jsonl",
                "sha256": sha256_bytes(decision_bytes),
            }
        ],
        "formal_use_allowed": False,
        "remaining_decision_count": len(target_rows) - len(carried),
        "schema_version": 1,
        "source_human_review_manifest_sha256": (
            expected_source_human_review_manifest_sha256
        ),
        "source_packet_manifest_sha256": (expected_source_packet_manifest_sha256),
        "status": CARRY_FORWARD_STATUS,
        "target_packet_manifest_sha256": (expected_target_packet_manifest_sha256),
        "target_total_decision_count": len(target_rows),
    }
    manifest = {
        **unsigned_manifest,
        "manifest_self_sha256": sha256_bytes(canonical_json_bytes(unsigned_manifest)),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    staging = new_staging_directory(output_dir)
    try:
        (staging / "carried-decisions.jsonl").write_bytes(decision_bytes)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        refreshed = load_verified_abo_human_review_decisions(
            packet_root=source_packet_root,
            expected_packet_manifest_sha256=(expected_source_packet_manifest_sha256),
            human_review_root=source_human_review_root,
            expected_human_review_manifest_sha256=(
                expected_source_human_review_manifest_sha256
            ),
        )
        if canonical_jsonl_bytes(
            decision.model_dump(mode="json") for decision in refreshed
        ) != canonical_jsonl_bytes(
            decision.model_dump(mode="json") for decision in source_decisions
        ):
            raise ABOProvenanceError(
                "ABO source human decisions changed during carry-forward"
            )
        verify_abo_original_review_packet(
            target_packet_root,
            expected_manifest_sha256=expected_target_packet_manifest_sha256,
        )
        atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "carried_decisions": len(carried),
        "manifest_path": str(output_dir / "manifest.json"),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "output_dir": str(output_dir),
        "remaining_decisions": len(target_rows) - len(carried),
        "status": CARRY_FORWARD_STATUS,
    }


def verify_abo_review_decision_carry_forward(
    root: Path | str,
    *,
    expected_manifest_sha256: str,
    target_packet_root: Path | str,
    expected_target_packet_manifest_sha256: str,
    source_packet_root: Path | str,
    expected_source_packet_manifest_sha256: str,
    source_human_review_root: Path | str,
    expected_source_human_review_manifest_sha256: str,
) -> tuple[dict[str, Any], tuple[ABOImageReviewDecision, ...]]:
    """Verify carry-forward bytes against target and authoritative source."""

    _validate_digest(expected_manifest_sha256, "ABO carry-forward manifest")
    for digest, label in (
        (
            expected_target_packet_manifest_sha256,
            "ABO target packet manifest",
        ),
        (
            expected_source_packet_manifest_sha256,
            "ABO source packet manifest",
        ),
        (
            expected_source_human_review_manifest_sha256,
            "ABO source human-review manifest",
        ),
    ):
        _validate_digest(digest, label)
    root = Path(root).absolute()
    target_packet_root = Path(target_packet_root).absolute()
    source_packet_root = Path(source_packet_root).absolute()
    source_human_review_root = Path(source_human_review_root).absolute()
    _reject_links_in_tree(root, "ABO carry-forward bundle")
    _reject_links_in_tree(target_packet_root, "ABO target packet")
    _reject_links_in_tree(source_packet_root, "ABO source packet")
    _reject_links_in_tree(source_human_review_root, "ABO source human review")
    verify_abo_original_review_packet(
        target_packet_root,
        expected_manifest_sha256=expected_target_packet_manifest_sha256,
    )
    manifest_bytes = read_stable_regular_file(
        root / "manifest.json",
        label="ABO carry-forward manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        raise ABOProvenanceError("ABO carry-forward manifest digest mismatch")
    manifest = parse_canonical_json(
        manifest_bytes,
        label="ABO carry-forward manifest",
    )
    if not isinstance(manifest, dict):
        raise ABOProvenanceError("ABO carry-forward manifest is invalid")
    unsigned = dict(manifest)
    manifest_self_sha256 = unsigned.pop("manifest_self_sha256", None)
    if (
        manifest_self_sha256 != sha256_bytes(canonical_json_bytes(unsigned))
        or unsigned.get("schema_version") != 1
        or unsigned.get("bundle_id") != CARRY_FORWARD_BUNDLE_ID
        or unsigned.get("formal_use_allowed") is not False
        or unsigned.get("status") != CARRY_FORWARD_STATUS
        or unsigned.get("target_packet_manifest_sha256")
        != expected_target_packet_manifest_sha256
        or unsigned.get("source_packet_manifest_sha256")
        != expected_source_packet_manifest_sha256
        or unsigned.get("source_human_review_manifest_sha256")
        != expected_source_human_review_manifest_sha256
    ):
        raise ABOProvenanceError("ABO carry-forward manifest identity is invalid")
    files = unsigned.get("files")
    if (
        not isinstance(files, list)
        or len(files) != 1
        or not isinstance(files[0], dict)
        or files[0].get("path") != "carried-decisions.jsonl"
    ):
        raise ABOProvenanceError("ABO carry-forward file descriptor is invalid")
    decision_bytes = read_stable_regular_file(
        root / "carried-decisions.jsonl",
        label="ABO carried decisions",
        max_bytes=4 * 1024 * 1024,
    )
    if files[0].get("bytes") != len(decision_bytes) or files[0].get(
        "sha256"
    ) != sha256_bytes(decision_bytes):
        raise ABOProvenanceError("ABO carried decisions payload drifted")
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != {"manifest.json", "carried-decisions.jsonl"}:
        raise ABOProvenanceError("ABO carry-forward file set drifted")
    try:
        decisions = tuple(
            ABOImageReviewDecision.model_validate(row, strict=True)
            for row in parse_canonical_jsonl(
                decision_bytes,
                label="ABO carried decisions",
            )
        )
    except Exception as error:
        raise ABOProvenanceError("ABO carried decisions are invalid") from error
    if len({(row.item_id, row.image_id) for row in decisions}) != len(decisions):
        raise ABOProvenanceError("ABO carried decision identity is duplicated")
    target_rows = _load_packet_rows(target_packet_root)
    target_by_identity = {(row["item_id"], row["image_id"]): row for row in target_rows}
    for decision in decisions:
        target = target_by_identity.get((decision.item_id, decision.image_id))
        if target is None or any(
            getattr(decision, field) != target[field] for field in _IDENTITY_FIELDS
        ):
            raise ABOProvenanceError("ABO carried decision differs from target packet")
    source_decisions = load_verified_abo_human_review_decisions(
        packet_root=source_packet_root,
        expected_packet_manifest_sha256=expected_source_packet_manifest_sha256,
        human_review_root=source_human_review_root,
        expected_human_review_manifest_sha256=(
            expected_source_human_review_manifest_sha256
        ),
    )
    expected_decisions: list[ABOImageReviewDecision] = []
    for decision in source_decisions:
        target = target_by_identity.get((decision.item_id, decision.image_id))
        if target is None:
            continue
        if any(getattr(decision, field) != target[field] for field in _IDENTITY_FIELDS):
            raise ABOProvenanceError(
                "ABO target changed an authoritative reviewed identity"
            )
        expected_decisions.append(decision)
    expected_decisions.sort(key=lambda row: (row.item_id, row.image_id))
    if canonical_jsonl_bytes(
        decision.model_dump(mode="json") for decision in decisions
    ) != canonical_jsonl_bytes(
        decision.model_dump(mode="json") for decision in expected_decisions
    ):
        raise ABOProvenanceError(
            "ABO carried decisions are not the exact authoritative source subset"
        )
    if (
        unsigned.get("carried_decision_count") != len(decisions)
        or unsigned.get("target_total_decision_count") != len(target_rows)
        or unsigned.get("remaining_decision_count") != len(target_rows) - len(decisions)
        or len(decisions) >= len(target_rows)
    ):
        raise ABOProvenanceError("ABO carry-forward decision counts are invalid")
    return manifest, decisions


def _load_packet_rows(root: Path) -> list[dict[str, Any]]:
    rows = parse_canonical_jsonl(
        read_stable_regular_file(
            root / "review-packet.jsonl",
            label="ABO target review packet",
            max_bytes=4 * 1024 * 1024,
        ),
        label="ABO target review packet",
    )
    identities: set[tuple[str, str]] = set()
    roles: dict[str, set[str]] = {}
    validated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ABOProvenanceError("ABO target packet row is invalid")
        identity = (row.get("item_id"), row.get("image_id"))
        if (
            not all(isinstance(value, str) and value for value in identity)
            or identity in identities
            or row.get("image_role") not in {"main", "other"}
        ):
            raise ABOProvenanceError("ABO target packet identity is invalid")
        for field in _IDENTITY_FIELDS:
            _validate_digest(row.get(field), f"ABO target {field}")
        local_path = row.get("local_path")
        if not isinstance(local_path, str):
            raise ABOProvenanceError("ABO target local path is invalid")
        _canonical_relative_path(local_path)
        identities.add(identity)
        roles.setdefault(row["item_id"], set()).add(row["image_role"])
        validated.append(row)
    if (
        not validated
        or len(validated) != 2 * len(roles)
        or any(value != {"main", "other"} for value in roles.values())
    ):
        raise ABOProvenanceError("ABO target packet pair coverage is invalid")
    return sorted(validated, key=lambda row: (row["item_id"], row["image_id"]))


def _canonical_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ABOProvenanceError("ABO relative path is invalid")
    return value


def _validate_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


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


__all__ = [
    "CARRY_FORWARD_BUNDLE_ID",
    "CARRY_FORWARD_STATUS",
    "prepare_abo_review_decision_carry_forward",
    "verify_abo_review_decision_carry_forward",
]
