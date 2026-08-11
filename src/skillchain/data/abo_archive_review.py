"""Prepare a human-only ABO Exact Match review packet from approved archives.

This module deliberately stops before formal normalization or approval.  It
consumes only ``abo-listings.tar`` and ``abo-images-small.tar`` from the
owner-approved compact source lock, deterministically chooses candidate pairs
without looking at image pixels, and publishes original archive member bytes
for accountable human review.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
from typing import Any

from PIL import Image

from skillchain.data.abo import (
    ABO_FORMAL_PERMISSIONS,
    ABO_FORMAL_PURPOSES,
    ABOProvenanceError,
    VerifiedABOSourceApproval,
    load_verified_abo_source_approval,
)
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

SELECTION_POLICY_VERSION = "abo-compact-archive-review-hash-sample-v1"
PACKET_ID = "abo-compact-archive-review-v1"
_LISTINGS_PATH = "abo/archives/abo-listings.tar"
_IMAGES_PATH = "abo/archives/abo-images-small.tar"
_LISTING_MEMBER = re.compile(r"listings/metadata/listings_[0-9a-f]\.json\.gz")
_IMAGE_MEMBER = re.compile(r"images/small/[0-9a-f]{2}/[0-9a-f]{8}\.(?:jpg|png)")
_IMAGE_METADATA_MEMBER = "images/metadata/images.csv.gz"
_IMAGE_METADATA_PATH = re.compile(r"[0-9a-f]{2}/[0-9a-f]{8}\.(?:jpg|png)")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_MAX_IMAGE_BYTES = 50 * 1024 * 1024


def prepare_abo_archive_review_packet(
    *,
    required_source_lock_path: Path | str,
    expected_required_source_lock_sha256: str,
    raw_root: Path | str,
    license_evidence_path: Path | str,
    source_review_policy_path: Path | str,
    expected_source_review_policy_sha256: str,
    expected_source_review_portfolio_sha256: str,
    source_review_ledger_path: Path | str,
    expected_source_review_ledger_sha256: str,
    output_dir: Path | str,
    maximum_pair_candidates: int,
) -> dict[str, Any]:
    """Publish deterministic original-byte candidates for later human review."""

    if maximum_pair_candidates <= 0:
        raise ValueError("maximum_pair_candidates must be positive")
    output_dir = Path(output_dir).absolute()
    _require_real_directory(
        output_dir.parent,
        "ABO archive review output parent",
    )
    if os.path.lexists(output_dir):
        raise FileExistsError(f"ABO archive review output already exists: {output_dir}")
    approval = load_verified_abo_source_approval(
        required_source_lock_path=required_source_lock_path,
        expected_required_source_lock_sha256=(expected_required_source_lock_sha256),
        required_source_raw_root=raw_root,
        license_evidence_path=license_evidence_path,
        source_review_policy_path=source_review_policy_path,
        expected_source_review_policy_sha256=(expected_source_review_policy_sha256),
        expected_source_review_portfolio_sha256=(
            expected_source_review_portfolio_sha256
        ),
        source_review_ledger_path=source_review_ledger_path,
        expected_source_review_ledger_sha256=(expected_source_review_ledger_sha256),
        purposes=ABO_FORMAL_PURPOSES,
        permissions=ABO_FORMAL_PERMISSIONS,
    )
    return _prepare_verified_packet(
        approval,
        output_dir=output_dir,
        maximum_pair_candidates=maximum_pair_candidates,
    )


def verify_abo_archive_review_packet(
    root: Path | str,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Reload a published review packet under an external manifest digest."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None:
        raise ValueError("ABO review manifest digest must be lowercase SHA-256")
    root = Path(root).absolute()
    _reject_links_in_tree(root, "ABO archive review packet")
    manifest_path = root / "manifest.json"
    manifest_bytes = read_stable_regular_file(
        manifest_path,
        label="ABO archive review manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        raise ABOProvenanceError("ABO review manifest external digest mismatch")
    manifest = parse_canonical_json(
        manifest_bytes,
        label="ABO archive review manifest",
    )
    if isinstance(manifest, dict) and "manifest_self_sha256" in manifest:
        manifest_without_self_hash = dict(manifest)
        manifest_self_sha256 = manifest_without_self_hash.pop("manifest_self_sha256")
        if not isinstance(
            manifest_self_sha256, str
        ) or manifest_self_sha256 != sha256_bytes(
            canonical_json_bytes(manifest_without_self_hash)
        ):
            raise ABOProvenanceError("ABO archive review manifest self digest mismatch")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("packet_id") != PACKET_ID
        or manifest.get("status") != "pending_human_catalog_photo_review"
        or manifest.get("human_decision_required") is not True
    ):
        raise ABOProvenanceError("ABO archive review manifest is invalid")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ABOProvenanceError("ABO archive review file manifest is invalid")
    expected_paths = {"manifest.json"}
    for row in files:
        if not isinstance(row, dict):
            raise ABOProvenanceError("ABO archive review file row is invalid")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            raise ABOProvenanceError("ABO archive review file path is invalid")
        path = root / Path(*relative.split("/"))
        digest, size = stable_file_digest(path, label=f"ABO review {relative}")
        if digest != row.get("sha256") or size != row.get("bytes"):
            raise ABOProvenanceError(f"ABO archive review payload drifted: {relative}")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ABOProvenanceError("ABO archive review file set drifted")
    packet_path = root / "review-packet.jsonl"
    packet = parse_canonical_jsonl(
        packet_path.read_bytes(),
        label="ABO archive review packet",
    )
    item_roles: dict[str, set[str]] = {}
    for row in packet:
        if (
            not isinstance(row, dict)
            or row.get("packet_status") != "human_decision_required"
        ):
            raise ABOProvenanceError("ABO archive review packet row is invalid")
        item_roles.setdefault(row["item_id"], set()).add(row["image_role"])
    if (
        len(packet) != manifest.get("candidate_image_count")
        or len(item_roles) != manifest.get("candidate_pair_count")
        or any(roles != {"main", "other"} for roles in item_roles.values())
    ):
        raise ABOProvenanceError("ABO archive review pair coverage differs")
    return manifest


def _prepare_verified_packet(
    approval: VerifiedABOSourceApproval,
    *,
    output_dir: Path,
    maximum_pair_candidates: int,
) -> dict[str, Any]:
    if not isinstance(approval, VerifiedABOSourceApproval):
        raise TypeError("ABO archive review requires verified owner approval")
    identities = {
        value.logical_path: value
        for value in approval.required_source_lock.lock.acquisition_identities
    }
    if _LISTINGS_PATH not in identities or _IMAGES_PATH not in identities:
        raise ABOProvenanceError(
            "ABO compact source lock lacks listings or small-image archive identity"
        )
    listings_path = approval.raw_root / Path(*_LISTINGS_PATH.split("/"))
    images_path = approval.raw_root / Path(*_IMAGES_PATH.split("/"))
    _verify_archive_identity(listings_path, identities[_LISTINGS_PATH])
    _verify_archive_identity(images_path, identities[_IMAGES_PATH])

    image_members = _index_image_members(images_path)
    candidates, scanned = _select_candidates(
        listings_path,
        image_members,
        maximum_pair_candidates=maximum_pair_candidates,
    )
    if len(candidates) != maximum_pair_candidates:
        raise ABOProvenanceError(
            "ABO compact archives cannot fill the preregistered review packet: "
            f"selected={len(candidates)} requested={maximum_pair_candidates}"
        )

    staging = new_staging_directory(output_dir)
    try:
        packet_rows, image_files = _materialize_candidates(
            images_path,
            candidates,
            staging,
        )
        packet_bytes = canonical_jsonl_bytes(packet_rows)
        (staging / "review-packet.jsonl").write_bytes(packet_bytes)
        file_rows = [
            {
                "bytes": len(packet_bytes),
                "path": "review-packet.jsonl",
                "sha256": sha256_bytes(packet_bytes),
            },
            *image_files,
        ]
        file_rows.sort(key=lambda value: value["path"])
        review_record_sha256 = sha256_bytes(
            canonical_json_bytes(approval.review_record.model_dump(mode="json"))
        )
        manifest = {
            "candidate_image_count": len(packet_rows),
            "candidate_pair_count": len(candidates),
            "files": file_rows,
            "human_decision_required": True,
            "image_archive_sha256": identities[_IMAGES_PATH].local_sha256,
            "listing_archive_sha256": identities[_LISTINGS_PATH].local_sha256,
            "listing_records_scanned": scanned,
            "maximum_pair_candidates": maximum_pair_candidates,
            "packet_id": PACKET_ID,
            "raw_mutation_performed": False,
            "schema_version": 1,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "source_lock_sha256": (approval.required_source_lock.lock_file_sha256),
            "source_review_ledger_sha256": approval.ledger_file_sha256,
            "source_review_record_sha256": review_record_sha256,
            "source_revision": approval.required_source_lock.lock.source_revision,
            "status": "pending_human_catalog_photo_review",
        }
        manifest["manifest_self_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        manifest_bytes = canonical_json_bytes(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        _verify_archive_identity(listings_path, identities[_LISTINGS_PATH])
        _verify_archive_identity(images_path, identities[_IMAGES_PATH])
        for snapshot, label in (
            (approval.required_lock_snapshot, "required source lock"),
            (approval.policy_snapshot, "source-review policy"),
            (approval.ledger_snapshot, "source-review ledger"),
            (approval.license_evidence_snapshot, "license evidence"),
        ):
            digest, size = stable_file_digest(snapshot.path, label=label)
            if digest != snapshot.sha256 or size != len(snapshot.content):
                raise ABOProvenanceError(f"ABO {label} changed during review build")
        atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "candidate_images": len(packet_rows),
        "candidate_pairs": len(candidates),
        "manifest_path": str(output_dir / "manifest.json"),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "output_dir": str(output_dir),
        "status": "pending_human_catalog_photo_review",
    }


def _verify_archive_identity(path: Path, identity: Any) -> None:
    digest, size = stable_file_digest(path, label=f"ABO archive {path.name}")
    if digest != identity.local_sha256 or size != identity.bytes:
        raise ABOProvenanceError(f"ABO approved archive identity drifted: {path.name}")


def _index_image_members(path: Path) -> dict[str, str]:
    member_names: set[str] = set()
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if _IMAGE_MEMBER.fullmatch(member.name) is None:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ABOProvenanceError("ABO image archive member must be regular")
            if member.name in member_names:
                raise ABOProvenanceError(
                    f"duplicate ABO image archive member: {member.name}"
                )
            member_names.add(member.name)
        metadata = archive.getmember(_IMAGE_METADATA_MEMBER)
        if not metadata.isfile() or metadata.issym() or metadata.islnk():
            raise ABOProvenanceError("ABO image metadata member must be regular")
        extracted = archive.extractfile(metadata)
        if extracted is None:
            raise ABOProvenanceError("ABO image metadata cannot be read")
        members: dict[str, str] = {}
        mapped_paths: set[str] = set()
        with gzip.GzipFile(fileobj=extracted) as decompressed:
            reader = csv.DictReader(io.TextIOWrapper(decompressed, encoding="utf-8"))
            if reader.fieldnames != ["image_id", "height", "width", "path"]:
                raise ABOProvenanceError("ABO image metadata header is invalid")
            for row in reader:
                image_id = row["image_id"]
                relative = row["path"]
                if (
                    _SAFE_ID.fullmatch(image_id) is None
                    or _IMAGE_METADATA_PATH.fullmatch(relative) is None
                ):
                    raise ABOProvenanceError("ABO image metadata row is invalid")
                member_name = f"images/small/{relative}"
                if member_name not in member_names:
                    raise ABOProvenanceError(
                        "ABO image metadata references a missing small image"
                    )
                if image_id in members or member_name in mapped_paths:
                    raise ABOProvenanceError("ABO image metadata is not one-to-one")
                members[image_id] = member_name
                mapped_paths.add(member_name)
    if not members or mapped_paths != member_names:
        raise ABOProvenanceError("ABO small-image archive contains no images")
    return members


def _select_candidates(
    listings_path: Path,
    image_members: dict[str, str],
    *,
    maximum_pair_candidates: int,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[tuple[int, str, str, int, dict[str, Any]]] = []
    seen_item_ids: set[str] = set()
    scanned = 0
    with tarfile.open(listings_path, mode="r:") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if _LISTING_MEMBER.fullmatch(member.name)
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise ABOProvenanceError("ABO listings archive has no canonical shards")
        for member in members:
            if not member.isfile() or member.issym() or member.islnk():
                raise ABOProvenanceError("ABO listing shard must be regular")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ABOProvenanceError("ABO listing shard cannot be read")
            with gzip.GzipFile(fileobj=extracted) as decompressed:
                for line_number, raw_line in enumerate(decompressed, start=1):
                    scanned += 1
                    candidate = _candidate_from_listing(
                        raw_line,
                        member.name,
                        line_number,
                        image_members,
                    )
                    if candidate is None:
                        continue
                    if candidate["item_id"] in seen_item_ids:
                        continue
                    seen_item_ids.add(candidate["item_id"])
                    score = int(
                        hashlib.sha256(
                            f"{SELECTION_POLICY_VERSION}:{candidate['item_id']}".encode(
                                "utf-8"
                            )
                        ).hexdigest(),
                        16,
                    )
                    entry = (
                        -score,
                        candidate["item_id"],
                        candidate["listing_member"],
                        candidate["listing_line_number"],
                        candidate,
                    )
                    if len(selected) < maximum_pair_candidates:
                        heapq.heappush(selected, entry)
                    elif score < -selected[0][0]:
                        heapq.heapreplace(selected, entry)
    candidates = [value[4] for value in selected]
    candidates.sort(key=lambda value: value["item_id"])
    return candidates, scanned


def _candidate_from_listing(
    raw_line: bytes,
    member_name: str,
    line_number: int,
    image_members: dict[str, str],
) -> dict[str, Any] | None:
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ABOProvenanceError("ABO listing record is invalid JSON") from error
    item_id = value.get("item_id")
    main_id = value.get("main_image_id")
    other_ids = value.get("other_image_id")
    if (
        not isinstance(item_id, str)
        or _SAFE_ID.fullmatch(item_id) is None
        or not isinstance(main_id, str)
        or _SAFE_ID.fullmatch(main_id) is None
        or not isinstance(other_ids, list)
        or main_id not in image_members
    ):
        return None
    available = sorted(
        {
            image_id
            for image_id in other_ids
            if isinstance(image_id, str)
            and _SAFE_ID.fullmatch(image_id) is not None
            and image_id != main_id
            and image_id in image_members
        }
    )
    if not available:
        return None
    other_id = min(
        available,
        key=lambda image_id: hashlib.sha256(
            f"{SELECTION_POLICY_VERSION}:{item_id}:{image_id}".encode("utf-8")
        ).hexdigest(),
    )
    normalized = {
        "item_id": item_id,
        "main_image_id": main_id,
        "other_image_id": available,
    }
    return {
        "item_id": item_id,
        "listing_member": member_name,
        "listing_line_number": line_number,
        "listing_raw_sha256": sha256_bytes(raw_line),
        "listing_record_sha256": sha256_bytes(canonical_json_bytes(normalized)),
        "main_image_id": main_id,
        "other_image_id": other_id,
        "main_member": image_members[main_id],
        "other_member": image_members[other_id],
    }


def _materialize_candidates(
    images_path: Path,
    candidates: list[dict[str, Any]],
    staging: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packet_rows: list[dict[str, Any]] = []
    image_files: list[dict[str, Any]] = []
    images_root = staging / "images"
    images_root.mkdir()
    with tarfile.open(images_path, mode="r:") as archive:
        for candidate in candidates:
            item_root = images_root / candidate["item_id"]
            item_root.mkdir()
            for role in ("main", "other"):
                image_id = candidate[f"{role}_image_id"]
                member_name = candidate[f"{role}_member"]
                member = archive.getmember(member_name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ABOProvenanceError("ABO selected image cannot be read")
                content = extracted.read(_MAX_IMAGE_BYTES + 1)
                if not content or len(content) > _MAX_IMAGE_BYTES:
                    raise ABOProvenanceError("ABO selected image size is invalid")
                try:
                    with Image.open(io.BytesIO(content)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(content)) as image:
                        width, height = image.size
                except Exception as error:
                    raise ABOProvenanceError(
                        "ABO selected image is not decodable"
                    ) from error
                suffix = Path(member_name).suffix
                relative = f"images/{candidate['item_id']}/{role}-{image_id}{suffix}"
                (staging / Path(*relative.split("/"))).write_bytes(content)
                image_sha256 = sha256_bytes(content)
                image_record = {
                    "archive_member": member_name,
                    "image_id": image_id,
                    "source_image_sha256": image_sha256,
                }
                row = {
                    "archive_member": member_name,
                    "height": height,
                    "image_id": image_id,
                    "image_record_sha256": sha256_bytes(
                        canonical_json_bytes(image_record)
                    ),
                    "image_role": role,
                    "item_id": candidate["item_id"],
                    "listing_line_number": candidate["listing_line_number"],
                    "listing_member": candidate["listing_member"],
                    "listing_raw_sha256": candidate["listing_raw_sha256"],
                    "listing_record_sha256": candidate["listing_record_sha256"],
                    "local_path": relative,
                    "packet_status": "human_decision_required",
                    "schema_version": 1,
                    "source_image_sha256": image_sha256,
                    "width": width,
                }
                packet_rows.append(row)
                image_files.append(
                    {
                        "bytes": len(content),
                        "path": relative,
                        "sha256": image_sha256,
                    }
                )
    packet_rows.sort(key=lambda value: (value["item_id"], value["image_role"]))
    image_files.sort(key=lambda value: value["path"])
    return packet_rows, image_files


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
