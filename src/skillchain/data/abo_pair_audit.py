"""Create-only local leakage audit for human-approved ABO original pairs.

Human catalog-photo approval answers whether each image is usable in isolation.
It does not prove that two approved views are distinct, or that the same/near-
duplicate image is not reused by another candidate pair.  This module closes
that narrower boundary without promoting the result to formal dataset status.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

import imagehash
from PIL import Image, ImageOps

from skillchain.data.abo import ABOImageReviewDecision, ABOProvenanceError
from skillchain.data.abo_original_review import (
    DECISION_BUNDLE_ID,
    verify_abo_original_review_packet,
)
from skillchain.data.asset_catalog import (
    DEFAULT_NEAR_DUPLICATE_POLICY,
    NearDuplicatePolicy,
)
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)

AUDIT_ID = "abo-original-pair-leakage-audit-v1"
AUDIT_ALGORITHM = "exact-conflict-graph-mis-components-lexicographic-v1"
_APPROVED = "approve_catalog_product_photo"
_HUMAN_STATUS = "human_review_complete_pending_owner_scope_and_near_duplicate_audit"
_MAX_IMAGE_BYTES = 50 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROLE_ORDER = {"main": 0, "other": 1}


@dataclass(frozen=True)
class _HumanBundle:
    decisions: tuple[ABOImageReviewDecision, ...]
    manifest_bytes: bytes
    ledger_bytes: bytes
    summary_bytes: bytes


def load_verified_abo_human_review_decisions(
    *,
    packet_root: Path | str,
    expected_packet_manifest_sha256: str,
    human_review_root: Path | str,
    expected_human_review_manifest_sha256: str,
) -> tuple[ABOImageReviewDecision, ...]:
    """Load a complete human ledger under two externally supplied digests."""

    _validate_expected_digest(
        expected_packet_manifest_sha256,
        "ABO original packet manifest",
    )
    _validate_expected_digest(
        expected_human_review_manifest_sha256,
        "ABO human review manifest",
    )
    packet_root = Path(packet_root).absolute()
    human_review_root = Path(human_review_root).absolute()
    _reject_links_in_tree(packet_root, "ABO original packet")
    _reject_links_in_tree(human_review_root, "ABO human review")
    verify_abo_original_review_packet(
        packet_root,
        expected_manifest_sha256=expected_packet_manifest_sha256,
    )
    packet_rows = _load_packet_rows(packet_root)
    return _load_human_bundle(
        human_review_root,
        expected_manifest_sha256=expected_human_review_manifest_sha256,
        expected_packet_manifest_sha256=expected_packet_manifest_sha256,
        packet_rows=packet_rows,
    ).decisions


def audit_abo_original_pairs(
    *,
    packet_root: Path | str,
    expected_packet_manifest_sha256: str,
    human_review_root: Path | str,
    expected_human_review_manifest_sha256: str,
    output_path: Path | str,
    near_duplicate_policy: NearDuplicatePolicy = DEFAULT_NEAR_DUPLICATE_POLICY,
) -> dict[str, Any]:
    """Publish a canonical receipt for local pair conflicts and exact MIS choice.

    The output is create-only.  Both input manifests must be pinned by digests
    supplied outside those manifests, and all files used by the audit are
    reloaded before publication to reject observed input drift.
    """

    if not isinstance(near_duplicate_policy, NearDuplicatePolicy):
        raise TypeError("near_duplicate_policy must be a NearDuplicatePolicy")
    _validate_expected_digest(
        expected_packet_manifest_sha256,
        "ABO original packet manifest",
    )
    _validate_expected_digest(
        expected_human_review_manifest_sha256,
        "ABO human review manifest",
    )
    packet_root = Path(packet_root).absolute()
    human_review_root = Path(human_review_root).absolute()
    output_path = Path(output_path).absolute()
    _require_real_directory(packet_root, "ABO original packet root")
    _require_real_directory(human_review_root, "ABO human review root")
    _require_real_directory(output_path.parent, "ABO pair audit output parent")
    if os.path.lexists(output_path):
        raise FileExistsError(f"ABO pair audit output already exists: {output_path}")

    unsigned = _compute_unsigned_receipt(
        packet_root=packet_root,
        expected_packet_manifest_sha256=expected_packet_manifest_sha256,
        human_review_root=human_review_root,
        expected_human_review_manifest_sha256=(expected_human_review_manifest_sha256),
        near_duplicate_policy=near_duplicate_policy,
    )
    refreshed = _compute_unsigned_receipt(
        packet_root=packet_root,
        expected_packet_manifest_sha256=expected_packet_manifest_sha256,
        human_review_root=human_review_root,
        expected_human_review_manifest_sha256=(expected_human_review_manifest_sha256),
        near_duplicate_policy=near_duplicate_policy,
    )
    if refreshed != unsigned:
        raise ABOProvenanceError("ABO pair audit inputs changed during computation")

    receipt = {
        **unsigned,
        "receipt_self_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    receipt_bytes = canonical_json_bytes(receipt)
    _require_real_directory(output_path.parent, "ABO pair audit output parent")
    atomic_create_file(output_path, receipt_bytes)
    return {
        "approved_pair_count": receipt["approved_pair_count"],
        "cross_pair_conflict_count": receipt["cross_pair_conflict_count"],
        "excluded_pair_count": len(receipt["excluded_pair_ids"]),
        "output_path": str(output_path),
        "receipt_sha256": sha256_bytes(receipt_bytes),
        "retained_pair_count": receipt["retained_pair_count"],
        "within_pair_conflict_count": receipt["within_pair_conflict_count"],
    }


def _compute_unsigned_receipt(
    *,
    packet_root: Path,
    expected_packet_manifest_sha256: str,
    human_review_root: Path,
    expected_human_review_manifest_sha256: str,
    near_duplicate_policy: NearDuplicatePolicy,
) -> dict[str, Any]:
    _reject_links_in_tree(packet_root, "ABO original packet")
    packet_manifest = verify_abo_original_review_packet(
        packet_root,
        expected_manifest_sha256=expected_packet_manifest_sha256,
    )
    packet_rows = _load_packet_rows(packet_root)
    human = _load_human_bundle(
        human_review_root,
        expected_manifest_sha256=expected_human_review_manifest_sha256,
        expected_packet_manifest_sha256=expected_packet_manifest_sha256,
        packet_rows=packet_rows,
    )
    approved = _approved_pairs(packet_rows, human.decisions)
    pair_images, fingerprints = _fingerprint_approved_pairs(
        packet_root,
        approved,
    )
    within_conflicts, cross_conflicts, conflict_edges = _pair_conflicts(
        pair_images,
        max_hamming_distance=near_duplicate_policy.max_hamming_distance,
    )
    invalid_pair_ids = {row["pair_id"] for row in within_conflicts}
    approved_pair_ids = tuple(sorted(pair_images))
    retained_pair_ids = _maximum_independent_pair_ids(
        approved_pair_ids,
        conflict_edges,
        forbidden_pair_ids=invalid_pair_ids,
    )
    retained = set(retained_pair_ids)
    excluded_pair_ids = tuple(
        pair_id for pair_id in approved_pair_ids if pair_id not in retained
    )
    approved_pairs = [
        {
            "main_image_id": roles["main"]["image_id"],
            "main_local_path": roles["main"]["local_path"],
            "other_image_id": roles["other"]["image_id"],
            "other_local_path": roles["other"]["local_path"],
            "pair_id": pair_id,
        }
        for pair_id, roles in sorted(pair_images.items())
    ]
    return {
        "approved_image_reference_count": 2 * len(approved_pair_ids),
        "approved_pair_count": len(approved_pair_ids),
        "approved_pairs": approved_pairs,
        "audit_algorithm": AUDIT_ALGORITHM,
        "audit_id": AUDIT_ID,
        "cross_pair_conflict_count": len(cross_conflicts),
        "cross_pair_conflicts": cross_conflicts,
        "excluded_pair_ids": list(excluded_pair_ids),
        "expected_human_review_manifest_sha256": (
            expected_human_review_manifest_sha256
        ),
        "expected_packet_manifest_sha256": expected_packet_manifest_sha256,
        "fingerprint_policy": {
            "content_digest_algorithm": "sha256",
            "near_duplicate_policy": near_duplicate_policy.model_dump(mode="json"),
        },
        "formal_use_allowed": False,
        "human_review_ledger_sha256": sha256_bytes(human.ledger_bytes),
        "image_fingerprints": fingerprints,
        "packet_review_rows_sha256": packet_manifest["files"][
            _packet_file_index(packet_manifest)
        ]["sha256"],
        "retained_pair_count": len(retained_pair_ids),
        "retained_pair_ids": list(retained_pair_ids),
        "schema_version": 1,
        "status": "local_pair_leakage_audit_complete",
        "unique_image_count": len(fingerprints),
        "within_pair_conflict_count": len(within_conflicts),
        "within_pair_conflicts": within_conflicts,
    }


def _packet_file_index(manifest: dict[str, Any]) -> int:
    for index, row in enumerate(manifest["files"]):
        if row.get("path") == "review-packet.jsonl":
            return index
    raise ABOProvenanceError("ABO original manifest lacks review-packet.jsonl")


def _load_packet_rows(packet_root: Path) -> tuple[dict[str, Any], ...]:
    content = _read_real_file_below(
        packet_root,
        "review-packet.jsonl",
        label="ABO original review packet",
        max_bytes=4 * 1024 * 1024,
    )
    parsed = parse_canonical_jsonl(content, label="ABO original review packet")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    roles: dict[str, set[str]] = defaultdict(set)
    for raw in parsed:
        if not isinstance(raw, dict):
            raise ABOProvenanceError("ABO original packet row must be an object")
        for field in (
            "image_id",
            "image_record_sha256",
            "image_role",
            "item_id",
            "listing_record_sha256",
            "local_path",
            "source_image_sha256",
        ):
            if not isinstance(raw.get(field), str) or not raw[field]:
                raise ABOProvenanceError(f"ABO original packet row has invalid {field}")
        if raw["image_role"] not in _ROLE_ORDER:
            raise ABOProvenanceError("ABO original packet image role is invalid")
        for field in (
            "image_record_sha256",
            "listing_record_sha256",
            "source_image_sha256",
        ):
            _validate_expected_digest(raw[field], f"ABO packet {field}")
        _canonical_relative_path(raw["local_path"])
        identity = (raw["item_id"], raw["image_id"])
        if identity in seen:
            raise ABOProvenanceError("ABO original packet identity is duplicated")
        seen.add(identity)
        roles[raw["item_id"]].add(raw["image_role"])
        rows.append(raw)
    if not rows or any(value != {"main", "other"} for value in roles.values()):
        raise ABOProvenanceError("ABO original packet pair coverage is invalid")
    if len(rows) != 2 * len(roles):
        raise ABOProvenanceError("ABO original packet must contain two rows per pair")
    return tuple(sorted(rows, key=lambda row: (row["item_id"], row["image_id"])))


def _load_human_bundle(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_packet_manifest_sha256: str,
    packet_rows: tuple[dict[str, Any], ...],
) -> _HumanBundle:
    _reject_links_in_tree(root, "ABO human review")
    manifest_bytes = _read_real_file_below(
        root,
        "manifest.json",
        label="ABO human review manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        raise ABOProvenanceError("ABO human review manifest digest mismatch")
    manifest = parse_canonical_json(
        manifest_bytes,
        label="ABO human review manifest",
    )
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "bundle_id",
            "files",
            "formal_use_allowed",
            "packet_manifest_sha256",
            "schema_version",
            "status",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("bundle_id") != DECISION_BUNDLE_ID
        or manifest.get("formal_use_allowed") is not False
        or manifest.get("packet_manifest_sha256") != expected_packet_manifest_sha256
        or manifest.get("status") != _HUMAN_STATUS
    ):
        raise ABOProvenanceError("ABO human review manifest is invalid")
    descriptors = manifest.get("files")
    if not isinstance(descriptors, list) or len(descriptors) != 2:
        raise ABOProvenanceError("ABO human review file descriptors are invalid")
    payloads: dict[str, bytes] = {}
    for descriptor in descriptors:
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"bytes", "path", "sha256"}
            or descriptor.get("path") not in {"review-ledger.jsonl", "summary.json"}
            or not isinstance(descriptor.get("bytes"), int)
            or descriptor["bytes"] < 0
        ):
            raise ABOProvenanceError("ABO human review file descriptor is invalid")
        _validate_expected_digest(
            descriptor.get("sha256"),
            "ABO human review payload",
        )
        path = descriptor["path"]
        if path in payloads:
            raise ABOProvenanceError("ABO human review file descriptor is duplicated")
        content = _read_real_file_below(
            root,
            path,
            label=f"ABO human review {path}",
            max_bytes=4 * 1024 * 1024,
        )
        if (
            len(content) != descriptor["bytes"]
            or sha256_bytes(content) != descriptor["sha256"]
        ):
            raise ABOProvenanceError(f"ABO human review payload drifted: {path}")
        payloads[path] = content
    if set(payloads) != {"review-ledger.jsonl", "summary.json"}:
        raise ABOProvenanceError("ABO human review payload coverage is invalid")
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != {"manifest.json", *payloads}:
        raise ABOProvenanceError("ABO human review file set drifted")

    raw_decisions = parse_canonical_jsonl(
        payloads["review-ledger.jsonl"],
        label="ABO human review ledger",
    )
    try:
        decisions = tuple(
            ABOImageReviewDecision.model_validate(raw, strict=True)
            for raw in raw_decisions
        )
    except Exception as error:
        raise ABOProvenanceError("ABO human review ledger is invalid") from error
    _validate_decisions_and_summary(
        decisions,
        packet_rows=packet_rows,
        summary_bytes=payloads["summary.json"],
        expected_packet_manifest_sha256=expected_packet_manifest_sha256,
    )
    return _HumanBundle(
        decisions=decisions,
        manifest_bytes=manifest_bytes,
        ledger_bytes=payloads["review-ledger.jsonl"],
        summary_bytes=payloads["summary.json"],
    )


def _validate_decisions_and_summary(
    decisions: tuple[ABOImageReviewDecision, ...],
    *,
    packet_rows: tuple[dict[str, Any], ...],
    summary_bytes: bytes,
    expected_packet_manifest_sha256: str,
) -> None:
    packet_by_key = {(row["item_id"], row["image_id"]): row for row in packet_rows}
    decision_by_key = {
        (decision.item_id, decision.image_id): decision for decision in decisions
    }
    if len(decision_by_key) != len(decisions):
        raise ABOProvenanceError("ABO human review identities are duplicated")
    if set(decision_by_key) != set(packet_by_key):
        raise ABOProvenanceError("ABO human review does not exactly cover the packet")
    for key, decision in decision_by_key.items():
        packet = packet_by_key[key]
        if (
            decision.listing_record_sha256 != packet["listing_record_sha256"]
            or decision.image_record_sha256 != packet["image_record_sha256"]
            or decision.source_image_sha256 != packet["source_image_sha256"]
        ):
            raise ABOProvenanceError(
                "ABO human review identity differs from the packet"
            )
    reviewers = {decision.reviewer_id for decision in decisions}
    if len(reviewers) != 1:
        raise ABOProvenanceError("ABO human review must have one reviewer")
    roles_by_item: dict[str, dict[str, ABOImageReviewDecision]] = defaultdict(dict)
    for row in packet_rows:
        roles_by_item[row["item_id"]][row["image_role"]] = decision_by_key[
            (row["item_id"], row["image_id"])
        ]
    approved_pairs = sum(
        all(roles[role].decision == _APPROVED for role in ("main", "other"))
        for roles in roles_by_item.values()
    )
    counts = Counter(decision.decision for decision in decisions)
    expected_summary = {
        "approved_pair_candidates": approved_pairs,
        "decision_counts": dict(sorted(counts.items())),
        "formal_use_allowed": False,
        "human_decision_count": len(decisions),
        "packet_manifest_sha256": expected_packet_manifest_sha256,
        "rejected_pair_candidates": len(roles_by_item) - approved_pairs,
        "reviewer_id": next(iter(reviewers)),
        "schema_version": 1,
        "status": _HUMAN_STATUS,
        "total_pair_candidates": len(roles_by_item),
    }
    summary = parse_canonical_json(summary_bytes, label="ABO human review summary")
    if summary != expected_summary:
        raise ABOProvenanceError("ABO human review summary differs from ledger")


def _approved_pairs(
    packet_rows: tuple[dict[str, Any], ...],
    decisions: tuple[ABOImageReviewDecision, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    decision_by_key = {
        (decision.item_id, decision.image_id): decision for decision in decisions
    }
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in packet_rows:
        by_item[row["item_id"]][row["image_role"]] = row
    return {
        item_id: roles
        for item_id, roles in sorted(by_item.items())
        if all(
            decision_by_key[(item_id, roles[role]["image_id"])].decision == _APPROVED
            for role in ("main", "other")
        )
    }


def _fingerprint_approved_pairs(
    packet_root: Path,
    approved: dict[str, dict[str, dict[str, Any]]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
]:
    by_path: dict[str, dict[str, Any]] = {}
    identity_by_image_id: dict[str, tuple[str, str]] = {}
    pair_images: dict[str, dict[str, dict[str, Any]]] = {}
    associations: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    fingerprint_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for pair_id, roles in sorted(approved.items()):
        pair_images[pair_id] = {}
        for role in ("main", "other"):
            row = roles[role]
            local_path = _canonical_relative_path(row["local_path"])
            cached = by_path.get(local_path)
            if cached is None:
                content = _read_real_file_below(
                    packet_root,
                    local_path,
                    label=f"ABO approved image {local_path}",
                    max_bytes=_MAX_IMAGE_BYTES,
                )
                sha256 = hashlib.sha256(content).hexdigest()
                if sha256 != row["source_image_sha256"]:
                    raise ABOProvenanceError(
                        f"ABO approved image SHA drifted: {local_path}"
                    )
                cached = {
                    "bytes": len(content),
                    "phash": _phash(content, local_path),
                    "source_image_sha256": sha256,
                }
                by_path[local_path] = cached
            elif cached["source_image_sha256"] != row["source_image_sha256"]:
                raise ABOProvenanceError(
                    "ABO approved image path has contradictory SHA identities"
                )
            identity = (local_path, row["source_image_sha256"])
            previous = identity_by_image_id.setdefault(row["image_id"], identity)
            if previous != identity:
                raise ABOProvenanceError(
                    "ABO image_id has contradictory path/SHA identities"
                )
            evidence = {
                **cached,
                "image_id": row["image_id"],
                "image_role": role,
                "local_path": local_path,
            }
            pair_images[pair_id][role] = evidence
            fingerprint_key = (row["image_id"], local_path)
            fingerprint_by_identity.setdefault(
                fingerprint_key,
                {
                    **cached,
                    "image_id": row["image_id"],
                    "local_path": local_path,
                },
            )
            associations[fingerprint_key].append(
                {"image_role": role, "pair_id": pair_id}
            )
    fingerprints = [
        {
            **fingerprint,
            "pair_references": sorted(
                associations[key],
                key=lambda row: (row["pair_id"], _ROLE_ORDER[row["image_role"]]),
            ),
        }
        for key, fingerprint in sorted(fingerprint_by_identity.items())
    ]
    return pair_images, fingerprints


def _phash(content: bytes, label: str) -> str:
    try:
        with Image.open(io.BytesIO(content)) as opened:
            opened.load()
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
            return str(imagehash.phash(normalized, hash_size=8))
    except (OSError, ValueError) as error:
        raise ABOProvenanceError(
            f"ABO approved image cannot be decoded for pHash: {label}"
        ) from error


def _pair_conflicts(
    pair_images: dict[str, dict[str, dict[str, Any]]],
    *,
    max_hamming_distance: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[str, str]]]:
    within: list[dict[str, Any]] = []
    for pair_id, roles in sorted(pair_images.items()):
        evidence = _image_conflict(
            roles["main"],
            roles["other"],
            left_role="main",
            right_role="other",
            max_hamming_distance=max_hamming_distance,
        )
        if evidence is not None:
            within.append({"pair_id": pair_id, **evidence})

    cross: list[dict[str, Any]] = []
    edges: set[tuple[str, str]] = set()
    pair_ids = sorted(pair_images)
    for left_index, left_pair_id in enumerate(pair_ids):
        for right_pair_id in pair_ids[left_index + 1 :]:
            conflicts = []
            for left_role in ("main", "other"):
                for right_role in ("main", "other"):
                    evidence = _image_conflict(
                        pair_images[left_pair_id][left_role],
                        pair_images[right_pair_id][right_role],
                        left_role=left_role,
                        right_role=right_role,
                        max_hamming_distance=max_hamming_distance,
                    )
                    if evidence is not None:
                        conflicts.append(evidence)
            if conflicts:
                edges.add((left_pair_id, right_pair_id))
                cross.append(
                    {
                        "image_conflicts": sorted(
                            conflicts,
                            key=lambda row: (
                                _ROLE_ORDER[row["left_image_role"]],
                                _ROLE_ORDER[row["right_image_role"]],
                                row["left_image_id"],
                                row["right_image_id"],
                            ),
                        ),
                        "left_pair_id": left_pair_id,
                        "right_pair_id": right_pair_id,
                    }
                )
    return within, cross, edges


def _image_conflict(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_role: str,
    right_role: str,
    max_hamming_distance: int,
) -> dict[str, Any] | None:
    distance = (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count()
    same_content = left["source_image_sha256"] == right["source_image_sha256"]
    if not same_content and distance > max_hamming_distance:
        return None
    relations = []
    if same_content:
        relations.append("content_sha256")
    if distance <= max_hamming_distance:
        relations.append("phash_within_threshold")
    return {
        "left_image_id": left["image_id"],
        "left_image_role": left_role,
        "left_local_path": left["local_path"],
        "phash_hamming_distance": distance,
        "relations": relations,
        "right_image_id": right["image_id"],
        "right_image_role": right_role,
        "right_local_path": right["local_path"],
    }


def _maximum_independent_pair_ids(
    pair_ids: tuple[str, ...],
    conflict_edges: set[tuple[str, str]],
    *,
    forbidden_pair_ids: set[str] | None = None,
) -> tuple[str, ...]:
    """Return the lexicographically smallest exact maximum independent set."""

    forbidden = forbidden_pair_ids or set()
    eligible = tuple(sorted(set(pair_ids).difference(forbidden)))
    eligible_set = set(eligible)
    normalized_edges = {
        tuple(sorted(edge))
        for edge in conflict_edges
        if len(edge) == 2 and set(edge).issubset(eligible_set) and edge[0] != edge[1]
    }
    adjacency: dict[str, set[str]] = {pair_id: set() for pair_id in eligible}
    for left, right in normalized_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    retained: list[str] = []
    unseen = set(eligible)
    while unseen:
        start = min(unseen)
        component: set[str] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(
                sorted(adjacency[current].difference(component), reverse=True)
            )
        unseen.difference_update(component)
        if len(component) == 1:
            retained.extend(component)
            continue
        component_edges = frozenset(
            edge for edge in normalized_edges if set(edge).issubset(component)
        )
        retained.extend(
            _solve_independent_component(
                tuple(sorted(component)),
                component_edges,
            )
        )
    return tuple(sorted(retained))


def _solve_independent_component(
    component: tuple[str, ...],
    edges: frozenset[tuple[str, str]],
) -> tuple[str, ...]:
    @lru_cache(maxsize=None)
    def solve(active: tuple[str, ...]) -> tuple[str, ...]:
        active_set = set(active)
        edge = next(
            (
                candidate
                for candidate in sorted(edges)
                if candidate[0] in active_set and candidate[1] in active_set
            ),
            None,
        )
        if edge is None:
            return active
        left, right = edge
        without_left = solve(tuple(value for value in active if value != left))
        without_right = solve(tuple(value for value in active if value != right))
        if len(without_left) != len(without_right):
            return (
                without_left
                if len(without_left) > len(without_right)
                else without_right
            )
        return min(without_left, without_right)

    return solve(component)


def _validate_expected_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} digest must be lowercase SHA-256")


def _canonical_relative_path(value: str) -> str:
    if not value or value != value.strip() or "\\" in value:
        raise ABOProvenanceError("ABO pair audit path must be canonical POSIX")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ABOProvenanceError("ABO pair audit path must stay below its root")
    return path.as_posix()


def _read_real_file_below(
    root: Path,
    relative: str,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    canonical = _canonical_relative_path(relative)
    current = root
    parts = PurePosixPath(canonical).parts
    for part in parts[:-1]:
        current = current / part
        _require_real_directory(current, f"{label} parent")
    return read_stable_regular_file(
        current / parts[-1],
        label=label,
        max_bytes=max_bytes,
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
            raise ABOProvenanceError(f"{label} contains a non-file filesystem entry")


__all__ = [
    "AUDIT_ALGORITHM",
    "AUDIT_ID",
    "audit_abo_original_pairs",
    "load_verified_abo_human_review_decisions",
]
