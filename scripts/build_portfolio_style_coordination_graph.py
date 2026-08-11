"""Build a query-blind, evidence-bound Core Style coordination graph.

The graph does not claim FashionIQ or ABO outfit co-occurrence.  It binds
Codex-reviewed candidate attributes to exact selected images and applies one
small deterministic coordination rule to FashionIQ ``dress`` anchors:

* reviewed neutral candidates are eligible for every dress anchor;
* reviewed colour candidates are eligible only when the anchor's deterministic
  local-image palette intersects the seed's compatible palette; and
* candidates reviewed for a men's audience never become dress edges.

No query, split, treatment, or score artifact is accepted as input.  The output
is canonical JSON, self-hashed, and published create-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import colorsys
from io import BytesIO
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    parse_strict_json,
    read_stable_regular_file,
    sha256_bytes,
)


SCHEMA_VERSION = 1
POLICY_VERSION = "portfolio-style-coordination-graph-v1"
ANNOTATION_POLICY_VERSION = "portfolio-style-coordination-candidate-review-v1"
GRAPH_KIND = "portfolio-style-coordination-graph"
SEED_KIND = "portfolio-style-coordination-candidate-seed"
RELATION_KIND = "portfolio_curated_coordination_rule"

MAX_SELECTION_BYTES = 8 * 1024 * 1024
MAX_CATALOG_BYTES = 16 * 1024 * 1024
MAX_SEED_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_ABO_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ABO_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ABO_RECORD_BYTES = 2 * 1024 * 1024

FORBIDDEN_FIELD_NAMES = frozenset(
    {"query_id", "split", "user_text", "query_text", "config", "score"}
)
ALLOWED_CATEGORIES = frozenset({"footwear", "bag", "jewelry"})
ALLOWED_AUDIENCES = frozenset({"women", "men", "unisex"})
ALLOWED_ELIGIBILITY_MODES = frozenset({"neutral_universal", "compatible_palette"})
ALLOWED_COLOR_FAMILIES = frozenset(
    {
        "black",
        "silver",
        "taupe",
        "rose",
        "cognac",
        "turquoise",
        "blue",
        "gold",
        "neutral",
        "olive",
    }
)
ALLOWED_FEATURE_TAGS = frozenset(
    {
        "ankle_boot",
        "block_heel",
        "low_block_heel",
        "sandal",
        "flat",
        "camera_bag",
        "crossbody",
        "small",
        "earrings",
        "drop",
        "silver_tone",
        "stud",
        "blue_stone",
        "gold_tone",
        "necklace",
        "pendant",
        "minimal",
        "circle",
        "loafer",
        "slip_on",
        "low_heel",
        "sneaker",
    }
)

# Palette labels emitted from anchor pixels.  The candidate-side public colour
# vocabulary above is deliberately smaller and manually reviewed.
PALETTE_ORDER = (
    "black",
    "gray",
    "white",
    "red",
    "rose",
    "orange",
    "gold",
    "yellow",
    "green",
    "turquoise",
    "blue",
    "purple",
    "brown",
    "taupe",
)
ALLOWED_ANCHOR_PALETTE = frozenset(PALETTE_ORDER)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID = re.compile(r"^asset\.v2\.[0-9a-f]{64}$")
_ABO_RECORD_ID = re.compile(r"^listing:([A-Z0-9]+)/image:([^/]+)$")
_ABO_METADATA_MEMBER = re.compile(r"^listings/metadata/listings_[0-9a-f]\.json\.gz$")


class StyleCoordinationGraphError(ValueError):
    """One input cannot support the query-blind coordination graph."""


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StyleCoordinationGraphError(f"{label} must be a JSON object")
    return value


def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise StyleCoordinationGraphError(f"{label} must be a JSON array")
    return value


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StyleCoordinationGraphError(f"{label} must be a non-empty trimmed string")
    return value


def _required_sha256(value: object, *, label: str) -> str:
    item = _required_string(value, label=label)
    if _SHA256.fullmatch(item) is None:
        raise StyleCoordinationGraphError(f"{label} must be a lowercase SHA-256")
    return item


def _exact_keys(
    value: Mapping[str, object],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        raise StyleCoordinationGraphError(
            f"{label} fields differ; missing={missing}, extra={extra}"
        )


def _reject_forbidden_fields(value: object, *, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in FORBIDDEN_FIELD_NAMES:
                raise StyleCoordinationGraphError(
                    f"forbidden field {key!r} at {location}"
                )
            _reject_forbidden_fields(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, location=f"{location}[{index}]")


def _stable_content(path: Path, *, label: str, max_bytes: int) -> bytes:
    try:
        return read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    except ArtifactFormatError as error:
        raise StyleCoordinationGraphError(str(error)) from error


def _canonical_object_input(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    content = _stable_content(path, label=label, max_bytes=max_bytes)
    try:
        value = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise StyleCoordinationGraphError(str(error)) from error
    return _object(value, label=label), content


def _seed_input(path: Path) -> tuple[dict[str, Any], bytes]:
    content = _stable_content(
        path, label="coordination candidate seed", max_bytes=MAX_SEED_BYTES
    )
    try:
        value = parse_strict_json(content, label="coordination candidate seed")
    except ArtifactFormatError as error:
        raise StyleCoordinationGraphError(str(error)) from error
    seed = _object(value, label="coordination candidate seed")
    _reject_forbidden_fields(seed, location="seed")
    return seed, content


def _catalog_input(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    content = _stable_content(
        path, label="runtime catalog assets", max_bytes=MAX_CATALOG_BYTES
    )
    try:
        values = parse_canonical_jsonl(content, label="runtime catalog assets")
    except ArtifactFormatError as error:
        raise StyleCoordinationGraphError(str(error)) from error
    rows = [_object(value, label="runtime catalog asset") for value in values]
    if not rows:
        raise StyleCoordinationGraphError("runtime catalog assets must not be empty")
    return rows, content


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    identity_matches = (
        left.st_dev == right.st_dev and left.st_ino == right.st_ino
        if left.st_ino and right.st_ino
        else True
    )
    return (
        identity_matches
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _hash_regular_file(
    path: Path, *, label: str, max_bytes: int
) -> tuple[str, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as error:
        raise StyleCoordinationGraphError(
            f"unable to inspect {label}: {path}"
        ) from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > max_bytes
    ):
        raise StyleCoordinationGraphError(
            f"{label} must be a regular non-symlink file <= {max_bytes} bytes"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not _same_snapshot(before, opened):
                raise StyleCoordinationGraphError(f"{label} changed before read")
            while block := handle.read(1024 * 1024):
                digest.update(block)
            after_read = os.fstat(handle.fileno())
        after = path.lstat()
    except StyleCoordinationGraphError:
        raise
    except OSError as error:
        raise StyleCoordinationGraphError(f"unable to read {label}: {path}") from error
    if not _same_snapshot(opened, after_read) or not _same_snapshot(after_read, after):
        raise StyleCoordinationGraphError(f"{label} changed during read")
    return digest.hexdigest(), after


def _relative_image_path(value: object, *, label: str) -> str:
    image_path = _required_string(value, label=label)
    pure = PurePosixPath(image_path)
    if (
        pure.is_absolute()
        or "\\" in image_path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise StyleCoordinationGraphError(
            f"{label} must be a normalized relative POSIX path"
        )
    return image_path


def _asset_path(asset_root: Path, image_path: str) -> Path:
    root = asset_root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(image_path).parts).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise StyleCoordinationGraphError("image path escapes asset root") from error
    return candidate


def _image_content(
    asset_root: Path,
    image_path: str,
    *,
    expected_sha256: str,
) -> bytes:
    path = _asset_path(asset_root, image_path)
    content = _stable_content(
        path, label=f"image {image_path}", max_bytes=MAX_IMAGE_BYTES
    )
    observed = sha256_bytes(content)
    if observed != expected_sha256:
        raise StyleCoordinationGraphError(
            f"image SHA-256 drift for {image_path}: {observed} != {expected_sha256}"
        )
    return content


def _palette_label(red: int, green: int, blue: int) -> str | None:
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
    degrees = hue * 360
    if value >= 0.94 and saturation <= 0.10:
        return None  # common catalogue background, not garment evidence
    if value <= 0.20:
        return "black"
    if saturation <= 0.16:
        return "gray" if value < 0.84 else "white"
    if 15 <= degrees < 45 and value < 0.62:
        return "brown"
    if 20 <= degrees < 55 and saturation < 0.38:
        return "taupe"
    if degrees < 12 or degrees >= 348:
        return "red"
    if degrees >= 325:
        return "rose"
    if degrees < 38:
        return "orange"
    if degrees < 55:
        return "gold"
    if degrees < 72:
        return "yellow"
    if degrees < 168:
        return "green"
    if degrees < 202:
        return "turquoise"
    if degrees < 258:
        return "blue"
    if degrees < 325:
        return "purple"
    return "rose"


def _anchor_palette(content: bytes, *, image_path: str) -> tuple[str, ...]:
    try:
        with Image.open(BytesIO(content)) as opened:
            opened.verify()
        with Image.open(BytesIO(content)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((64, 64), Image.Resampling.BILINEAR)
            labels = [
                label
                for pixel in image.get_flattened_data()
                if (label := _palette_label(*pixel)) is not None
            ]
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise StyleCoordinationGraphError(
            f"unable to decode image {image_path}"
        ) from error
    if not labels:
        raise StyleCoordinationGraphError(
            f"image {image_path} has no deterministic non-background palette"
        )
    counts = Counter(labels)
    threshold = max(1, int(len(labels) * 0.04))
    selected = tuple(label for label in PALETTE_ORDER if counts[label] >= threshold)
    if not selected:
        selected = (counts.most_common(1)[0][0],)
    return selected


def _string_list(
    value: object,
    *,
    label: str,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    raw = _array(value, label=label)
    items = tuple(_required_string(item, label=label) for item in raw)
    if not items or len(set(items)) != len(items):
        raise StyleCoordinationGraphError(
            f"{label} must be non-empty and duplicate-free"
        )
    if allowed is not None and not set(items).issubset(allowed):
        raise StyleCoordinationGraphError(
            f"{label} has unsupported values: {sorted(set(items) - allowed)}"
        )
    return items


def _load_seed_candidates(seed: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    _exact_keys(
        seed,
        required={
            "abo_listings_archive_sha256",
            "candidates",
            "kind",
            "policy_version",
            "review_boundary",
            "schema_version",
        },
        label="coordination candidate seed",
    )
    if seed.get("kind") != SEED_KIND or seed.get("schema_version") != 1:
        raise StyleCoordinationGraphError("unsupported coordination candidate seed")
    policy_version = _required_string(
        seed.get("policy_version"), label="seed policy_version"
    )
    if policy_version != ANNOTATION_POLICY_VERSION:
        raise StyleCoordinationGraphError(
            "unsupported coordination candidate seed policy_version"
        )
    _required_sha256(
        seed.get("abo_listings_archive_sha256"), label="seed ABO archive SHA-256"
    )
    review = _object(seed.get("review_boundary"), label="seed review_boundary")
    _exact_keys(
        review,
        required={
            "coordination_claim",
            "native_outfit_cooccurrence_claimed",
            "review_method",
        },
        label="seed review_boundary",
    )
    if review.get("native_outfit_cooccurrence_claimed") is not False:
        raise StyleCoordinationGraphError(
            "seed must not claim native outfit co-occurrence"
        )
    _required_string(review.get("coordination_claim"), label="seed coordination claim")
    _required_string(review.get("review_method"), label="seed review method")

    candidates: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    seen_products: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw in enumerate(
        _array(seed.get("candidates"), label="seed candidates")
    ):
        candidate = _object(raw, label=f"seed candidate {index}")
        _exact_keys(
            candidate,
            required={
                "asset_id",
                "audience",
                "category_l1",
                "color_families",
                "confidence",
                "display_title",
                "eligibility_mode",
                "feature_tags",
                "image_path",
                "image_sha256",
                "product_id",
                "source_metadata",
                "source_record_id",
            },
            optional={"compatible_anchor_color_families"},
            label=f"seed candidate {index}",
        )
        asset_id = _required_string(
            candidate.get("asset_id"), label=f"candidate {index} asset_id"
        )
        if _ASSET_ID.fullmatch(asset_id) is None:
            raise StyleCoordinationGraphError(f"candidate {index} has invalid asset_id")
        product_id = _required_string(
            candidate.get("product_id"), label=f"candidate {index} product_id"
        )
        if not product_id.startswith("abo:"):
            raise StyleCoordinationGraphError(
                f"candidate {index} must bind an ABO product"
            )
        image_path = _relative_image_path(
            candidate.get("image_path"), label=f"candidate {index} image_path"
        )
        image_sha256 = _required_sha256(
            candidate.get("image_sha256"), label=f"candidate {index} image SHA-256"
        )
        source_record_id = _required_string(
            candidate.get("source_record_id"),
            label=f"candidate {index} source_record_id",
        )
        record_match = _ABO_RECORD_ID.fullmatch(source_record_id)
        if record_match is None or product_id != f"abo:{record_match.group(1)}":
            raise StyleCoordinationGraphError(
                f"candidate {index} source identity is inconsistent"
            )
        category = _required_string(
            candidate.get("category_l1"), label=f"candidate {index} category_l1"
        )
        if category not in ALLOWED_CATEGORIES or category == "dress":
            raise StyleCoordinationGraphError(
                f"candidate {index} has same/unsupported category {category!r}"
            )
        audience = _required_string(
            candidate.get("audience"), label=f"candidate {index} audience"
        )
        if audience not in ALLOWED_AUDIENCES:
            raise StyleCoordinationGraphError(
                f"candidate {index} has unsupported audience"
            )
        colors = _string_list(
            candidate.get("color_families"),
            label=f"candidate {index} colors",
            allowed=ALLOWED_COLOR_FAMILIES,
        )
        tags = _string_list(
            candidate.get("feature_tags"),
            label=f"candidate {index} feature tags",
            allowed=ALLOWED_FEATURE_TAGS,
        )
        eligibility = _required_string(
            candidate.get("eligibility_mode"),
            label=f"candidate {index} eligibility_mode",
        )
        if eligibility not in ALLOWED_ELIGIBILITY_MODES:
            raise StyleCoordinationGraphError(
                f"candidate {index} has unsupported eligibility_mode"
            )
        compatible_raw = candidate.get("compatible_anchor_color_families")
        compatible: tuple[str, ...] = ()
        if eligibility == "compatible_palette":
            if compatible_raw is None:
                raise StyleCoordinationGraphError(
                    f"candidate {index} needs compatible anchor colors"
                )
            compatible = _string_list(
                compatible_raw, label=f"candidate {index} compatible colors"
            )
            if not set(compatible).issubset(ALLOWED_ANCHOR_PALETTE):
                raise StyleCoordinationGraphError(
                    f"candidate {index} has unsupported compatible anchor colors"
                )
        elif compatible_raw is not None:
            raise StyleCoordinationGraphError(
                f"candidate {index} neutral rule cannot declare compatible colors"
            )
        confidence = candidate.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 < float(confidence) <= 1
        ):
            raise StyleCoordinationGraphError(
                f"candidate {index} confidence must be in (0, 1]"
            )
        metadata = _object(
            candidate.get("source_metadata"), label=f"candidate {index} source_metadata"
        )
        _exact_keys(
            metadata,
            required={"archive_member", "item_id", "main_image_id", "product_type"},
            label=f"candidate {index} source_metadata",
        )
        normalized_metadata = {
            key: _required_string(
                metadata.get(key), label=f"candidate {index} source_metadata.{key}"
            )
            for key in ("archive_member", "item_id", "main_image_id", "product_type")
        }
        if (
            _ABO_METADATA_MEMBER.fullmatch(normalized_metadata["archive_member"])
            is None
        ):
            raise StyleCoordinationGraphError(
                f"candidate {index} has an invalid ABO metadata member"
            )
        if normalized_metadata["item_id"] != record_match.group(
            1
        ) or normalized_metadata["main_image_id"] != record_match.group(2):
            raise StyleCoordinationGraphError(
                f"candidate {index} source metadata identity is inconsistent"
            )
        if (
            asset_id in seen_assets
            or product_id in seen_products
            or image_path in seen_paths
        ):
            raise StyleCoordinationGraphError(
                "seed candidates contain duplicate asset/product/path"
            )
        seen_assets.add(asset_id)
        seen_products.add(product_id)
        seen_paths.add(image_path)
        candidates.append(
            {
                "asset_id": asset_id,
                "audience": audience,
                "category_l1": category,
                "color_families": colors,
                "compatible_anchor_color_families": compatible,
                "confidence": float(confidence),
                "display_title": _required_string(
                    candidate.get("display_title"),
                    label=f"candidate {index} display_title",
                ),
                "eligibility_mode": eligibility,
                "feature_tags": tags,
                "image_path": image_path,
                "image_sha256": image_sha256,
                "product_id": product_id,
                "source_metadata": normalized_metadata,
                "source_record_id": source_record_id,
            }
        )
    if not candidates:
        raise StyleCoordinationGraphError("seed candidates must not be empty")
    return policy_version, candidates


def _index_selection(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _array(selection.get("selections"), label="selection manifest selections")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _object(raw, label=f"selection {index}")
        path = _relative_image_path(
            row.get("destination_path"), label=f"selection {index} destination_path"
        )
        if path in result:
            raise StyleCoordinationGraphError(f"duplicate selection path: {path}")
        result[path] = row
    if not result:
        raise StyleCoordinationGraphError("selection manifest contains no selections")
    return result


def _index_catalog(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    seen_assets: set[str] = set()
    for index, row in enumerate(rows):
        path = _relative_image_path(
            row.get("local_path"), label=f"catalog row {index} local_path"
        )
        asset_id = _required_string(
            row.get("asset_id"), label=f"catalog row {index} asset_id"
        )
        if path in result or asset_id in seen_assets:
            raise StyleCoordinationGraphError(
                "runtime catalog has duplicate path or asset_id"
            )
        result[path] = row
        seen_assets.add(asset_id)
    return result


def _bound_asset(
    image_path: str,
    *,
    selection_by_path: Mapping[str, dict[str, Any]],
    catalog_by_path: Mapping[str, dict[str, Any]],
    asset_root: Path,
) -> tuple[dict[str, Any], bytes]:
    selection = selection_by_path.get(image_path)
    catalog = catalog_by_path.get(image_path)
    if selection is None or catalog is None:
        raise StyleCoordinationGraphError(
            f"unknown selection/catalog image: {image_path}"
        )
    draft = _object(selection.get("draft"), label=f"selection draft {image_path}")
    selection_sha = _required_sha256(
        selection.get("expected_sha256"), label=f"selection image SHA-256 {image_path}"
    )
    catalog_sha = _required_sha256(
        catalog.get("sha256"), label=f"catalog image SHA-256 {image_path}"
    )
    for field in ("product_id", "source_dataset", "source_record_id"):
        if draft.get(field) != catalog.get(field):
            raise StyleCoordinationGraphError(
                f"selection/catalog {field} drift for {image_path}"
            )
    if selection_sha != catalog_sha:
        raise StyleCoordinationGraphError(
            f"selection/catalog image SHA-256 drift for {image_path}"
        )
    content = _image_content(asset_root, image_path, expected_sha256=selection_sha)
    expected_bytes = selection.get("expected_bytes")
    if not isinstance(expected_bytes, int) or expected_bytes != len(content):
        raise StyleCoordinationGraphError(
            f"selection byte-count drift for {image_path}"
        )
    return catalog, content


def _verify_candidate_bindings(
    candidates: Sequence[dict[str, Any]],
    *,
    selection_by_path: Mapping[str, dict[str, Any]],
    catalog_by_path: Mapping[str, dict[str, Any]],
    asset_root: Path,
) -> None:
    for candidate in candidates:
        catalog, _content = _bound_asset(
            candidate["image_path"],
            selection_by_path=selection_by_path,
            catalog_by_path=catalog_by_path,
            asset_root=asset_root,
        )
        expected = {
            "asset_id": candidate["asset_id"],
            "product_id": candidate["product_id"],
            "source_dataset": "abo",
            "source_record_id": candidate["source_record_id"],
            "sha256": candidate["image_sha256"],
        }
        observed = {key: catalog.get(key) for key in expected}
        if observed != expected:
            raise StyleCoordinationGraphError(
                f"candidate selection/catalog binding drift for {candidate['image_path']}"
            )


def _verify_abo_metadata(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    candidates: Sequence[dict[str, Any]],
) -> str:
    observed_sha256, archive_snapshot = _hash_regular_file(
        archive_path,
        label="ABO listings archive",
        max_bytes=MAX_ABO_ARCHIVE_BYTES,
    )
    if observed_sha256 != expected_archive_sha256:
        raise StyleCoordinationGraphError("ABO listings archive SHA-256 mismatch")
    by_member: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in candidates:
        metadata = candidate["source_metadata"]
        by_member.setdefault(metadata["archive_member"], {})[metadata["item_id"]] = (
            candidate
        )
    found: dict[str, dict[str, Any]] = {}
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member_name, wanted in sorted(by_member.items()):
                try:
                    member = archive.getmember(member_name)
                except KeyError as error:
                    raise StyleCoordinationGraphError(
                        f"ABO source metadata member missing: {member_name}"
                    ) from error
                if not member.isfile() or member.size > MAX_ABO_MEMBER_BYTES:
                    raise StyleCoordinationGraphError(
                        f"ABO source metadata member is unsafe: {member_name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise StyleCoordinationGraphError(
                        f"ABO source metadata member cannot be read: {member_name}"
                    )
                with extracted, gzip.GzipFile(fileobj=extracted) as decompressed:
                    for line_number, raw_line in enumerate(decompressed, start=1):
                        if len(raw_line) > MAX_ABO_RECORD_BYTES:
                            raise StyleCoordinationGraphError(
                                f"ABO source metadata record too large: {member_name}:{line_number}"
                            )
                        try:
                            value = parse_strict_json(
                                raw_line,
                                label=f"ABO metadata {member_name}:{line_number}",
                            )
                        except ArtifactFormatError as error:
                            raise StyleCoordinationGraphError(str(error)) from error
                        row = _object(
                            value, label=f"ABO metadata {member_name}:{line_number}"
                        )
                        item_id = row.get("item_id")
                        if item_id in wanted:
                            if item_id in found:
                                raise StyleCoordinationGraphError(
                                    f"duplicate ABO source metadata for {item_id}"
                                )
                            found[item_id] = row
    except StyleCoordinationGraphError:
        raise
    except (OSError, tarfile.TarError, gzip.BadGzipFile, EOFError) as error:
        raise StyleCoordinationGraphError(
            "unable to read ABO listings archive"
        ) from error
    try:
        after = archive_path.lstat()
    except OSError as error:
        raise StyleCoordinationGraphError(
            "unable to re-inspect ABO listings archive"
        ) from error
    if not _same_snapshot(archive_snapshot, after):
        raise StyleCoordinationGraphError(
            "ABO listings archive changed during metadata verification"
        )

    for candidate in candidates:
        metadata = candidate["source_metadata"]
        row = found.get(metadata["item_id"])
        raw_product_types = row.get("product_type") if row is not None else None
        product_types = (
            {
                item.get("value")
                for item in raw_product_types
                if isinstance(item, dict) and isinstance(item.get("value"), str)
            }
            if isinstance(raw_product_types, list)
            else set()
        )
        if (
            row is None
            or row.get("item_id") != metadata["item_id"]
            or row.get("main_image_id") != metadata["main_image_id"]
            or metadata["product_type"] not in product_types
        ):
            raise StyleCoordinationGraphError(
                f"ABO source metadata mismatch for {candidate['product_id']}"
            )
    return observed_sha256


def _fashioniq_dress_anchors(
    *,
    selection_by_path: Mapping[str, dict[str, Any]],
    catalog_by_path: Mapping[str, dict[str, Any]],
    asset_root: Path,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for image_path, selection in selection_by_path.items():
        draft = _object(selection.get("draft"), label=f"selection draft {image_path}")
        if draft.get("source_dataset") != "fashioniq":
            continue
        source_record_id = draft.get("source_record_id")
        if not isinstance(source_record_id, str) or not source_record_id.startswith(
            "dress:"
        ):
            continue
        record_product_id = source_record_id.split(":", 1)[1]
        if (
            not record_product_id
            or draft.get("product_id") != f"fashioniq:{record_product_id}"
        ):
            raise StyleCoordinationGraphError(
                f"invalid FashionIQ dress identity: {image_path}"
            )
        catalog, content = _bound_asset(
            image_path,
            selection_by_path=selection_by_path,
            catalog_by_path=catalog_by_path,
            asset_root=asset_root,
        )
        asset_id = _required_string(
            catalog.get("asset_id"), label=f"anchor asset_id {image_path}"
        )
        if _ASSET_ID.fullmatch(asset_id) is None:
            raise StyleCoordinationGraphError(f"invalid anchor asset_id: {image_path}")
        anchors.append(
            {
                "asset_id": asset_id,
                "product_id": draft["product_id"],
                "source_dataset": "fashioniq",
                "source_record_id": source_record_id,
                "image_sha256": _required_sha256(
                    catalog.get("sha256"), label=f"anchor image SHA-256 {image_path}"
                ),
                "image_path": image_path,
                "category_l1": "dress",
                "color_families": list(_anchor_palette(content, image_path=image_path)),
            }
        )
    anchors.sort(key=lambda item: (item["asset_id"], item["image_path"]))
    if not anchors:
        raise StyleCoordinationGraphError("selection has no FashionIQ dress anchors")
    return anchors


def _candidate_output(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": candidate["asset_id"],
        "product_id": candidate["product_id"],
        "source_dataset": "abo",
        "source_record_id": candidate["source_record_id"],
        "image_sha256": candidate["image_sha256"],
        "image_path": candidate["image_path"],
        "category_l1": candidate["category_l1"],
        "display_title": candidate["display_title"],
        "audience": candidate["audience"],
        "feature_tags": list(candidate["feature_tags"]),
        "color_families": list(candidate["color_families"]),
    }


def _edge(
    *,
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
    annotation_policy_version: str,
) -> dict[str, Any]:
    rule = (
        "neutral_palette_rule"
        if candidate["eligibility_mode"] == "neutral_universal"
        else "compatible_palette_rule"
    )
    confidence = float(candidate["confidence"])
    identity = {
        "anchor_asset_id": anchor["asset_id"],
        "annotation_policy_version": annotation_policy_version,
        "candidate_asset_id": candidate["asset_id"],
        "relation_kind": RELATION_KIND,
        "rule": rule,
    }
    edge_id = f"style.edge.v1.{sha256_bytes(canonical_json_bytes(identity))}"
    return {
        "edge_id": edge_id,
        "anchor": dict(anchor),
        "candidate": _candidate_output(candidate),
        "relation_kind": RELATION_KIND,
        "confidence": confidence,
        "facets": [
            {"facet": "category", "value": candidate["category_l1"], "confidence": 1.0},
            {
                "facet": "palette",
                "value": ",".join(candidate["color_families"]),
                "confidence": 0.95,
            },
            {
                "facet": "verified_attributes",
                "value": ",".join(candidate["feature_tags"]),
                "confidence": 0.95,
            },
            {"facet": "coordination_rule", "value": rule, "confidence": confidence},
        ],
        "annotation_policy_version": annotation_policy_version,
    }


def build_graph(
    *,
    selection_manifest_path: Path,
    runtime_catalog_assets_path: Path,
    asset_root: Path,
    abo_listings_tar_path: Path,
    candidate_seed_path: Path,
) -> dict[str, Any]:
    """Validate exact inputs and return one self-hashed coordination graph."""

    selection, selection_content = _canonical_object_input(
        selection_manifest_path,
        label="selection manifest",
        max_bytes=MAX_SELECTION_BYTES,
    )
    catalog_rows, catalog_content = _catalog_input(runtime_catalog_assets_path)
    seed, seed_content = _seed_input(candidate_seed_path)
    annotation_policy_version, candidates = _load_seed_candidates(seed)
    selection_by_path = _index_selection(selection)
    catalog_by_path = _index_catalog(catalog_rows)
    _verify_candidate_bindings(
        candidates,
        selection_by_path=selection_by_path,
        catalog_by_path=catalog_by_path,
        asset_root=asset_root,
    )
    archive_sha256 = _verify_abo_metadata(
        abo_listings_tar_path,
        expected_archive_sha256=_required_sha256(
            seed.get("abo_listings_archive_sha256"),
            label="seed ABO archive SHA-256",
        ),
        candidates=candidates,
    )
    anchors = _fashioniq_dress_anchors(
        selection_by_path=selection_by_path,
        catalog_by_path=catalog_by_path,
        asset_root=asset_root,
    )

    edges: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_colors = set(anchor["color_families"])
        for candidate in sorted(candidates, key=lambda item: item["asset_id"]):
            if candidate["audience"] not in {"women", "unisex"}:
                continue
            if candidate["category_l1"] == anchor["category_l1"]:
                raise StyleCoordinationGraphError(
                    "same-category coordination edge is forbidden"
                )
            eligible = candidate["eligibility_mode"] == "neutral_universal" or bool(
                anchor_colors & set(candidate["compatible_anchor_color_families"])
            )
            if eligible:
                edges.append(
                    _edge(
                        anchor=anchor,
                        candidate=candidate,
                        annotation_policy_version=annotation_policy_version,
                    )
                )
    edges.sort(key=lambda item: item["edge_id"])
    edge_ids = [item["edge_id"] for item in edges]
    edge_pairs = [
        (item["anchor"]["asset_id"], item["candidate"]["asset_id"]) for item in edges
    ]
    if (
        not edges
        or len(set(edge_ids)) != len(edge_ids)
        or len(set(edge_pairs)) != len(edge_pairs)
    ):
        raise StyleCoordinationGraphError(
            "coordination graph has no edges or duplicate edges"
        )

    unsigned: dict[str, Any] = {
        "kind": GRAPH_KIND,
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "source_bindings": {
            "selection_manifest_sha256": sha256_bytes(selection_content),
            "runtime_catalog_assets_sha256": sha256_bytes(catalog_content),
            "abo_listings_archive_sha256": archive_sha256,
            "candidate_seed_sha256": sha256_bytes(seed_content),
        },
        "edges": edges,
    }
    _reject_forbidden_fields(unsigned, location="graph")
    graph = {
        **unsigned,
        "graph_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    return graph


def publish_graph(
    *,
    selection_manifest_path: Path,
    runtime_catalog_assets_path: Path,
    asset_root: Path,
    abo_listings_tar_path: Path,
    candidate_seed_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build and create one canonical graph without replacing existing bytes."""

    graph = build_graph(
        selection_manifest_path=selection_manifest_path,
        runtime_catalog_assets_path=runtime_catalog_assets_path,
        asset_root=asset_root,
        abo_listings_tar_path=abo_listings_tar_path,
        candidate_seed_path=candidate_seed_path,
    )
    atomic_create_file(output_path, canonical_json_bytes(graph))
    return graph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--runtime-catalog-assets", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--abo-listings-tar", required=True, type=Path)
    parser.add_argument("--candidate-seed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    graph = publish_graph(
        selection_manifest_path=arguments.selection_manifest,
        runtime_catalog_assets_path=arguments.runtime_catalog_assets,
        asset_root=arguments.asset_root,
        abo_listings_tar_path=arguments.abo_listings_tar,
        candidate_seed_path=arguments.candidate_seed,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "edge_count": len(graph["edges"]),
                "graph_sha256": graph["graph_sha256"],
                "output": str(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
