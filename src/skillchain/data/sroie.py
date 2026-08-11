"""Validate the owner-acquired official SROIE browser export."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from skillchain import config

DEFAULT_MANIFEST = (
    config.ROOT / "specs" / "data_sources" / "sroie-browser-acquisition-v1.json"
)
DEFAULT_RAW_ROOT = config.DATA_DIR / "raw"
COPY_SUFFIX = re.compile(r"\(\d+\)$")


class SROIEValidationError(RuntimeError):
    """Raised when the acquired archives differ from the reviewed snapshot."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_id(filename: str) -> str:
    return COPY_SUFFIX.sub("", PurePosixPath(filename).stem)


def _inspect_archive(path: Path) -> tuple[dict[str, object], dict[str, set[str]]]:
    logical_files: dict[tuple[str, str], list[zipfile.ZipInfo]] = defaultdict(list)
    identifiers: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        unsafe = [
            info.filename
            for info in infos
            if PurePosixPath(info.filename).is_absolute()
            or ".." in PurePosixPath(info.filename).parts
        ]
        if unsafe:
            raise SROIEValidationError(f"{path.name}: unsafe ZIP paths: {unsafe[:3]}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SROIEValidationError(f"{path.name}: CRC failure at {bad_member!r}")
        for info in infos:
            suffix = PurePosixPath(info.filename).suffix.lower()
            logical = _logical_id(info.filename)
            logical_files[(logical, suffix)].append(info)
            identifiers[suffix].add(logical)
        duplicate_groups = [group for group in logical_files.values() if len(group) > 1]
        conflicting = 0
        for group in duplicate_groups:
            digests = {hashlib.sha256(archive.read(info)).digest() for info in group}
            if len(digests) > 1:
                conflicting += 1
    counts = Counter(suffix for _, suffix in logical_files)
    return (
        {
            "physical_members": len(infos),
            "logical_members_by_extension": dict(sorted(counts.items())),
            "duplicate_logical_groups": len(duplicate_groups),
            "conflicting_duplicate_groups": conflicting,
        },
        identifiers,
    )


def validate_acquisition(manifest_path: Path, raw_root: Path) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SROIEValidationError("unsupported SROIE acquisition manifest schema")
    results: list[dict[str, object]] = []
    identifiers_by_role: dict[str, dict[str, set[str]]] = {}
    total_bytes = 0
    for artifact in manifest.get("artifacts", []):
        relative = PurePosixPath(str(artifact["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SROIEValidationError(f"unsafe artifact path: {relative}")
        path = Path(raw_root).joinpath(*relative.parts)
        if not path.is_file():
            raise SROIEValidationError(f"missing SROIE archive: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != artifact["bytes"]:
            raise SROIEValidationError(
                f"{path.name}: {actual_bytes} bytes, expected {artifact['bytes']}"
            )
        digest = _sha256(path)
        if digest != artifact["sha256"]:
            raise SROIEValidationError(f"{path.name}: SHA-256 mismatch")
        observed, identifiers = _inspect_archive(path)
        if observed != artifact["expected"]:
            raise SROIEValidationError(f"{path.name}: structure differs: {observed!r}")
        role = str(artifact["role"])
        identifiers_by_role[role] = identifiers
        total_bytes += actual_bytes
        results.append(
            {
                "role": role,
                "path": relative.as_posix(),
                "bytes": actual_bytes,
                "sha256": digest,
                **observed,
            }
        )
    for check in manifest.get("pair_checks", []):
        left = identifiers_by_role[str(check["left_role"])][
            str(check["left_extension"])
        ]
        right = identifiers_by_role[str(check["right_role"])][
            str(check["right_extension"])
        ]
        observed = {
            "intersection": len(left & right),
            "left_only": sorted(left - right),
            "right_only": sorted(right - left),
        }
        expected = {
            "intersection": check["expected_intersection"],
            "left_only": check["expected_left_only"],
            "right_only": check["expected_right_only"],
        }
        if observed != expected:
            raise SROIEValidationError(
                f"SROIE pair check differs: observed={observed!r}, expected={expected!r}"
            )
    return {
        "status": "locally_verified_raw_not_formal_source_lock",
        "manifest_id": manifest["manifest_id"],
        "artifacts": results,
        "artifact_count": len(results),
        "total_bytes": total_bytes,
        "license_status": manifest["license_status"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate_acquisition(args.manifest, args.raw_root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
