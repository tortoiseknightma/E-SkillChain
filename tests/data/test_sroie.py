import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from skillchain.data.sroie import SROIEValidationError, validate_acquisition


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _artifact(path: Path, raw_root: Path, role: str, expected: dict) -> dict:
    return {
        "path": path.relative_to(raw_root).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected": expected,
    }


def test_validate_acquisition_accepts_identical_drive_copy_suffixes(tmp_path):
    raw = tmp_path / "raw"
    images = raw / "sroie" / "images.zip"
    text = raw / "sroie" / "text.zip"
    _write_zip(
        images,
        {
            "folder/A.jpg": b"image-a",
            "folder/A(1).jpg": b"image-a",
            "folder/B.jpg": b"image-b",
        },
    )
    _write_zip(text, {"folder/A.txt": b"text-a", "folder/B.txt": b"text-b"})
    manifest = {
        "schema_version": 1,
        "manifest_id": "fixture",
        "license_status": "pending",
        "artifacts": [
            _artifact(
                images,
                raw,
                "images",
                {
                    "physical_members": 3,
                    "logical_members_by_extension": {".jpg": 2},
                    "duplicate_logical_groups": 1,
                    "conflicting_duplicate_groups": 0,
                },
            ),
            _artifact(
                text,
                raw,
                "text",
                {
                    "physical_members": 2,
                    "logical_members_by_extension": {".txt": 2},
                    "duplicate_logical_groups": 0,
                    "conflicting_duplicate_groups": 0,
                },
            ),
        ],
        "pair_checks": [
            {
                "left_role": "images",
                "left_extension": ".jpg",
                "right_role": "text",
                "right_extension": ".txt",
                "expected_intersection": 2,
                "expected_left_only": [],
                "expected_right_only": [],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_acquisition(manifest_path, raw)

    assert report["artifact_count"] == 2
    assert report["status"] == "locally_verified_raw_not_formal_source_lock"


def test_validate_acquisition_rejects_conflicting_drive_copy_suffixes(tmp_path):
    raw = tmp_path / "raw"
    archive = raw / "sroie" / "bad.zip"
    _write_zip(archive, {"A.jpg": b"first", "A(1).jpg": b"different"})
    manifest = {
        "schema_version": 1,
        "manifest_id": "fixture",
        "license_status": "pending",
        "artifacts": [
            _artifact(
                archive,
                raw,
                "images",
                {
                    "physical_members": 2,
                    "logical_members_by_extension": {".jpg": 1},
                    "duplicate_logical_groups": 1,
                    "conflicting_duplicate_groups": 0,
                },
            )
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SROIEValidationError, match="structure differs"):
        validate_acquisition(manifest_path, raw)
