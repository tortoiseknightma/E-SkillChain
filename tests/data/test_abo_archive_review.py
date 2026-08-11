from __future__ import annotations

from io import BytesIO
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from PIL import Image

from skillchain.data.abo_archive_review import (
    SELECTION_POLICY_VERSION,
    _index_image_members,
    _materialize_candidates,
    _select_candidates,
)


def _image_bytes(color: str, image_format: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (9, 7), color).save(output, format=image_format)
    return output.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mtime = 0
    archive.addfile(member, BytesIO(content))


def _archives(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "abo-images-small.tar"
    image_rows = [
        ("AA+MAIN", "00/00000001.jpg", "red", "JPEG"),
        ("AB_OTHER", "00/00000002.jpg", "blue", "JPEG"),
        ("BA_MAIN", "00/00000003.jpg", "green", "JPEG"),
        ("BB_OTHER", "00/00000004.png", "yellow", "PNG"),
    ]
    with tarfile.open(images, "w") as archive:
        for _, path, color, image_format in image_rows:
            _add_bytes(
                archive,
                f"images/small/{path}",
                _image_bytes(color, image_format),
            )
        metadata = "image_id,height,width,path\n" + "".join(
            f"{image_id},7,9,{path}\n"
            for image_id, path, _, _ in image_rows
        )
        _add_bytes(
            archive,
            "images/metadata/images.csv.gz",
            gzip.compress(metadata.encode("utf-8"), mtime=0),
        )
    rows = [
        {
            "item_id": "ITEM_A",
            "main_image_id": "AA+MAIN",
            "other_image_id": ["AB_OTHER"],
        },
        {
            "item_id": "ITEM_B",
            "main_image_id": "BA_MAIN",
            "other_image_id": ["BB_OTHER"],
        },
        {
            "item_id": "ITEM_A",
            "main_image_id": "AA+MAIN",
            "other_image_id": ["AB_OTHER"],
        },
        {
            "item_id": "ITEM_MISSING",
            "main_image_id": "NO_MAIN",
            "other_image_id": ["NO_OTHER"],
        },
    ]
    raw = b"".join(
        json.dumps(value, sort_keys=True).encode("utf-8") + b"\n" for value in rows
    )
    listings = tmp_path / "abo-listings.tar"
    with tarfile.open(listings, "w") as archive:
        _add_bytes(
            archive,
            "listings/metadata/listings_0.json.gz",
            gzip.compress(raw, mtime=0),
        )
    return listings, images


def test_archive_review_selection_is_identity_only_and_deterministic(tmp_path: Path):
    listings, images = _archives(tmp_path)
    members = _index_image_members(images)

    candidates, scanned = _select_candidates(
        listings,
        members,
        maximum_pair_candidates=1,
    )

    expected = min(
        ("ITEM_A", "ITEM_B"),
        key=lambda item_id: hashlib.sha256(
            f"{SELECTION_POLICY_VERSION}:{item_id}".encode()
        ).hexdigest(),
    )
    assert scanned == 4
    assert [value["item_id"] for value in candidates] == [expected]
    assert candidates[0]["listing_line_number"] in (1, 2)


def test_archive_review_materializes_original_member_bytes(tmp_path: Path):
    listings, images = _archives(tmp_path)
    members = _index_image_members(images)
    candidates, _ = _select_candidates(
        listings,
        members,
        maximum_pair_candidates=2,
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    packet, files = _materialize_candidates(images, candidates, staging)

    assert len(packet) == 4
    assert len(files) == 4
    assert {value["packet_status"] for value in packet} == {
        "human_decision_required"
    }
    assert {value["image_role"] for value in packet} == {"main", "other"}
    for row in packet:
        content = (staging / row["local_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["source_image_sha256"]
        assert (row["width"], row["height"]) == (9, 7)
