from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from skillchain.data import abo_original_review
from skillchain.data.abo import (
    ABOImageManifestRecord,
    ABOListingRecord,
    ABOProvenanceError,
)
from skillchain.data.abo_original_review import (
    _build_packet_rows,
    _download_one,
    _parse_exported_decisions,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


def _image_bytes(width: int, height: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "magenta").save(output, format="JPEG")
    return output.getvalue()


class _Response:
    status = 200

    def __init__(self, content: bytes) -> None:
        self._content = content
        self.headers = {
            "Content-Type": "image/jpeg",
            "ETag": '"fixture-etag"',
            "Last-Modified": "Thu, 17 Jun 2021 13:42:33 GMT",
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, _maximum: int) -> bytes:
        return self._content


def test_download_one_preserves_official_original_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    content = _image_bytes(1200, 1800)
    monkeypatch.setattr(
        abo_original_review,
        "urlopen",
        lambda *_args, **_kwargs: _Response(content),
    )
    relative = "0a/01234567.jpg"
    uri = (
        "https://amazon-berkeley-objects.s3.amazonaws.com/"
        f"images/original/{relative}"
    )

    actual_relative, receipt = _download_one(
        relative,
        {"image_id": "71OkUYGf+eL", "official_image_uri": uri},
        tmp_path,
    )

    assert actual_relative == relative
    assert (tmp_path / "0a" / "01234567.jpg").read_bytes() == content
    assert (receipt["width"], receipt["height"]) == (1200, 1800)
    assert receipt["source_image_sha256"] == sha256_bytes(content)
    assert receipt["official_image_etag"] == '"fixture-etag"'


def test_original_review_rows_bind_formal_listing_and_image_records() -> None:
    listing = ABOListingRecord(
        item_id="ITEM-1",
        main_image_id="71OkUYGf+eL",
        other_image_id=("61Kp5bPJkhL",),
    )
    compact = {
        "archive_member": "images/small/0a/01234567.jpg",
        "image_id": "71OkUYGf+eL",
        "image_role": "main",
        "item_id": "ITEM-1",
        "source_image_sha256": "1" * 64,
    }
    receipt = {
        "height": 1800,
        "official_image_etag": '"fixture-etag"',
        "official_image_uri": (
            "https://amazon-berkeley-objects.s3.amazonaws.com/"
            "images/original/0a/01234567.jpg"
        ),
        "source_image_sha256": "2" * 64,
        "width": 1200,
    }

    [row] = _build_packet_rows(
        [compact],
        listing_records={"ITEM-1": listing},
        downloaded={"0a/01234567.jpg": receipt},
    )

    image = ABOImageManifestRecord(
        image_id="71OkUYGf+eL",
        path="0a/01234567.jpg",
        official_image_uri=receipt["official_image_uri"],
        official_image_etag=receipt["official_image_etag"],
        source_image_sha256=receipt["source_image_sha256"],
    )
    assert row["source_image_sha256"] == "2" * 64
    assert row["parent_small_image_sha256"] == "1" * 64
    assert row["listing_record_sha256"] == sha256_bytes(
        canonical_json_bytes(listing.model_dump(mode="json"))
    )
    assert row["image_record_sha256"] == sha256_bytes(
        canonical_json_bytes(image.model_dump(mode="json"))
    )


def test_exported_human_decision_parser_accepts_ui_order_and_rejects_duplicates() -> None:
    decision = {
        "schema_version": 1,
        "review_policy_version": "abo-catalog-photo-human-review-v1",
        "item_id": "ITEM-1",
        "image_id": "71OkUYGf+eL",
        "listing_record_sha256": "1" * 64,
        "image_record_sha256": "2" * 64,
        "source_image_sha256": "3" * 64,
        "decision": "approve_catalog_product_photo",
        "reviewer_kind": "human",
        "reviewer_id": "owner_1",
        "reviewed_at": "2026-07-25T12:00:00Z",
    }
    content = (json.dumps(decision, separators=(",", ":")) + "\n").encode()

    [parsed] = _parse_exported_decisions(content)

    assert parsed.image_id == "71OkUYGf+eL"
    duplicate = content.replace(
        b'"schema_version":1',
        b'"schema_version":1,"schema_version":1',
    )
    with pytest.raises(ABOProvenanceError, match="row 1 is invalid"):
        _parse_exported_decisions(duplicate)
