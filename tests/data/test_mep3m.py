import hashlib
import json
import os
import random
import re
import zlib
from dataclasses import replace
from pathlib import Path

import imagehash
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

import skillchain.data.mep3m as mep3m_module
from skillchain.data.mep3m import (
    ANNOTATION_SPEC,
    ARCHIVE_SPECS,
    PRODUCT_SCHEMA,
    SELECTED_SUBCLASS_IDS,
    MEP3MProvenanceError,
    MEP3MExtractionMember,
    MEP3MExtractionReceipt,
    MEP3MLicenseUseReview,
    MEP3MSourceLock,
    _parse_7zip_listing,
    _download_verified_file,
    audit_exact_eligibility,
    clean_dataset,
    export_dataset_asset_drafts,
    iter_selected_annotations,
    load_canonical_mep3m_draft_bundle,
    load_verified_mep3m_draft_bundle,
    load_verified_mep3m_extraction_receipt,
    load_verified_mep3m_license_use_review,
    load_verified_mep3m_provenance,
    load_verified_mep3m_source_lock,
    validate_rar_members,
    validate_remote_archive_index,
)


def _write_annotations(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)


def _annotation(sid: int, image_id: int, **updates) -> dict:
    row = {
        "class_id": 1,
        "class_name": "服饰",
        "sub_class_id": sid,
        "sub_class_name": f"子类-{sid}",
        "subsub_class_id": 9,
        "subsub_class_name": "连衣裙",
        "img_path": f"{sid}/{image_id}.jpg",
        "img_resolution": "300x300",
        "title": f"商品-{image_id}",
        "OCR": f"文字-{image_id}",
    }
    row.update(updates)
    return row


def _save_image(path: Path, *, size=(240, 240), pattern="left") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (220, 20, 20))
    draw = ImageDraw.Draw(image)
    if pattern == "left":
        draw.rectangle((0, 0, size[0] // 3, size[1]), fill=(10, 10, 10))
    elif pattern == "top":
        draw.rectangle((0, 0, size[0], size[1] // 3), fill=(10, 10, 10))
    else:
        draw.ellipse((20, 20, size[0] - 20, size[1] - 20), fill=(10, 10, 10))
    image.save(path, format="JPEG")


def _save_preencode_distinct_postencode_collision(left: Path, right: Path) -> None:
    width = 240
    source_rng = random.Random(0)
    original = bytearray(source_rng.randrange(256) for _ in range(width * width * 3))
    variant = bytearray(original)
    variant_rng = random.Random(12)
    for _ in range(100):
        index = variant_rng.randrange(len(variant))
        variant[index] = min(255, max(0, variant[index] + variant_rng.choice((-1, 1))))
    left.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", (width, width), bytes(original)).save(left, format="PNG")
    Image.frombytes("RGB", (width, width), bytes(variant)).save(right, format="PNG")


def _normalized_jpeg_phash(source: Path, destination: Path) -> str:
    with Image.open(source) as opened:
        opened.load()
        opened.convert("RGB").save(
            destination, format="JPEG", quality=90, optimize=True
        )
    with Image.open(destination) as normalized:
        normalized.load()
        return str(imagehash.phash(normalized.convert("RGB")))


def _write_products(path: Path, rows: list[dict], schema=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _legacy_muge_row(image_path="product_images/muge/muge-1.jpg") -> dict:
    return {
        "product_id": "muge-1",
        "title": "MUGE 商品",
        "category_l1": "unknown",
        "category_l2": None,
        "image_path": image_path,
        "source": "muge",
    }


def test_iter_selected_annotations_converts_false_and_round_robins_with_cap(tmp_path):
    path = tmp_path / "annotations.parquet"
    _write_annotations(
        path,
        [
            _annotation(34, 1),
            _annotation(34, 2, title="FALSE", OCR="FALSE", subsub_class_name="FALSE"),
            _annotation(34, 3),
            _annotation(63, 4),
            _annotation(63, 5),
            _annotation(999, 6),
        ],
    )

    rows = list(
        iter_selected_annotations(
            path, selected_ids=(34, 63), per_subclass=2, limit=4, batch_size=2
        )
    )

    assert [(row["sub_class_id"], row["img_path"]) for row in rows] == [
        (34, "34/1.jpg"),
        (63, "63/4.jpg"),
        (34, "34/2.jpg"),
        (63, "63/5.jpg"),
    ]
    assert rows[2]["title"] is None
    assert rows[2]["OCR"] is None
    assert rows[2]["subsub_class_name"] is None


@pytest.mark.parametrize(
    "member",
    ["../34/1.jpg", "34/../../evil.jpg", "/34/1.jpg", "34/a.jpg", "63/1.jpg"],
)
def test_validate_rar_members_rejects_unsafe_or_unexpected_paths(member):
    with pytest.raises(ValueError, match="RAR"):
        validate_rar_members(34, [member])


def test_validate_rar_members_normalizes_separators_and_allows_jpg_or_png():
    assert validate_rar_members(34, ["34\\1.jpg", "34/2.png"]) == (
        "34/1.jpg",
        "34/2.png",
    )


def test_validate_rar_members_rejects_encrypted_and_multivolume_archives():
    with pytest.raises(ValueError, match="encrypted"):
        validate_rar_members(34, ["34/1.jpg"], encrypted=True)
    with pytest.raises(ValueError, match="volume"):
        validate_rar_members(34, ["34/1.jpg"], volumes=2)


def test_parse_7zip_listing_skips_archive_header_and_reads_members():
    listing = """Path = C:\\data\\34.rar
Type = Rar5
Solid = -
Encrypted = -
Multivolume = -
Volumes = 1

----------
Path = 34\\1.jpg
Encrypted = -

Path = 34/2.png
Encrypted = -
"""

    assert _parse_7zip_listing(34, listing) == ("34/1.jpg", "34/2.png")


def test_parse_7zip_listing_rejects_multivolume_header():
    listing = """Path = C:\\data\\34.rar
Type = Rar5
Multivolume = +
Volumes = 2

----------
Path = 34/1.jpg
"""

    with pytest.raises(ValueError, match="volume"):
        _parse_7zip_listing(34, listing)


def test_parse_7zip_listing_ignores_only_explicit_subclass_root_directory():
    listing = """Path = 34\\1.jpg
Folder = -
Attributes = A

Path = 34/2.png
Folder = -
Attributes = A

Path = 34
Folder = +
Attributes = D
"""

    assert _parse_7zip_listing(34, listing) == ("34/1.jpg", "34/2.png")


@pytest.mark.parametrize(
    "directory",
    ["../34", "34/..", "34/nested", "34/../evil", "63"],
)
def test_parse_7zip_listing_rejects_any_other_directory_record(directory):
    listing = f"""Path = 34/1.jpg
Folder = -

Path = {directory}
Folder = +
Attributes = D
"""

    with pytest.raises(ValueError, match="RAR directory"):
        _parse_7zip_listing(34, listing)


def test_validate_extracted_staging_removes_exact_empty_subclass_root(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "1.jpg").write_bytes(b"image")
    (staging / "34").mkdir()

    mep3m_module._validate_extracted_staging(34, staging, ("34/1.jpg",))

    assert not (staging / "34").exists()
    assert [path.name for path in staging.iterdir()] == ["1.jpg"]


def test_validate_extracted_staging_rejects_wrong_empty_directory(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "1.jpg").write_bytes(b"image")
    (staging / "63").mkdir()

    with pytest.raises(RuntimeError, match="directory"):
        mep3m_module._validate_extracted_staging(34, staging, ("34/1.jpg",))


def test_validate_extracted_staging_rejects_nonempty_subclass_root(tmp_path):
    staging = tmp_path / "staging"
    root = staging / "34"
    root.mkdir(parents=True)
    (staging / "1.jpg").write_bytes(b"image")
    (root / "nested.jpg").write_bytes(b"nested")

    with pytest.raises(RuntimeError, match="empty"):
        mep3m_module._validate_extracted_staging(34, staging, ("34/1.jpg",))


def test_validate_extracted_staging_keeps_strict_file_set_reconciliation(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "1.jpg").write_bytes(b"image")

    with pytest.raises(RuntimeError, match="member validation"):
        mep3m_module._validate_extracted_staging(34, staging, ("34/2.jpg",))


def test_validate_remote_archive_index_rejects_size_or_hash_mismatch():
    items = [
        {
            "type": "file",
            "path": spec.path,
            "size": spec.size,
            "lfs": {"oid": spec.sha256, "size": spec.size},
        }
        for spec in ARCHIVE_SPECS.values()
    ]
    items[0]["lfs"]["oid"] = "0" * 64

    with pytest.raises(RuntimeError, match="remote archive index"):
        validate_remote_archive_index(items, require_full_index=False)


def test_archive_421_uses_official_lfs_sha256():
    assert ARCHIVE_SPECS[421].sha256 == (
        "6a7ea32e16b86290874991c0d5bb53f90bebfcea45dbefcf5b2b734c83850664"
    )


def test_all_archive_sha256_values_are_64_lowercase_hex_characters():
    invalid = {
        subclass_id: spec.sha256
        for subclass_id, spec in ARCHIVE_SPECS.items()
        if re.fullmatch(r"[0-9a-f]{64}", spec.sha256) is None
    }

    assert invalid == {}


@pytest.mark.parametrize(
    ("subclass_id", "official_sha256"),
    [
        (
            367,
            "185a402e96d91ac18cd6a268d023a2241b444b973fc4a965087a3611dfb773c0",
        ),
        (
            490,
            "765baf6fcb7a1e90e39cfaa23ab4c922c8436c6189cf3d92064e4751db0c5591",
        ),
    ],
)
def test_archive_uses_official_lfs_sha256(subclass_id, official_sha256):
    assert ARCHIVE_SPECS[subclass_id].sha256 == official_sha256


class _FakeResponse:
    def __init__(self, body: bytes, *, status_code=200):
        self.body = body
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeSession:
    def __init__(self, body: bytes):
        self.body = body
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(self.body, status_code=200)


def test_download_verified_file_rejects_wrong_final_hash(tmp_path):
    body = b"not-the-expected-file"
    session = _FakeSession(body)

    with pytest.raises(RuntimeError, match="SHA256"):
        _download_verified_file(
            "https://example.test/file",
            tmp_path / "file",
            expected_size=len(body),
            expected_sha256="0" * 64,
            session=session,
            sleep=lambda _: None,
        )


def test_download_verified_file_resumes_and_rejects_wrong_length(tmp_path):
    destination = tmp_path / "file"
    destination.write_bytes(b"abc")
    session = _FakeSession(b"replacement")

    with pytest.raises(RuntimeError, match="length"):
        _download_verified_file(
            "https://example.test/file",
            destination,
            expected_size=99,
            expected_sha256=hashlib.sha256(b"replacement").hexdigest(),
            session=session,
            sleep=lambda _: None,
        )

    assert session.calls[0][1]["headers"] == {"Range": "bytes=3-"}


def test_validate_local_artifact_rejects_same_size_sha256_tamper(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"wrong")
    spec = mep3m_module.FileSpec(
        path="artifact.bin",
        size=5,
        sha256=hashlib.sha256(b"right").hexdigest(),
    )

    with pytest.raises(RuntimeError, match="SHA256"):
        mep3m_module._validate_local_artifact(artifact, spec)


def test_extract_rejects_truncated_archive_before_subprocess(tmp_path, monkeypatch):
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "34.rar").write_bytes(b"truncated")
    subprocess_calls = []
    monkeypatch.setattr(mep3m_module, "ARCHIVES_DIR", archives)
    monkeypatch.setattr(mep3m_module, "_find_7zip", lambda: "7z")
    monkeypatch.setattr(
        mep3m_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="size"):
        mep3m_module.extract((34,))

    assert subprocess_calls == []


def test_probe_rejects_truncated_parquet_before_parquet_read(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "0000.parquet").write_bytes(b"truncated")
    parquet_reads = []
    monkeypatch.setattr(mep3m_module, "ANNOTATIONS_DIR", annotations)
    monkeypatch.setattr(
        mep3m_module.pq,
        "ParquetFile",
        lambda *args, **kwargs: parquet_reads.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="size"):
        mep3m_module.probe()

    assert parquet_reads == []


def test_probe_rejects_tampered_archive_before_parquet_read(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations"
    archives = tmp_path / "archives"
    annotations.mkdir()
    archives.mkdir()
    parquet_bytes = b"valid-parquet-placeholder"
    (annotations / "0000.parquet").write_bytes(parquet_bytes)
    (archives / "34.rar").write_bytes(b"wrong")
    monkeypatch.setattr(mep3m_module, "ANNOTATIONS_DIR", annotations)
    monkeypatch.setattr(mep3m_module, "ARCHIVES_DIR", archives)
    monkeypatch.setattr(mep3m_module, "SELECTED_SUBCLASS_IDS", (34,))
    monkeypatch.setattr(
        mep3m_module,
        "ANNOTATION_SPEC",
        mep3m_module.FileSpec(
            path="annotations/0000.parquet",
            size=len(parquet_bytes),
            sha256=hashlib.sha256(parquet_bytes).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        mep3m_module,
        "ARCHIVE_SPECS",
        {
            34: mep3m_module.FileSpec(
                path="Images/34.rar",
                size=5,
                sha256=hashlib.sha256(b"right").hexdigest(),
            )
        },
    )
    parquet_reads = []
    monkeypatch.setattr(
        mep3m_module.pq,
        "ParquetFile",
        lambda *args, **kwargs: parquet_reads.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="SHA256"):
        mep3m_module.probe()

    assert parquet_reads == []


def test_clean_rejects_truncated_parquet_before_clean_or_publish(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "0000.parquet").write_bytes(b"truncated")
    clean_calls = []
    monkeypatch.setattr(mep3m_module, "ANNOTATIONS_DIR", annotations)
    monkeypatch.setattr(
        mep3m_module,
        "clean_dataset",
        lambda **kwargs: clean_calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="size"):
        mep3m_module.clean()

    assert clean_calls == []


def test_clean_dataset_preserves_legacy_muge_and_writes_mep_hierarchy(tmp_path):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    muge_image = clean_dir / "product_images" / "muge" / "muge-1.jpg"
    _save_image(muge_image, pattern="left")
    legacy_schema = pa.schema(
        [
            ("product_id", pa.string()),
            ("title", pa.string()),
            ("category_l1", pa.string()),
            ("category_l2", pa.string()),
            ("image_path", pa.string()),
            ("source", pa.string()),
        ]
    )
    _write_products(parquet_path, [_legacy_muge_row()], legacy_schema)
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", pattern="top")

    report = clean_dataset(
        annotations=[_annotation(34, 1)],
        extracted_dir=extracted,
        output_dir=clean_dir / "product_images" / "mep3m",
        parquet_path=parquet_path,
        limit=1,
        required_subclass_ids=(34,),
    )

    assert report.kept == 1
    rows = pq.read_table(parquet_path).to_pylist()
    assert rows[0]["source"] == "muge"
    assert rows[0]["category_l1"] == "unknown"
    assert rows[0]["category_l3"] is None
    assert rows[0]["ocr_text"] is None
    assert rows[1] == {
        "product_id": "mep3m-34-1",
        "title": "商品-1",
        "category_l1": "服饰",
        "category_l2": "子类-34",
        "category_l3": "连衣裙",
        "ocr_text": "文字-1",
        "image_path": "product_images/mep3m/mep3m-34-1.jpg",
        "source": "mep3m",
    }


@pytest.mark.parametrize(
    "existing_image", [None, b"not-an-image"], ids=["missing", "damaged"]
)
def test_clean_dataset_fails_before_staging_for_unreadable_retained_image(
    tmp_path, monkeypatch, existing_image
):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    output_dir = clean_dir / "product_images" / "mep3m"
    output_dir.mkdir(parents=True)
    (output_dir / "old.jpg").write_bytes(b"old-mep-image")
    muge_image = clean_dir / "product_images" / "muge" / "muge-1.jpg"
    if existing_image is not None:
        muge_image.parent.mkdir(parents=True)
        muge_image.write_bytes(existing_image)
    _write_products(
        parquet_path,
        [{**_legacy_muge_row(), "category_l3": None, "ocr_text": None}],
        PRODUCT_SCHEMA,
    )
    old_parquet = parquet_path.read_bytes()
    staging_calls = []

    def unexpected_staging(*args, **kwargs):
        staging_calls.append((args, kwargs))
        raise AssertionError("staging must not start")

    monkeypatch.setattr(mep3m_module, "prepare_staging_directory", unexpected_staging)

    with pytest.raises(RuntimeError, match="retained product image"):
        clean_dataset(
            annotations=[],
            extracted_dir=tmp_path / "extracted",
            output_dir=output_dir,
            parquet_path=parquet_path,
            limit=1,
        )

    assert staging_calls == []
    assert (output_dir / "old.jpg").read_bytes() == b"old-mep-image"
    assert parquet_path.read_bytes() == old_parquet


def test_clean_dataset_filters_cross_source_duplicate_small_and_damaged(tmp_path):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    muge_image = clean_dir / "product_images" / "muge" / "muge-1.jpg"
    _save_image(muge_image, pattern="left")
    _write_products(parquet_path, [_legacy_muge_row()], PRODUCT_SCHEMA)
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", pattern="left")
    _save_image(extracted / "34" / "2.jpg", size=(100, 240), pattern="top")
    (extracted / "34" / "3.jpg").write_bytes(b"damaged")
    _save_image(extracted / "34" / "4.jpg", pattern="circle")

    report = clean_dataset(
        annotations=[_annotation(34, image_id) for image_id in range(1, 5)],
        extracted_dir=extracted,
        output_dir=clean_dir / "product_images" / "mep3m",
        parquet_path=parquet_path,
        limit=1,
        required_subclass_ids=(34,),
    )

    assert (report.kept, report.duplicates, report.too_small, report.damaged) == (
        1,
        1,
        1,
        1,
    )
    assert pq.read_table(parquet_path).column("product_id").to_pylist() == [
        "muge-1",
        "mep3m-34-4",
    ]


def test_clean_dataset_deduplicates_final_jpeg_phash_and_backfills(tmp_path):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    extracted = tmp_path / "extracted"
    first = extracted / "34" / "1.png"
    collision = extracted / "34" / "2.png"
    _save_preencode_distinct_postencode_collision(first, collision)
    _save_image(extracted / "34" / "3.jpg", pattern="circle")

    with Image.open(first) as first_source, Image.open(collision) as second_source:
        assert imagehash.phash(first_source) != imagehash.phash(second_source)
    assert _normalized_jpeg_phash(first, tmp_path / "first.jpg") == (
        _normalized_jpeg_phash(collision, tmp_path / "collision.jpg")
    )

    report = clean_dataset(
        annotations=[
            _annotation(34, 1, img_path="Images/34/1.png"),
            _annotation(34, 2, img_path="Images/34/2.png"),
            _annotation(34, 3),
        ],
        extracted_dir=extracted,
        output_dir=clean_dir / "product_images" / "mep3m",
        parquet_path=parquet_path,
        limit=2,
        required_subclass_ids=(34,),
    )

    assert report.duplicates == 1
    assert pq.read_table(parquet_path).column("product_id").to_pylist() == [
        "mep3m-34-1",
        "mep3m-34-3",
    ]
    assert not (clean_dir / "product_images" / "mep3m" / "mep3m-34-2.jpg").exists()


def test_clean_dataset_applies_cap_after_quality_filter_then_round_robins(tmp_path):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", size=(100, 240), pattern="left")
    _save_image(extracted / "34" / "2.jpg", pattern="top")
    _save_image(extracted / "34" / "3.jpg", pattern="circle")
    _save_image(extracted / "63" / "4.png", pattern="left")

    report = clean_dataset(
        annotations=[
            _annotation(34, 1, img_path="Images/34/1"),
            _annotation(34, 2, img_path="Images/34/2"),
            _annotation(34, 3, img_path="Images/34/3"),
            _annotation(63, 4, img_path="Images\\63\\4"),
        ],
        extracted_dir=extracted,
        output_dir=clean_dir / "product_images" / "mep3m",
        parquet_path=parquet_path,
        limit=2,
        per_subclass=1,
        required_subclass_ids=(34, 63),
    )

    assert report.too_small == 1
    assert pq.read_table(parquet_path).column("product_id").to_pylist() == [
        "mep3m-34-2",
        "mep3m-63-4",
    ]


def test_clean_dataset_checks_image_quality_before_skipping_full_subclass(tmp_path):
    clean_dir = tmp_path / "clean"
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    _save_image(extracted / "34" / "2.jpg", size=(100, 240), pattern="left")

    report = clean_dataset(
        annotations=[_annotation(34, 1), _annotation(34, 2)],
        extracted_dir=extracted,
        output_dir=clean_dir / "product_images" / "mep3m",
        parquet_path=clean_dir / "products.parquet",
        limit=1,
        per_subclass=1,
        required_subclass_ids=(34,),
    )

    assert report.too_small == 1


def test_clean_dataset_rejects_missing_required_extracted_directory_before_staging(
    tmp_path, monkeypatch
):
    staging_calls = []

    def unexpected_staging(*args, **kwargs):
        staging_calls.append((args, kwargs))
        raise AssertionError("staging must not start")

    monkeypatch.setattr(mep3m_module, "prepare_staging_directory", unexpected_staging)

    with pytest.raises(RuntimeError, match="extracted subclass directory"):
        clean_dataset(
            annotations=[_annotation(34, 1)],
            extracted_dir=tmp_path / "extracted",
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
        )

    assert staging_calls == []


def test_clean_dataset_rejects_missing_annotation_image_before_publish(tmp_path):
    extracted = tmp_path / "extracted"
    (extracted / "34").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="missing extracted image"):
        clean_dataset(
            annotations=[_annotation(34, 1)],
            extracted_dir=extracted,
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
        )

    assert not (tmp_path / "clean" / "products.parquet").exists()


def test_clean_dataset_rejects_required_subclass_with_no_valid_image(tmp_path):
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    (extracted / "63").mkdir(parents=True)
    (extracted / "63" / "2.jpg").write_bytes(b"damaged")

    with pytest.raises(RuntimeError, match="subclass coverage"):
        clean_dataset(
            annotations=[_annotation(34, 1), _annotation(63, 2)],
            extracted_dir=extracted,
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34, 63),
        )

    assert not (tmp_path / "clean" / "products.parquet").exists()


def test_clean_dataset_tracks_subclass_coverage_by_id_not_category_name(tmp_path):
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    _save_image(extracted / "63" / "2.jpg", pattern="circle")

    report = clean_dataset(
        annotations=[
            _annotation(34, 1, sub_class_name="same display name"),
            _annotation(63, 2, sub_class_name="same display name"),
        ],
        extracted_dir=extracted,
        output_dir=tmp_path / "clean" / "product_images" / "mep3m",
        parquet_path=tmp_path / "clean" / "products.parquet",
        limit=2,
        required_subclass_ids=(34, 63),
    )

    assert report.kept == 2


def test_full_selected_coverage_requires_fourteen_top_level_classes():
    with pytest.raises(RuntimeError, match="top-level class coverage"):
        mep3m_module._validate_selected_coverage(
            set(SELECTED_SUBCLASS_IDS),
            set(SELECTED_SUBCLASS_IDS),
            set(SELECTED_SUBCLASS_IDS),
            set(range(13)),
        )


def test_clean_dataset_refuses_insufficient_candidates_without_publishing(tmp_path):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    output_dir = clean_dir / "product_images" / "mep3m"
    output_dir.mkdir(parents=True)
    (output_dir / "old.jpg").write_bytes(b"old-image")
    _save_image(clean_dir / "product_images" / "muge" / "muge-1.jpg")
    old_rows = [{**_legacy_muge_row(), "category_l3": None, "ocr_text": None}]
    _write_products(parquet_path, old_rows, PRODUCT_SCHEMA)
    old_parquet = parquet_path.read_bytes()
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", pattern="top")

    with pytest.raises(RuntimeError, match="insufficient"):
        clean_dataset(
            annotations=[_annotation(34, 1)],
            extracted_dir=extracted,
            output_dir=output_dir,
            parquet_path=parquet_path,
            limit=2,
            required_subclass_ids=(34,),
        )

    assert (output_dir / "old.jpg").read_bytes() == b"old-image"
    assert parquet_path.read_bytes() == old_parquet


def test_clean_dataset_joint_publish_failure_restores_old_directory_and_parquet(
    tmp_path, monkeypatch
):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    output_dir = clean_dir / "product_images" / "mep3m"
    output_dir.mkdir(parents=True)
    (output_dir / "old.jpg").write_bytes(b"old-image")
    _save_image(clean_dir / "product_images" / "muge" / "muge-1.jpg")
    old_rows = [
        {**_legacy_muge_row(), "category_l3": None, "ocr_text": None},
        {
            "product_id": "mep3m-34-old",
            "title": "old",
            "category_l1": "old",
            "category_l2": "old",
            "category_l3": None,
            "ocr_text": None,
            "image_path": "product_images/mep3m/old.jpg",
            "source": "mep3m",
        },
    ]
    _write_products(parquet_path, old_rows, PRODUCT_SCHEMA)
    old_parquet = parquet_path.read_bytes()
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    real_replace = os.replace

    def fail_final_file_replace(source, destination):
        if (
            str(source).endswith("products.parquet.tmp")
            and Path(destination) == parquet_path
        ):
            raise PermissionError("simulated Windows file lock")
        return real_replace(source, destination)

    monkeypatch.setattr("skillchain.data.os.replace", fail_final_file_replace)

    with pytest.raises(PermissionError, match="file lock"):
        clean_dataset(
            annotations=[_annotation(34, 1)],
            extracted_dir=extracted,
            output_dir=output_dir,
            parquet_path=parquet_path,
            limit=1,
            required_subclass_ids=(34,),
        )

    assert (output_dir / "old.jpg").read_bytes() == b"old-image"
    assert not (output_dir / "mep3m-34-1.jpg").exists()
    assert parquet_path.read_bytes() == old_parquet


def test_clean_dataset_rerun_replaces_old_mep_rows(tmp_path):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    _save_image(clean_dir / "product_images" / "muge" / "muge-1.jpg")
    _write_products(
        parquet_path,
        [{**_legacy_muge_row(), "category_l3": None, "ocr_text": None}],
        PRODUCT_SCHEMA,
    )
    extracted = tmp_path / "extracted"
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    kwargs = {
        "annotations": [_annotation(34, 1)],
        "extracted_dir": extracted,
        "output_dir": clean_dir / "product_images" / "mep3m",
        "parquet_path": parquet_path,
        "limit": 1,
        "required_subclass_ids": (34,),
    }

    clean_dataset(**kwargs)
    clean_dataset(**kwargs)

    rows = pq.read_table(parquet_path).to_pylist()
    assert [row["product_id"] for row in rows] == ["muge-1", "mep3m-34-1"]
    assert ANNOTATION_SPEC.size == 233_520_842
    assert ANNOTATION_SPEC.sha256 == (
        "46f26558baa8343b3cafef64944b2221342811d6fdaad1a768c7e1d8d97e3de9"
    )
    assert ANNOTATION_SPEC.source_sha256 == (
        "930ace6fa1fd920252b5d285cac227b9ffd176e376b16c759df4fd620d72b451"
    )


def _formal_fixture(
    tmp_path: Path,
    monkeypatch,
    annotations: list[dict],
    *,
    stub_archive_verification: bool = True,
) -> tuple[Path, Path, dict[int, Path], MEP3MSourceLock]:
    annotation_path = tmp_path / "annotations.parquet"
    _write_annotations(annotation_path, annotations)
    annotation_bytes = annotation_path.read_bytes()
    annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    monkeypatch.setattr(
        mep3m_module,
        "ANNOTATION_SPEC",
        mep3m_module.AnnotationSpec(
            path="annotations.parquet",
            size=len(annotation_bytes),
            sha256=annotation_sha256,
            source_path="annotations.json",
            source_size=1,
            source_sha256="1" * 64,
            converter_url="https://example.test/converter",
        ),
    )
    archive_paths: dict[int, Path] = {}
    archive_specs = dict(mep3m_module.ARCHIVE_SPECS)
    for subclass_id in sorted({int(row["sub_class_id"]) for row in annotations}):
        archive_path = tmp_path / f"{subclass_id}.rar"
        archive_path.write_bytes(f"fixture-archive-{subclass_id}".encode())
        archive_paths[subclass_id] = archive_path
        archive_specs[subclass_id] = mep3m_module.FileSpec(
            path=f"Images/{subclass_id}.rar",
            size=archive_path.stat().st_size,
            sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        )
    monkeypatch.setattr(mep3m_module, "ARCHIVE_SPECS", archive_specs)
    if stub_archive_verification:
        monkeypatch.setattr(
            mep3m_module,
            "_verify_archive_receipt_members",
            lambda *_: None,
        )
    license_evidence = b"Fixture-only research permission evidence for MEP-3M tests.\n"
    (tmp_path / "mep3m-license-evidence.txt").write_bytes(license_evidence)
    lock = MEP3MSourceLock(
        source_revision=mep3m_module.REVISION,
        annotation_sha256=annotation_sha256,
        archive_sha256_by_subclass={
            subclass_id: archive_specs[subclass_id].sha256
            for subclass_id in archive_paths
        },
        license_id="research-only-fixture-license",
        license_evidence_sha256=hashlib.sha256(license_evidence).hexdigest(),
        license_evidence_url="https://example.test/mep3m/license-evidence.txt",
        source_url=(
            "https://huggingface.co/datasets/chendelong/MEP-3M/tree/"
            f"{mep3m_module.REVISION}"
        ),
        attribution="MEP-3M fixture authors",
    )
    return annotation_path, tmp_path / "extracted", archive_paths, lock


def _lock_args(tmp_path: Path, lock: MEP3MSourceLock) -> dict[str, object]:
    path = tmp_path / "mep3m-source-lock.json"
    content = (
        json.dumps(
            lock.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(content)
    review = MEP3MLicenseUseReview(
        source_lock_sha256=hashlib.sha256(content).hexdigest(),
        source_revision=lock.source_revision,
        license_id=lock.license_id,
        license_evidence_sha256=lock.license_evidence_sha256,
        license_evidence_uri=lock.license_evidence_url,
        license_evidence_revision="fixture-license-evidence-0123456789abcdef",
        decision="approved_local_noncommercial_research_only",
        reviewer_kind="human",
        reviewer_id="fixture-license-reviewer",
        reviewed_at="2026-07-21T01:02:03Z",
    )
    review_content = (
        json.dumps(
            review.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    review_path = tmp_path / "mep3m-human-license-use-review.json"
    review_path.write_bytes(review_content)
    return {
        "source_lock_path": path,
        "expected_source_lock_sha256": hashlib.sha256(content).hexdigest(),
        "license_evidence_path": tmp_path / "mep3m-license-evidence.txt",
        "license_review_path": review_path,
        "expected_license_review_sha256": hashlib.sha256(review_content).hexdigest(),
    }


def _extraction_args(
    tmp_path: Path,
    extracted: Path,
    archive_paths: dict[int, Path],
    annotations: list[dict],
) -> dict[str, object]:
    receipt_paths: dict[int, Path] = {}
    expected_digests: dict[int, str] = {}
    annotations_by_subclass: dict[int, list[dict]] = {}
    for annotation in annotations:
        annotations_by_subclass.setdefault(int(annotation["sub_class_id"]), []).append(
            annotation
        )
    for subclass_id, rows in sorted(annotations_by_subclass.items()):
        archive_path = archive_paths[subclass_id]
        members: list[MEP3MExtractionMember] = []
        for annotation in rows:
            source = mep3m_module._source_image_path(annotation, extracted)
            content = source.read_bytes()
            members.append(
                MEP3MExtractionMember(
                    member_path=f"{subclass_id}/{source.name}",
                    uncompressed_size=len(content),
                    crc32=f"{zlib.crc32(content) & 0xFFFFFFFF:08x}",
                    member_sha256=hashlib.sha256(content).hexdigest(),
                )
            )
        members.sort(key=lambda item: item.member_path)
        archive_bytes = archive_path.read_bytes()
        receipt = MEP3MExtractionReceipt(
            source_revision=mep3m_module.REVISION,
            archive_subclass_id=subclass_id,
            archive_filename=f"{subclass_id}.rar",
            archive_size=len(archive_bytes),
            archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
            members=tuple(members),
        )
        content = (
            json.dumps(
                receipt.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        path = tmp_path / f"mep3m-extraction-{subclass_id}.json"
        path.write_bytes(content)
        receipt_paths[subclass_id] = path
        expected_digests[subclass_id] = hashlib.sha256(content).hexdigest()
    return {
        "extraction_receipt_paths": receipt_paths,
        "expected_extraction_receipt_sha256_by_subclass": expected_digests,
    }


def _formal_clean_args(
    tmp_path: Path,
    lock: MEP3MSourceLock,
    extracted: Path,
    archive_paths: dict[int, Path],
    annotations: list[dict],
) -> dict[str, object]:
    return {
        **_lock_args(tmp_path, lock),
        **_extraction_args(tmp_path, extracted, archive_paths, annotations),
    }


def _provenance_args(product_root: Path) -> dict[str, str]:
    content = (product_root / "mep3m-provenance.jsonl").read_bytes()
    return {"expected_provenance_sha256": hashlib.sha256(content).hexdigest()}


def _seven_zip_listing(member_paths: list[str]) -> str:
    blocks = [
        "\n".join(
            (
                f"Path = {member_path}",
                "Folder = -",
                "Size = 1",
                "CRC = 00000000",
            )
        )
        for member_path in member_paths
    ]
    return (
        "Type = Rar5\nSolid = -\nMultivolume = -\n\n----------\n"
        + "\n\n".join(blocks)
        + "\n"
    )


def _install_fake_7zip_reader(
    monkeypatch,
    *,
    listing_members: list[str],
    extracted_bytes: dict[str, bytes],
) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(mep3m_module, "_find_7zip", lambda: "fixture-7z")

    def fake_run(command, **kwargs):
        del kwargs
        command = [str(value) for value in command]
        calls.append(command)
        if command[1] == "l":
            return mep3m_module.subprocess.CompletedProcess(
                command,
                0,
                stdout=_seven_zip_listing(listing_members),
                stderr="",
            )
        if command[1] != "x":
            raise AssertionError(f"unexpected 7-Zip action: {command}")
        output_argument = next(value for value in command if value.startswith("-o"))
        output_root = Path(output_argument[2:])
        for member_path, content in extracted_bytes.items():
            destination = output_root.joinpath(*member_path.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        return mep3m_module.subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(mep3m_module.subprocess, "run", fake_run)
    return calls


def _archive_receipt_loader_args(tmp_path, monkeypatch):
    annotations = [_annotation(34, 1)]
    _, extracted, archive_paths, lock = _formal_fixture(
        tmp_path,
        monkeypatch,
        annotations,
        stub_archive_verification=False,
    )
    source = extracted / "34" / "1.jpg"
    _save_image(source, pattern="top")
    receipt_args = _extraction_args(tmp_path, extracted, archive_paths, annotations)
    return {
        "receipt_path": receipt_args["extraction_receipt_paths"][34],
        "expected_receipt_sha256": receipt_args[
            "expected_extraction_receipt_sha256_by_subclass"
        ][34],
        "source_lock": lock,
        "archive_subclass_id": 34,
        "archive_snapshot": mep3m_module._digest_regular_file(
            archive_paths[34], "fixture archive"
        ),
        "member_path": "34/1.jpg",
        "member_bytes": source.read_bytes(),
    }


def test_formal_clean_exports_strict_drafts_without_claiming_multiview_identity(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1), _annotation(34, 2)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    _save_image(extracted / "34" / "2.jpg", pattern="circle")
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "mep3m"
    parquet_path = clean_root / "products.parquet"

    clean_dataset(
        annotations=annotations,
        extracted_dir=extracted,
        output_dir=product_root,
        parquet_path=parquet_path,
        limit=2,
        required_subclass_ids=(34,),
        annotation_source=annotation_path,
        archive_paths=archive_paths,
        **_formal_clean_args(tmp_path, lock, extracted, archive_paths, annotations),
    )
    provenance_args = _provenance_args(product_root)
    output = tmp_path / "mep3m-drafts.jsonl"
    drafts = export_dataset_asset_drafts(
        parquet_path=parquet_path,
        asset_root=clean_root,
        product_image_root=product_root,
        output_path=output,
        **_lock_args(tmp_path, lock),
        **provenance_args,
    )

    assert [draft.source_record_id for draft in drafts] == ["image:34/1", "image:34/2"]
    assert all(draft.source_revision == mep3m_module.REVISION for draft in drafts)
    assert all(
        draft.transform_policy_version == "mep3m-exif-rgb-jpeg-q90-optimize-v1"
        for draft in drafts
    )
    assert all(draft.derivation_parent_asset_ids == [] for draft in drafts)
    assert all(draft.license_id == lock.license_id for draft in drafts)
    assert all(draft.cloud_upload_allowed is False for draft in drafts)
    assert all(draft.public_demo_allowed is False for draft in drafts)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2

    coverage = audit_exact_eligibility(
        product_image_root=product_root,
        **_lock_args(tmp_path, lock),
        **provenance_args,
    )
    assert coverage.total_assets == 2
    assert coverage.eligible_assets == 0
    assert coverage.stable_product_groups == 0
    assert coverage.exclusion_reason == (
        "annotations_have_no_stable_product_or_view_group_identity"
    )
    provenance = [
        json.loads(line)
        for line in (product_root / "mep3m-provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["identity_scope"] == "single_asset_only" for row in provenance)
    assert all(row["exact_eligibility_candidate"] is False for row in provenance)
    assert all(row["normalized_asset_sha256"] for row in provenance)
    assert all(row["license_review_sha256"] for row in provenance)
    assert all(
        row["local_noncommercial_research_allowed"] is True for row in provenance
    )
    assert all(row["remote_embedding_allowed"] is False for row in provenance)
    assert all(row["redistribution_allowed"] is False for row in provenance)
    bundle_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    canonical_bundle = load_canonical_mep3m_draft_bundle(
        output, expected_bundle_sha256=bundle_digest
    )
    assert canonical_bundle.drafts == drafts
    assert canonical_bundle.formal_eligible is False
    with pytest.raises(MEP3MProvenanceError, match="source-unbound"):
        load_verified_mep3m_draft_bundle(output, expected_bundle_sha256=bundle_digest)
    original_bundle = output.read_bytes()
    duplicate_bundle = b'{"schema_version":1,' + original_bundle[1:]
    output.write_bytes(duplicate_bundle)
    with pytest.raises(MEP3MProvenanceError, match="row 1 is invalid"):
        load_canonical_mep3m_draft_bundle(
            output,
            expected_bundle_sha256=hashlib.sha256(duplicate_bundle).hexdigest(),
        )
    nonfinite_draft = json.loads(original_bundle.splitlines()[0])
    nonfinite_draft["public_demo_allowed"] = float("nan")
    nonfinite_bundle = (
        json.dumps(nonfinite_draft, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    output.write_bytes(nonfinite_bundle)
    with pytest.raises(MEP3MProvenanceError, match="row 1 is invalid"):
        load_canonical_mep3m_draft_bundle(
            output,
            expected_bundle_sha256=hashlib.sha256(nonfinite_bundle).hexdigest(),
        )
    output.write_bytes(original_bundle + b" ")
    with pytest.raises(MEP3MProvenanceError, match="external expected digest"):
        load_canonical_mep3m_draft_bundle(output, expected_bundle_sha256=bundle_digest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_revision", "main", "pinned MEP-3M revision"),
        ("source_revision", "unknown", "pinned MEP-3M revision"),
        ("source_revision", "", "pinned MEP-3M revision"),
        ("license_id", "unknown", "explicit"),
        ("license_id", "", "explicit"),
        ("research_use_allowed", True, "Extra inputs"),
    ],
)
def test_source_lock_rejects_mutable_unknown_or_unpermitted_provenance(
    tmp_path, monkeypatch, field, value, message
):
    annotation_path, _, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, [_annotation(34, 1)]
    )
    del annotation_path, archive_paths
    values = lock.model_dump(mode="python")
    values[field] = value
    with pytest.raises(ValidationError, match=message):
        MEP3MSourceLock(**values)


def test_formal_lock_loader_rejects_missing_pin_tamper_and_noncanonical_bytes(
    tmp_path, monkeypatch
):
    _, _, _, lock = _formal_fixture(tmp_path, monkeypatch, [_annotation(34, 1)])
    args = _lock_args(tmp_path, lock)
    path = args["source_lock_path"]
    expected = args["expected_source_lock_sha256"]
    assert (
        load_verified_mep3m_source_lock(path, expected_lock_sha256=expected).lock
        == lock
    )

    with pytest.raises(MEP3MProvenanceError, match="external"):
        load_verified_mep3m_source_lock(path, expected_lock_sha256="self-computed")

    changed = lock.model_copy(update={"attribution": "different approved attribution"})
    changed_content = (
        json.dumps(
            changed.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    path.write_bytes(changed_content)
    with pytest.raises(MEP3MProvenanceError, match="external expected digest"):
        load_verified_mep3m_source_lock(path, expected_lock_sha256=expected)

    nonfinite = changed_content.replace(b'"schema_version":1', b'"schema_version":NaN')
    path.write_bytes(nonfinite)
    with pytest.raises(MEP3MProvenanceError, match="strict schema"):
        load_verified_mep3m_source_lock(
            path, expected_lock_sha256=hashlib.sha256(nonfinite).hexdigest()
        )

    noncanonical = changed_content.rstrip() + b" \n"
    path.write_bytes(noncanonical)
    with pytest.raises(MEP3MProvenanceError, match="canonical JSON"):
        load_verified_mep3m_source_lock(
            path, expected_lock_sha256=hashlib.sha256(noncanonical).hexdigest()
        )


def test_license_use_review_is_independent_human_evidence_and_cannot_self_upgrade(
    tmp_path, monkeypatch
):
    _, _, _, lock = _formal_fixture(tmp_path, monkeypatch, [_annotation(34, 1)])
    arguments = _lock_args(tmp_path, lock)
    source = load_verified_mep3m_source_lock(
        arguments["source_lock_path"],
        expected_lock_sha256=arguments["expected_source_lock_sha256"],
    )
    review = load_verified_mep3m_license_use_review(
        arguments["license_review_path"],
        expected_review_sha256=arguments["expected_license_review_sha256"],
        source_lock=source,
        license_evidence_path=arguments["license_evidence_path"],
    )
    assert "research_use_allowed" not in source.lock.model_dump(mode="json")
    assert review.review.reviewer_kind == "human"
    assert review.review.license_evidence_uri == lock.license_evidence_url
    assert review.review.license_evidence_revision.endswith("0123456789abcdef")
    assert review.review.permissions.cloud_upload_allowed is False
    assert review.review.permissions.redistribution_allowed is False

    with pytest.raises(MEP3MProvenanceError, match="verified source lock"):
        load_verified_mep3m_license_use_review(
            arguments["license_review_path"],
            expected_review_sha256=arguments["expected_license_review_sha256"],
            source_lock=replace(source, _verification_token=None),
            license_evidence_path=arguments["license_evidence_path"],
        )

    review_path = Path(arguments["license_review_path"])
    forged = json.loads(review_path.read_text(encoding="utf-8"))
    forged["permissions"]["cloud_upload_allowed"] = True
    forged_content = (
        json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    review_path.write_bytes(forged_content)
    with pytest.raises(MEP3MProvenanceError, match="strict schema"):
        load_verified_mep3m_license_use_review(
            review_path,
            expected_review_sha256=hashlib.sha256(forged_content).hexdigest(),
            source_lock=source,
            license_evidence_path=arguments["license_evidence_path"],
        )


def test_formal_lock_loader_rejects_duplicate_keys_and_symlink(tmp_path, monkeypatch):
    _, _, _, lock = _formal_fixture(tmp_path, monkeypatch, [_annotation(34, 1)])
    args = _lock_args(tmp_path, lock)
    path = args["source_lock_path"]
    content = path.read_bytes()
    duplicate = b'{"schema_version":1,' + content[1:]
    path.write_bytes(duplicate)
    with pytest.raises(MEP3MProvenanceError, match="strict schema"):
        load_verified_mep3m_source_lock(
            path, expected_lock_sha256=hashlib.sha256(duplicate).hexdigest()
        )

    path.write_bytes(content)
    link = tmp_path / "mep3m-lock-link.json"
    try:
        os.symlink(path, link)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(MEP3MProvenanceError, match="regular non-symlink"):
        load_verified_mep3m_source_lock(
            link, expected_lock_sha256=hashlib.sha256(content).hexdigest()
        )


def test_formal_clean_rejects_lock_toctou_before_publish(tmp_path, monkeypatch):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    lock_args = _formal_clean_args(
        tmp_path, lock, extracted, archive_paths, annotations
    )
    lock_path = lock_args["source_lock_path"]
    original_verify = mep3m_module._verify_file_snapshot
    checks = 0

    def mutate_on_final_check(snapshot, label):
        nonlocal checks
        if label == "MEP-3M source lock":
            checks += 1
            if checks == 2:
                lock_path.write_bytes(lock_path.read_bytes() + b" ")
        return original_verify(snapshot, label)

    monkeypatch.setattr(mep3m_module, "_verify_file_snapshot", mutate_on_final_check)
    with pytest.raises(MEP3MProvenanceError, match="changed during"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **lock_args,
        )


def test_in_process_source_lock_cannot_request_formal_clean(tmp_path, monkeypatch):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    with pytest.raises(MEP3MProvenanceError, match="in-process"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "images",
            parquet_path=tmp_path / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            source_lock=lock,
        )


def test_formal_clean_rejects_archive_tamper_and_extracted_symlink(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "real.jpg", pattern="top")
    image_path = extracted / "34" / "1.jpg"
    try:
        os.symlink(extracted / "34" / "real.jpg", image_path)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    with pytest.raises(MEP3MProvenanceError, match="regular non-symlink"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **_formal_clean_args(tmp_path, lock, extracted, archive_paths, annotations),
        )

    image_path.unlink()
    _save_image(image_path, pattern="top")
    archive_paths[34].write_bytes(b"tampered-archive")
    with pytest.raises(MEP3MProvenanceError, match="external source lock"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **_formal_clean_args(tmp_path, lock, extracted, archive_paths, annotations),
        )


def test_legacy_clean_is_diagnostic_and_cannot_be_formally_exported(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    _, extracted, _, lock = _formal_fixture(tmp_path, monkeypatch, annotations)
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "mep3m"
    parquet_path = clean_root / "products.parquet"
    clean_dataset(
        annotations=annotations,
        extracted_dir=extracted,
        output_dir=product_root,
        parquet_path=parquet_path,
        limit=1,
        required_subclass_ids=(34,),
    )
    marker = json.loads(
        (product_root / "mep3m-diagnostic.json").read_text(encoding="utf-8")
    )
    assert marker["eligible_for_formal_export"] is False
    assert marker["exact_eligibility_coverage"] == 0
    with pytest.raises(MEP3MProvenanceError, match="diagnostic"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "drafts.jsonl",
            **_lock_args(tmp_path, lock),
            expected_provenance_sha256="0" * 64,
        )


def test_formal_clean_detects_extracted_image_change_before_publish(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    source_image = extracted / "34" / "1.jpg"
    _save_image(source_image, pattern="top")
    original_writer = mep3m_module._write_normalized_jpeg

    def mutate_source_after_normalization(image, destination):
        result = original_writer(image, destination)
        _save_image(source_image, pattern="circle")
        return result

    monkeypatch.setattr(
        mep3m_module, "_write_normalized_jpeg", mutate_source_after_normalization
    )
    formal_args = _formal_clean_args(
        tmp_path, lock, extracted, archive_paths, annotations
    )
    output_dir = tmp_path / "clean" / "product_images" / "mep3m"
    with pytest.raises(MEP3MProvenanceError, match="changed during"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=output_dir,
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **formal_args,
        )
    assert not output_dir.exists()


def test_formal_export_revalidates_lock_and_rejects_sidecar_rewrite(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "mep3m"
    parquet_path = clean_root / "products.parquet"
    clean_dataset(
        annotations=annotations,
        extracted_dir=extracted,
        output_dir=product_root,
        parquet_path=parquet_path,
        limit=1,
        required_subclass_ids=(34,),
        annotation_source=annotation_path,
        archive_paths=archive_paths,
        **_formal_clean_args(tmp_path, lock, extracted, archive_paths, annotations),
    )
    provenance_args = _provenance_args(product_root)

    forged = MEP3MSourceLock.model_construct(
        **{**lock.model_dump(mode="python"), "source_revision": "main"}
    )
    with pytest.raises(MEP3MProvenanceError, match="strict schema"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "forged.jsonl",
            **_lock_args(tmp_path, forged),
            **provenance_args,
        )

    sidecar = product_root / "mep3m-provenance.jsonl"
    rewritten = json.loads(sidecar.read_text(encoding="utf-8"))
    rewritten["license_id"] = "attacker-license"
    sidecar.write_text(
        json.dumps(
            rewritten,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MEP3MProvenanceError, match="external expected digest"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "rewritten.jsonl",
            **_lock_args(tmp_path, lock),
            **provenance_args,
        )


def test_formal_clean_requires_external_receipt_and_rejects_unrelated_extracted_bytes(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    source = extracted / "34" / "1.jpg"
    _save_image(source, pattern="top")
    lock_args = _lock_args(tmp_path, lock)
    with pytest.raises(MEP3MProvenanceError, match="extraction_receipt_paths"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "missing-receipt" / "mep3m",
            parquet_path=tmp_path / "missing-receipt" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **lock_args,
        )

    formal_args = _formal_clean_args(
        tmp_path, lock, extracted, archive_paths, annotations
    )
    _save_image(source, pattern="circle")
    with pytest.raises(MEP3MProvenanceError, match="archive member"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "changed-member" / "mep3m",
            parquet_path=tmp_path / "changed-member" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **formal_args,
        )


def test_formal_clean_hashes_license_evidence_bytes(tmp_path, monkeypatch):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    formal_args = _formal_clean_args(
        tmp_path, lock, extracted, archive_paths, annotations
    )
    Path(formal_args["license_evidence_path"]).write_text(
        "attacker-provided license assertion\n", encoding="utf-8"
    )
    with pytest.raises(MEP3MProvenanceError, match="license evidence bytes"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **formal_args,
        )


def test_formal_clean_rejects_empty_digest_matched_license_evidence(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    license_path = tmp_path / "mep3m-license-evidence.txt"
    license_path.write_bytes(b"")
    empty_lock = MEP3MSourceLock.model_validate(
        {
            **lock.model_dump(mode="python"),
            "license_evidence_sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    formal_args = _formal_clean_args(
        tmp_path, empty_lock, extracted, archive_paths, annotations
    )

    with pytest.raises(MEP3MProvenanceError, match="must not be empty"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **formal_args,
        )


def test_receipt_loader_reextracts_and_hashes_locked_archive_members(
    tmp_path, monkeypatch
):
    values = _archive_receipt_loader_args(tmp_path, monkeypatch)
    calls = _install_fake_7zip_reader(
        monkeypatch,
        listing_members=[values["member_path"]],
        extracted_bytes={values["member_path"]: values["member_bytes"]},
    )

    verified = load_verified_mep3m_extraction_receipt(
        values["receipt_path"],
        expected_receipt_sha256=values["expected_receipt_sha256"],
        source_lock=values["source_lock"],
        archive_subclass_id=values["archive_subclass_id"],
        archive_snapshot=values["archive_snapshot"],
    )

    assert verified.receipt.members[0].member_path == values["member_path"]
    assert [command[1] for command in calls] == ["l", "x"]


@pytest.mark.parametrize(
    ("listing_members", "message"),
    [
        (["../outside.jpg"], "listing output is invalid"),
        (["34/2.jpg"], "absent from the verified archive"),
    ],
)
def test_receipt_loader_rejects_invalid_or_unmatched_7zip_listing(
    tmp_path, monkeypatch, listing_members, message
):
    values = _archive_receipt_loader_args(tmp_path, monkeypatch)
    calls = _install_fake_7zip_reader(
        monkeypatch,
        listing_members=listing_members,
        extracted_bytes={},
    )

    with pytest.raises(MEP3MProvenanceError, match=message):
        load_verified_mep3m_extraction_receipt(
            values["receipt_path"],
            expected_receipt_sha256=values["expected_receipt_sha256"],
            source_lock=values["source_lock"],
            archive_subclass_id=values["archive_subclass_id"],
            archive_snapshot=values["archive_snapshot"],
        )

    assert [command[1] for command in calls] == ["l"]


def test_receipt_loader_rejects_extracted_member_bytes_that_differ_from_receipt(
    tmp_path, monkeypatch
):
    values = _archive_receipt_loader_args(tmp_path, monkeypatch)
    _install_fake_7zip_reader(
        monkeypatch,
        listing_members=[values["member_path"]],
        extracted_bytes={values["member_path"]: b"different member bytes"},
    )

    with pytest.raises(MEP3MProvenanceError, match="member bytes do not match"):
        load_verified_mep3m_extraction_receipt(
            values["receipt_path"],
            expected_receipt_sha256=values["expected_receipt_sha256"],
            source_lock=values["source_lock"],
            archive_subclass_id=values["archive_subclass_id"],
            archive_snapshot=values["archive_snapshot"],
        )


def test_receipt_loader_fails_closed_when_7zip_is_unavailable(tmp_path, monkeypatch):
    values = _archive_receipt_loader_args(tmp_path, monkeypatch)

    def missing_7zip():
        raise RuntimeError("7-Zip was not found")

    monkeypatch.setattr(mep3m_module, "_find_7zip", missing_7zip)
    with pytest.raises(MEP3MProvenanceError, match="requires 7-Zip"):
        load_verified_mep3m_extraction_receipt(
            values["receipt_path"],
            expected_receipt_sha256=values["expected_receipt_sha256"],
            source_lock=values["source_lock"],
            archive_subclass_id=values["archive_subclass_id"],
            archive_snapshot=values["archive_snapshot"],
        )


def test_receipt_loader_rejects_duplicate_nonfinite_and_wrong_external_digest(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    _, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    receipt_args = _extraction_args(tmp_path, extracted, archive_paths, annotations)
    receipt_path = receipt_args["extraction_receipt_paths"][34]
    expected = receipt_args["expected_extraction_receipt_sha256_by_subclass"][34]
    archive_snapshot = mep3m_module._digest_regular_file(
        archive_paths[34], "fixture archive"
    )
    assert (
        load_verified_mep3m_extraction_receipt(
            receipt_path,
            expected_receipt_sha256=expected,
            source_lock=lock,
            archive_subclass_id=34,
            archive_snapshot=archive_snapshot,
        )
        .receipt.members[0]
        .member_path
        == "34/1.jpg"
    )

    original = receipt_path.read_bytes()
    duplicate = b'{"schema_version":1,' + original[1:]
    receipt_path.write_bytes(duplicate)
    with pytest.raises(MEP3MProvenanceError, match="strict schema"):
        load_verified_mep3m_extraction_receipt(
            receipt_path,
            expected_receipt_sha256=hashlib.sha256(duplicate).hexdigest(),
            source_lock=lock,
            archive_subclass_id=34,
            archive_snapshot=archive_snapshot,
        )

    nonfinite = original.replace(b'"archive_size":', b'"archive_size":NaN,"x":')
    receipt_path.write_bytes(nonfinite)
    with pytest.raises(MEP3MProvenanceError, match="strict schema"):
        load_verified_mep3m_extraction_receipt(
            receipt_path,
            expected_receipt_sha256=hashlib.sha256(nonfinite).hexdigest(),
            source_lock=lock,
            archive_subclass_id=34,
            archive_snapshot=archive_snapshot,
        )

    receipt_path.write_bytes(original)
    with pytest.raises(MEP3MProvenanceError, match="external expected digest"):
        load_verified_mep3m_extraction_receipt(
            receipt_path,
            expected_receipt_sha256="f" * 64,
            source_lock=lock,
            archive_subclass_id=34,
            archive_snapshot=archive_snapshot,
        )


def test_formal_export_rehashes_normalized_asset_and_bundle_is_externally_locked(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "mep3m"
    parquet_path = clean_root / "products.parquet"
    clean_dataset(
        annotations=annotations,
        extracted_dir=extracted,
        output_dir=product_root,
        parquet_path=parquet_path,
        limit=1,
        required_subclass_ids=(34,),
        annotation_source=annotation_path,
        archive_paths=archive_paths,
        **_formal_clean_args(tmp_path, lock, extracted, archive_paths, annotations),
    )
    provenance_args = _provenance_args(product_root)
    sidecar = product_root / "mep3m-provenance.jsonl"
    records = load_verified_mep3m_provenance(sidecar, **provenance_args)
    assert (
        records[0].normalized_asset_sha256
        == hashlib.sha256((product_root / records[0].filename).read_bytes()).hexdigest()
    )

    original_sidecar = sidecar.read_bytes()
    duplicate = b'{"schema_version":3,' + original_sidecar[1:]
    sidecar.write_bytes(duplicate)
    with pytest.raises(MEP3MProvenanceError, match="row 1 is invalid"):
        load_verified_mep3m_provenance(
            sidecar,
            expected_provenance_sha256=hashlib.sha256(duplicate).hexdigest(),
        )
    nonfinite_row = json.loads(original_sidecar)
    nonfinite_row["archive_member_uncompressed_size"] = float("nan")
    nonfinite = (
        json.dumps(nonfinite_row, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    sidecar.write_bytes(nonfinite)
    with pytest.raises(MEP3MProvenanceError, match="row 1 is invalid"):
        load_verified_mep3m_provenance(
            sidecar,
            expected_provenance_sha256=hashlib.sha256(nonfinite).hexdigest(),
        )
    sidecar.write_bytes(original_sidecar)

    _save_image(product_root / records[0].filename, pattern="circle")
    with pytest.raises(MEP3MProvenanceError, match="frozen provenance"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "tampered-drafts.jsonl",
            **_lock_args(tmp_path, lock),
            **provenance_args,
        )


def test_formal_clean_rejects_hardlinked_extracted_file(tmp_path, monkeypatch):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    real = extracted / "34" / "real.jpg"
    image_path = extracted / "34" / "1.jpg"
    _save_image(real, pattern="top")
    try:
        os.link(real, image_path)
    except OSError:
        pytest.skip("hard-link creation is unavailable on this filesystem")
    formal_args = _formal_clean_args(
        tmp_path, lock, extracted, archive_paths, annotations
    )
    with pytest.raises(MEP3MProvenanceError, match="hard-linked"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=tmp_path / "clean" / "product_images" / "mep3m",
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **formal_args,
        )


def test_formal_clean_rejects_reparse_staging_residue_without_deleting_target(
    tmp_path, monkeypatch
):
    annotations = [_annotation(34, 1)]
    annotation_path, extracted, archive_paths, lock = _formal_fixture(
        tmp_path, monkeypatch, annotations
    )
    _save_image(extracted / "34" / "1.jpg", pattern="top")
    output_dir = tmp_path / "clean" / "product_images" / "mep3m"
    output_dir.parent.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "must-survive.txt").write_text("keep", encoding="utf-8")
    staging_residue = output_dir.with_name(".mep3m.staging")
    try:
        os.symlink(victim, staging_residue, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable on this Windows host")

    with pytest.raises(MEP3MProvenanceError, match="symlink, junction, or reparse"):
        clean_dataset(
            annotations=annotations,
            extracted_dir=extracted,
            output_dir=output_dir,
            parquet_path=tmp_path / "clean" / "products.parquet",
            limit=1,
            required_subclass_ids=(34,),
            annotation_source=annotation_path,
            archive_paths=archive_paths,
            **_formal_clean_args(tmp_path, lock, extracted, archive_paths, annotations),
        )
    assert (victim / "must-survive.txt").read_text(encoding="utf-8") == "keep"
