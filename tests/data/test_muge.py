import base64
import hashlib
import io
import json
import os
import random
from dataclasses import replace
from pathlib import Path

import imagehash
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from PIL import ImageDraw
from pydantic import ValidationError

import skillchain.data.muge as muge_module
from skillchain.data.muge import (
    MUGECandidateDisposition,
    MUGEDispositionLedger,
    MUGELicenseReview,
    MUGEProvenanceError,
    MUGESelectionPlan,
    MUGESourceLock,
    clean_dataset,
    compute_muge_retained_gallery_binding,
    export_dataset_asset_drafts,
    load_verified_muge_license_review,
    load_verified_muge_draft_bundle,
    load_verified_muge_selection,
    load_verified_muge_selection_plan,
    load_verified_muge_source_lock,
    materialize_query_images,
    read_text_records,
    write_preview,
)


def _encoded_image(
    size=(240, 240), color=(220, 20, 20), image_format="PNG", pattern="left"
) -> str:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    if pattern == "left":
        draw.rectangle((0, 0, size[0] // 3, size[1]), fill=(10, 10, 10))
    else:
        draw.rectangle((0, 0, size[0], size[1] // 3), fill=(10, 10, 10))
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encoded_preencode_distinct_postencode_collision() -> tuple[str, str]:
    width = 240
    source_rng = random.Random(0)
    original = bytearray(source_rng.randrange(256) for _ in range(width * width * 3))
    variant = bytearray(original)
    variant_rng = random.Random(12)
    for _ in range(100):
        index = variant_rng.randrange(len(variant))
        variant[index] = min(255, max(0, variant[index] + variant_rng.choice((-1, 1))))

    encoded = []
    for pixels in (original, variant):
        buffer = io.BytesIO()
        Image.frombytes("RGB", (width, width), bytes(pixels)).save(buffer, format="PNG")
        encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return encoded[0], encoded[1]


def _decoded_phash(encoded: str) -> str:
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        image.load()
        return str(imagehash.phash(image.convert("RGB")))


def _normalized_jpeg_phash(encoded: str) -> str:
    jpeg = io.BytesIO()
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        image.load()
        image.convert("RGB").save(jpeg, format="JPEG", quality=90, optimize=True)
    jpeg.seek(0)
    with Image.open(jpeg) as normalized:
        normalized.load()
        return str(imagehash.phash(normalized.convert("RGB")))


def _write_normalized_encoded_image(encoded: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        image.load()
        image.convert("RGB").save(destination, format="JPEG", quality=90, optimize=True)


def _mep_row(image_path="product_images/mep3m/mep3m-34-1.jpg") -> dict:
    return {
        "product_id": "mep3m-34-1",
        "title": "MEP product",
        "category_l1": "clothing",
        "category_l2": "dress",
        "category_l3": "evening dress",
        "ocr_text": "SALE",
        "image_path": image_path,
        "source": "mep3m",
    }


def _muge_row(image_path="product_images/muge/muge-old.jpg") -> dict:
    return {
        "product_id": "muge-old",
        "title": "old MUGE product",
        "category_l1": "unknown",
        "category_l2": None,
        "category_l3": None,
        "ocr_text": None,
        "image_path": image_path,
        "source": "muge",
    }


def _write_products(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("product_id", pa.string()),
            ("title", pa.string()),
            ("category_l1", pa.string()),
            ("category_l2", pa.string()),
            ("category_l3", pa.string()),
            ("ocr_text", pa.string()),
            ("image_path", pa.string()),
            ("source", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_read_text_records_reports_actual_muge_fields(tmp_path):
    source = tmp_path / "train_texts.jsonl"
    source.write_text(
        json.dumps(
            {"text_id": 7, "text": "红色连衣裙", "image_ids": [11, 12]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = list(read_text_records(source))

    assert records == [{"text_id": 7, "text": "红色连衣裙", "image_ids": [11, 12]}]


def test_clean_dataset_filters_invalid_small_and_phash_duplicates(tmp_path):
    image_tsv = tmp_path / "train_imgs.tsv"
    image_tsv.write_text(
        "11\t" + _encoded_image() + "\n"
        "12\t" + _encoded_image(image_format="JPEG") + "\n"
        "13\t" + _encoded_image(size=(100, 240), color=(20, 220, 20)) + "\n"
        "14\tnot-base64\n"
        "15\t" + _encoded_image(color=(20, 20, 220), pattern="top") + "\n",
        encoding="utf-8",
    )
    titles = {11: "红裙", 12: "红裙重复图", 13: "太小", 14: "损坏", 15: "蓝裙"}

    report = clean_dataset(
        image_tsv=image_tsv,
        titles=titles,
        output_dir=tmp_path / "images",
        parquet_path=tmp_path / "products.parquet",
        min_side=200,
    )

    assert report.kept == 2
    assert report.duplicates == 1
    assert report.too_small == 1
    assert report.damaged == 1
    rows = pq.read_table(tmp_path / "products.parquet").to_pylist()
    assert [row["product_id"] for row in rows] == ["muge-11", "muge-15"]
    assert all(row["source"] == "muge" for row in rows)
    assert all(row["category_l1"] == "unknown" for row in rows)
    assert all(row["category_l2"] is None for row in rows)
    assert all(row["category_l3"] is None for row in rows)
    assert all(row["ocr_text"] is None for row in rows)
    assert all((tmp_path / row["image_path"]).suffix == ".jpg" for row in rows)
    assert all((tmp_path / row["image_path"]).is_file() for row in rows)


def test_clean_dataset_deduplicates_final_jpeg_phash_and_backfills(tmp_path):
    first, collision = _encoded_preencode_distinct_postencode_collision()
    assert _decoded_phash(first) != _decoded_phash(collision)
    assert _normalized_jpeg_phash(first) == _normalized_jpeg_phash(collision)
    image_tsv = tmp_path / "train_imgs.tsv"
    image_tsv.write_text(
        f"11\t{first}\n12\t{collision}\n13\t{_encoded_image(pattern='top')}\n",
        encoding="utf-8",
    )

    report = clean_dataset(
        image_tsv=image_tsv,
        titles={11: "first", 12: "collision", 13: "backfill"},
        output_dir=tmp_path / "images",
        parquet_path=tmp_path / "products.parquet",
    )

    assert report.kept == 2
    assert report.duplicates == 1
    assert pq.read_table(tmp_path / "products.parquet").column(
        "product_id"
    ).to_pylist() == ["muge-11", "muge-13"]
    assert not (tmp_path / "images" / "muge-12.jpg").exists()


def test_clean_dataset_preserves_mep_rows_and_is_idempotent(tmp_path):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    output_dir = clean_dir / "product_images" / "muge"
    mep_image = clean_dir / "product_images" / "mep3m" / "mep3m-34-1.jpg"
    _write_normalized_encoded_image(_encoded_image(pattern="left"), mep_image)
    _write_products(parquet_path, [_mep_row()])
    image_tsv = tmp_path / "train_imgs.tsv"
    image_tsv.write_text(f"11\t{_encoded_image(pattern='top')}\n", encoding="utf-8")
    kwargs = {
        "image_tsv": image_tsv,
        "titles": {11: "new MUGE product"},
        "output_dir": output_dir,
        "parquet_path": parquet_path,
    }

    clean_dataset(**kwargs)
    first_rows = pq.read_table(parquet_path).to_pylist()
    clean_dataset(**kwargs)
    second_rows = pq.read_table(parquet_path).to_pylist()

    assert [row["product_id"] for row in second_rows] == [
        "muge-11",
        "mep3m-34-1",
    ]
    assert second_rows == first_rows
    assert second_rows[1] == _mep_row()
    hashes = {
        str(imagehash.phash(Image.open(clean_dir / row["image_path"])))
        for row in second_rows
    }
    assert len(hashes) == 2


def test_clean_dataset_deduplicates_against_retained_mep_and_backfills(tmp_path):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    output_dir = clean_dir / "product_images" / "muge"
    collision = _encoded_image(pattern="left")
    mep_image = clean_dir / "product_images" / "mep3m" / "mep3m-34-1.jpg"
    _write_normalized_encoded_image(collision, mep_image)
    _write_products(parquet_path, [_mep_row()])
    image_tsv = tmp_path / "train_imgs.tsv"
    image_tsv.write_text(
        f"11\t{collision}\n12\t{_encoded_image(pattern='top')}\n",
        encoding="utf-8",
    )

    report = clean_dataset(
        image_tsv=image_tsv,
        titles={11: "cross-source collision", 12: "backfill"},
        output_dir=output_dir,
        parquet_path=parquet_path,
    )

    rows = pq.read_table(parquet_path).to_pylist()
    assert report.duplicates == 1
    assert [row["product_id"] for row in rows] == ["muge-12", "mep3m-34-1"]
    hashes = {
        str(imagehash.phash(Image.open(clean_dir / row["image_path"]))) for row in rows
    }
    assert len(hashes) == 2


@pytest.mark.parametrize(
    "existing_image", [None, b"not-an-image"], ids=["missing", "damaged"]
)
def test_clean_dataset_fails_before_staging_for_unreadable_retained_image(
    tmp_path, monkeypatch, existing_image
):
    clean_dir = tmp_path / "clean"
    parquet_path = clean_dir / "products.parquet"
    mep_image = clean_dir / "product_images" / "mep3m" / "mep3m-34-1.jpg"
    if existing_image is not None:
        mep_image.parent.mkdir(parents=True)
        mep_image.write_bytes(existing_image)
    _write_products(parquet_path, [_mep_row()])
    staging_calls = []

    def unexpected_staging(*args, **kwargs):
        staging_calls.append((args, kwargs))
        raise AssertionError("staging must not start")

    monkeypatch.setattr(
        "skillchain.data.muge.prepare_staging_directory", unexpected_staging
    )

    with pytest.raises(RuntimeError, match="retained product image"):
        clean_dataset(
            image_tsv=tmp_path / "unused.tsv",
            titles={},
            output_dir=clean_dir / "product_images" / "muge",
            parquet_path=parquet_path,
        )

    assert staging_calls == []


def test_clean_dataset_recovers_pair_before_staging_when_later_write_fails(
    tmp_path, monkeypatch
):
    clean_dir = tmp_path / "clean"
    output_dir = clean_dir / "product_images" / "muge"
    parquet_path = clean_dir / "products.parquet"
    directory_backup = output_dir.with_name(f".{output_dir.name}.backup")
    file_backup = parquet_path.with_name(f".{parquet_path.name}.backup")
    output_dir.mkdir(parents=True)
    (output_dir / "new.jpg").write_bytes(b"new-image")
    directory_backup.mkdir(parents=True)
    (directory_backup / "old.jpg").write_bytes(b"old-image")
    _write_products(file_backup, [_muge_row()])
    old_parquet = file_backup.read_bytes()
    image_tsv = tmp_path / "train_imgs.tsv"
    image_tsv.write_text(f"11\t{_encoded_image()}\n", encoding="utf-8")

    def fail_after_writing(table, where, **kwargs):
        Path(where).write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr("skillchain.data.muge.pq.write_table", fail_after_writing)

    with pytest.raises(RuntimeError, match="simulated"):
        clean_dataset(
            image_tsv=image_tsv,
            titles={11: "new MUGE product"},
            output_dir=output_dir,
            parquet_path=parquet_path,
        )

    assert (output_dir / "old.jpg").read_bytes() == b"old-image"
    assert not (output_dir / "new.jpg").exists()
    assert parquet_path.read_bytes() == old_parquet
    assert not directory_backup.exists()
    assert not file_backup.exists()


def test_write_preview_creates_three_by_three_contact_sheet(tmp_path):
    paths = []
    for index in range(9):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (240, 240), (index * 20, 40, 80)).save(path)
        paths.append(path)

    output = tmp_path / "preview.jpg"
    write_preview(paths, output, tile_size=100)

    with Image.open(output) as preview:
        assert preview.size == (300, 300)


def test_clean_dataset_failure_preserves_previous_parquet(tmp_path, monkeypatch):
    image_tsv = tmp_path / "train_imgs.tsv"
    image_tsv.write_text("11\t" + _encoded_image() + "\n", encoding="utf-8")
    parquet_path = tmp_path / "products.parquet"
    images = tmp_path / "images"
    images.mkdir()
    (images / "muge-old.jpg").write_bytes(b"old-image")
    _write_products(parquet_path, [_muge_row("images/muge-old.jpg")])
    previous_parquet = parquet_path.read_bytes()

    def fail_after_writing(table, where, **kwargs):
        Path(where).write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr("skillchain.data.muge.pq.write_table", fail_after_writing)

    with pytest.raises(RuntimeError, match="simulated"):
        clean_dataset(
            image_tsv=image_tsv,
            titles={11: "红裙"},
            output_dir=images,
            parquet_path=parquet_path,
        )

    assert parquet_path.read_bytes() == previous_parquet
    assert (images / "muge-old.jpg").read_bytes() == b"old-image"
    assert not (images / "muge-11.jpg").exists()


def _formal_sources(tmp_path: Path) -> tuple[Path, Path, dict[int, str]]:
    texts = tmp_path / "train_texts.jsonl"
    texts.write_text(
        json.dumps(
            {"text_id": 7, "text": "red dress", "image_ids": [11]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    images = tmp_path / "train_imgs.tsv"
    images.write_text(f"11\t{_encoded_image()}\n", encoding="ascii")
    return texts, images, {11: "red dress"}


_LICENSE_EVIDENCE = b"fixture license evidence revision 0123456789abcdef\n"


def _source_lock(texts: Path, images: Path, **updates) -> MUGESourceLock:
    values = {
        "source_revision": "fixture-commit-0123456789abcdef",
        "texts_sha256": hashlib.sha256(texts.read_bytes()).hexdigest(),
        "images_sha256": hashlib.sha256(images.read_bytes()).hexdigest(),
        "license_id": "test-fixture-license",
        "license_evidence_sha256": hashlib.sha256(_LICENSE_EVIDENCE).hexdigest(),
        "license_evidence_url": ("https://example.test/muge/fixture-commit/LICENSE"),
        "source_url": "https://example.test/muge/fixture-commit",
        "attribution": "MUGE fixture authors",
        "cloud_upload_allowed": False,
        "public_demo_allowed": False,
    }
    values.update(updates)
    return MUGESourceLock(**values)


def _lock_args(tmp_path: Path, lock: MUGESourceLock) -> dict[str, object]:
    path = tmp_path / "muge-source-lock.json"
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
    license_path = tmp_path / "muge-license-evidence.txt"
    license_path.write_bytes(_LICENSE_EVIDENCE)
    review = MUGELicenseReview(
        source_lock_sha256=hashlib.sha256(content).hexdigest(),
        source_revision=lock.source_revision,
        license_id=lock.license_id,
        license_evidence_sha256=lock.license_evidence_sha256,
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
    review_path = tmp_path / "muge-human-license-review.json"
    review_path.write_bytes(review_content)
    return {
        "source_lock_path": path,
        "expected_source_lock_sha256": hashlib.sha256(content).hexdigest(),
        "license_evidence_path": license_path,
        "license_review_path": review_path,
        "expected_license_review_sha256": hashlib.sha256(review_content).hexdigest(),
    }


def _canonical_json(value: dict) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _candidate_universe(texts: Path, count: int) -> tuple[tuple[int, ...], str]:
    titles: dict[int, str] = {}
    for line in texts.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for image_id in row["image_ids"]:
            titles.setdefault(int(image_id), row["text"].strip())
            if len(titles) >= count:
                break
        if len(titles) >= count:
            break
    ids = tuple(sorted(titles))
    content = b"".join(
        _canonical_json(
            {
                "product_id": f"muge-{image_id}",
                "source_record_id": f"image:{image_id}",
                "title_sha256": hashlib.sha256(
                    titles[image_id].encode("utf-8")
                ).hexdigest(),
            }
        )
        for image_id in ids
    )
    return ids, hashlib.sha256(content).hexdigest()


def _formal_args(
    tmp_path: Path,
    lock: MUGESourceLock,
    texts: Path,
    images: Path,
    *,
    candidate_count: int = 1,
    minimum_image_side: int = 200,
    retained_gallery_parquet_path: Path | None = None,
) -> dict[str, object]:
    arguments = _lock_args(tmp_path, lock)
    _, universe_sha256 = _candidate_universe(texts, candidate_count)
    retained_gallery_parquet_path = (
        retained_gallery_parquet_path
        if retained_gallery_parquet_path is not None
        else tmp_path / "clean" / "products.parquet"
    )
    retained_count, retained_digest = compute_muge_retained_gallery_binding(
        retained_gallery_parquet_path
    )
    plan = MUGESelectionPlan(
        source_lock_sha256=arguments["expected_source_lock_sha256"],
        license_review_sha256=arguments["expected_license_review_sha256"],
        source_revision=lock.source_revision,
        source_texts_sha256=lock.texts_sha256,
        source_images_sha256=lock.images_sha256,
        candidate_count=candidate_count,
        candidate_universe_sha256=universe_sha256,
        retained_gallery_row_count=retained_count,
        retained_gallery_manifest_sha256=retained_digest,
        minimum_image_side=minimum_image_side,
        owner_kind="human",
        owner_id="fixture-selection-owner",
        preregistered_at="2026-07-21T01:03:00Z",
    )
    content = _canonical_json(plan.model_dump(mode="json"))
    path = tmp_path / "muge-selection-plan.json"
    path.write_bytes(content)
    return {
        **arguments,
        "selection_plan_path": path,
        "expected_selection_plan_sha256": hashlib.sha256(content).hexdigest(),
    }


def _verified_selection(
    tmp_path: Path,
    lock: MUGESourceLock,
    texts: Path,
    images: Path,
    *,
    accepted_image_ids: set[int],
    rejected_reasons: dict[int, str] | None = None,
    candidate_count: int = 1,
    retained_gallery_parquet_path: Path | None = None,
):
    arguments = _formal_args(
        tmp_path,
        lock,
        texts,
        images,
        candidate_count=candidate_count,
        retained_gallery_parquet_path=retained_gallery_parquet_path,
    )
    source = load_verified_muge_source_lock(
        arguments["source_lock_path"],
        expected_lock_sha256=arguments["expected_source_lock_sha256"],
    )
    review = load_verified_muge_license_review(
        arguments["license_review_path"],
        expected_review_sha256=arguments["expected_license_review_sha256"],
        source_lock=source,
        license_evidence_path=arguments["license_evidence_path"],
    )
    plan = load_verified_muge_selection_plan(
        arguments["selection_plan_path"],
        expected_plan_sha256=arguments["expected_selection_plan_sha256"],
        text_source_path=texts,
        image_source_path=images,
        retained_gallery_parquet_path=(
            retained_gallery_parquet_path
            if retained_gallery_parquet_path is not None
            else tmp_path / "clean" / "products.parquet"
        ),
        source_lock=source,
        license_review=review,
    )
    rejected_reasons = rejected_reasons or {}
    dispositions = tuple(
        MUGECandidateDisposition(
            source_record_id=f"image:{image_id}",
            product_id=f"muge-{image_id}",
            disposition="accepted" if image_id in accepted_image_ids else "rejected",
            reason=(
                "accepted_by_preregistered_policy"
                if image_id in accepted_image_ids
                else rejected_reasons.get(image_id, "missing_source_image")
            ),
        )
        for image_id in plan.candidate_image_ids
    )
    ledger = MUGEDispositionLedger(
        selection_plan_sha256=plan.plan_sha256,
        source_lock_sha256=source.lock_sha256,
        license_review_sha256=review.review_sha256,
        source_revision=lock.source_revision,
        candidate_universe_sha256=plan.plan.candidate_universe_sha256,
        candidate_count=len(dispositions),
        accepted_count=sum(row.disposition == "accepted" for row in dispositions),
        rejected_count=sum(row.disposition == "rejected" for row in dispositions),
        dispositions=dispositions,
        reviewer_kind="human",
        reviewer_id="fixture-disposition-reviewer",
        reviewed_at="2026-07-21T01:04:00Z",
    )
    content = _canonical_json(ledger.model_dump(mode="json"))
    ledger_path = tmp_path / "muge-disposition-ledger.json"
    ledger_path.write_bytes(content)
    return load_verified_muge_selection(
        ledger_path,
        expected_ledger_sha256=hashlib.sha256(content).hexdigest(),
        selection_plan=plan,
    )


def _export_args(
    tmp_path: Path,
    lock: MUGESourceLock,
    product_root: Path,
    *,
    accepted_image_ids: set[int] | None = None,
) -> dict[str, object]:
    sidecar = product_root / "muge-provenance.jsonl"
    texts = tmp_path / "train_texts.jsonl"
    images = tmp_path / "train_imgs.tsv"
    parquet_path = product_root.parent.parent / "products.parquet"
    return {
        "verified_selection": _verified_selection(
            tmp_path,
            lock,
            texts,
            images,
            accepted_image_ids=accepted_image_ids or {11},
            retained_gallery_parquet_path=parquet_path,
        ),
        "expected_provenance_sha256": (
            hashlib.sha256(sidecar.read_bytes()).hexdigest()
            if sidecar.is_file()
            else "0" * 64
        ),
    }


def test_formal_clean_exports_strict_locked_dataset_asset_drafts(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    lock = _source_lock(texts, images)

    clean_dataset(
        image_tsv=images,
        text_source=texts,
        titles=titles,
        **_formal_args(tmp_path, lock, texts, images),
        output_dir=product_root,
        parquet_path=parquet_path,
    )
    output = tmp_path / "muge-drafts.jsonl"
    drafts = export_dataset_asset_drafts(
        parquet_path=parquet_path,
        asset_root=clean_root,
        product_image_root=product_root,
        output_path=output,
        **_export_args(tmp_path, lock, product_root),
    )

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.source_dataset == "muge"
    assert draft.source_revision == lock.source_revision
    assert draft.source_record_id == "image:11"
    assert draft.transform_policy_version == "muge-exif-rgb-jpeg-q90-optimize-v1"
    assert draft.derivation_parent_asset_ids == []
    assert draft.license_id == lock.license_id
    assert draft.cloud_upload_allowed is False
    assert draft.public_demo_allowed is False
    assert draft.local_path == "product_images/muge/muge-11.jpg"
    provenance = json.loads(
        (product_root / "muge-provenance.jsonl").read_text(encoding="utf-8")
    )
    assert provenance["license_evidence_sha256"] == lock.license_evidence_sha256
    assert provenance["selection_plan_sha256"]
    assert provenance["retained_gallery_manifest_sha256"]
    assert provenance["retained_gallery_row_count"] == 0
    assert (
        provenance["source_image_sha256"]
        == hashlib.sha256(base64.b64decode(_encoded_image())).hexdigest()
    )
    assert (
        provenance["output_asset_sha256"]
        == hashlib.sha256((product_root / "muge-11.jpg").read_bytes()).hexdigest()
    )
    assert json.loads(output.read_text(encoding="utf-8")) == draft.model_dump(
        mode="json"
    )
    verified_drafts = load_verified_muge_draft_bundle(
        output,
        expected_bundle_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        parquet_path=parquet_path,
        asset_root=clean_root,
        product_image_root=product_root,
        **_export_args(tmp_path, lock, product_root),
    )
    assert verified_drafts.drafts == drafts
    assert verified_drafts.selection.ledger.accepted_count == 1


def test_formal_clean_rejects_retained_gallery_changed_after_preregistration(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    retained_image = clean_root / "product_images" / "mep3m" / "mep3m-34-1.jpg"
    _write_normalized_encoded_image(_encoded_image(pattern="top"), retained_image)
    _write_products(parquet_path, [_mep_row()])
    lock = _source_lock(texts, images)
    formal_arguments = _formal_args(
        tmp_path,
        lock,
        texts,
        images,
        retained_gallery_parquet_path=parquet_path,
    )

    _write_normalized_encoded_image(_encoded_image(pattern="left"), retained_image)
    with pytest.raises(MUGEProvenanceError, match="retained gallery differs"):
        clean_dataset(
            image_tsv=images,
            text_source=texts,
            titles=titles,
            **formal_arguments,
            output_dir=product_root,
            parquet_path=parquet_path,
        )
    assert not product_root.exists()


def test_formal_export_and_load_revalidate_every_retained_gallery_dependency(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    retained_image = clean_root / "product_images" / "mep3m" / "mep3m-34-1.jpg"
    _write_normalized_encoded_image(_encoded_image(pattern="top"), retained_image)
    _write_products(parquet_path, [_mep_row()])
    lock = _source_lock(texts, images)
    clean_dataset(
        image_tsv=images,
        text_source=texts,
        titles=titles,
        **_formal_args(
            tmp_path,
            lock,
            texts,
            images,
            retained_gallery_parquet_path=parquet_path,
        ),
        output_dir=product_root,
        parquet_path=parquet_path,
    )
    selection = _verified_selection(
        tmp_path,
        lock,
        texts,
        images,
        accepted_image_ids={11},
        retained_gallery_parquet_path=parquet_path,
    )
    sidecar = product_root / "muge-provenance.jsonl"
    provenance_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    output = tmp_path / "retained-gallery-drafts.jsonl"
    export_dataset_asset_drafts(
        parquet_path=parquet_path,
        asset_root=clean_root,
        product_image_root=product_root,
        output_path=output,
        expected_provenance_sha256=provenance_sha256,
        verified_selection=selection,
    )

    retained_bytes = retained_image.read_bytes()
    _write_normalized_encoded_image(_encoded_image(pattern="left"), retained_image)
    with pytest.raises(MUGEProvenanceError, match="retained gallery differs"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "tampered-retained-image-drafts.jsonl",
            expected_provenance_sha256=provenance_sha256,
            verified_selection=selection,
        )

    retained_image.write_bytes(retained_bytes)
    rows = pq.read_table(parquet_path).to_pylist()
    for row in rows:
        if row["source"] != "muge":
            row["title"] = "coordinated retained-row rewrite"
    _write_products(parquet_path, rows)
    with pytest.raises(MUGEProvenanceError, match="retained gallery differs"):
        load_verified_muge_draft_bundle(
            output,
            expected_bundle_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            expected_provenance_sha256=provenance_sha256,
            verified_selection=selection,
        )


def test_formal_clean_requires_a_preregistered_plan_before_output(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)

    with pytest.raises(MUGEProvenanceError, match="selection_plan_path"):
        clean_dataset(
            image_tsv=images,
            text_source=texts,
            titles=titles,
            **_lock_args(tmp_path, lock),
            output_dir=tmp_path / "clean" / "product_images" / "muge",
            parquet_path=tmp_path / "clean" / "products.parquet",
        )

    assert not (tmp_path / "clean").exists()


def test_selection_plan_and_ledger_reject_llm_reviewers(tmp_path):
    texts, images, _ = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)
    arguments = _formal_args(tmp_path, lock, texts, images)
    plan_raw = json.loads(
        Path(arguments["selection_plan_path"]).read_text(encoding="utf-8")
    )
    plan_raw["owner_id"] = "human-gpt-reviewer"
    with pytest.raises(ValidationError, match="accountable human"):
        MUGESelectionPlan.model_validate(plan_raw, strict=True)

    license_raw = json.loads(
        Path(arguments["license_review_path"]).read_text(encoding="utf-8")
    )
    license_raw["reviewer_id"] = "human-gpt-reviewer"
    with pytest.raises(ValidationError, match="accountable human"):
        MUGELicenseReview.model_validate(license_raw, strict=True)

    selection = _verified_selection(
        tmp_path,
        lock,
        texts,
        images,
        accepted_image_ids={11},
    )
    ledger_raw = selection.ledger.model_dump(mode="json")
    ledger_raw["reviewer_id"] = "claude-reviewer"
    with pytest.raises(ValidationError, match="accountable human"):
        MUGEDispositionLedger.model_validate(ledger_raw, strict=True)


def test_selection_ledger_fails_closed_for_missing_extra_duplicate_and_unsorted_rows(
    tmp_path,
):
    texts = tmp_path / "train_texts.jsonl"
    texts.write_text(
        json.dumps(
            {"text_id": 7, "text": "red dress", "image_ids": [2, 10]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    images = tmp_path / "train_imgs.tsv"
    images.write_text(
        f"2\t{_encoded_image(pattern='left')}\n"
        f"10\t{_encoded_image(color=(10, 180, 30), pattern='top')}\n",
        encoding="ascii",
    )
    lock = _source_lock(texts, images)
    good = _verified_selection(
        tmp_path,
        lock,
        texts,
        images,
        accepted_image_ids={2, 10},
        candidate_count=2,
    )

    def load_rewritten(name: str, raw: dict):
        path = tmp_path / f"{name}.json"
        content = _canonical_json(raw)
        path.write_bytes(content)
        return load_verified_muge_selection(
            path,
            expected_ledger_sha256=hashlib.sha256(content).hexdigest(),
            selection_plan=good.plan,
        )

    missing = good.ledger.model_dump(mode="json")
    missing["dispositions"] = missing["dispositions"][:1]
    missing["candidate_count"] = 1
    missing["accepted_count"] = 1
    missing["rejected_count"] = 0
    with pytest.raises(MUGEProvenanceError, match="not bound"):
        load_rewritten("missing", missing)

    extra = good.ledger.model_dump(mode="json")
    extra["dispositions"].append(
        {
            "source_record_id": "image:999",
            "product_id": "muge-999",
            "disposition": "accepted",
            "reason": "accepted_by_preregistered_policy",
        }
    )
    extra["candidate_count"] = 3
    extra["accepted_count"] = 3
    with pytest.raises(MUGEProvenanceError, match="not bound"):
        load_rewritten("extra", extra)

    duplicate = good.ledger.model_dump(mode="json")
    duplicate["dispositions"][1] = duplicate["dispositions"][0]
    with pytest.raises(MUGEProvenanceError, match="strict schema"):
        load_rewritten("duplicate", duplicate)

    unsorted = good.ledger.model_dump(mode="json")
    unsorted["dispositions"].reverse()
    with pytest.raises(MUGEProvenanceError, match="strict schema"):
        load_rewritten("unsorted", unsorted)


def test_coordinated_selection_rehash_cannot_break_source_or_review_binding(tmp_path):
    texts, images, _ = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)
    selection = _verified_selection(
        tmp_path,
        lock,
        texts,
        images,
        accepted_image_ids={11},
    )
    forged_plan = selection.plan.plan.model_dump(mode="json")
    forged_plan["source_lock_sha256"] = "f" * 64
    plan_content = _canonical_json(forged_plan)
    plan_path = tmp_path / "coordinated-selection-plan.json"
    plan_path.write_bytes(plan_content)

    with pytest.raises(MUGEProvenanceError, match="not bound"):
        load_verified_muge_selection_plan(
            plan_path,
            expected_plan_sha256=hashlib.sha256(plan_content).hexdigest(),
            text_source_path=texts,
            image_source_path=images,
            retained_gallery_parquet_path=tmp_path / "clean" / "products.parquet",
            source_lock=selection.plan.source_lock,
            license_review=selection.plan.license_review,
        )

    forged_ledger = selection.ledger.model_dump(mode="json")
    forged_ledger["source_lock_sha256"] = "f" * 64
    ledger_content = _canonical_json(forged_ledger)
    ledger_path = tmp_path / "coordinated-disposition-ledger.json"
    ledger_path.write_bytes(ledger_content)
    with pytest.raises(MUGEProvenanceError, match="not bound"):
        load_verified_muge_selection(
            ledger_path,
            expected_ledger_sha256=hashlib.sha256(ledger_content).hexdigest(),
            selection_plan=selection.plan,
        )


def test_disposition_reasons_are_recomputed_from_every_locked_candidate(tmp_path):
    texts = tmp_path / "train_texts.jsonl"
    texts.write_text(
        json.dumps(
            {
                "text_id": 7,
                "text": "red dress",
                "image_ids": [11, 12, 13, 14, 15],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    valid = _encoded_image()
    images = tmp_path / "train_imgs.tsv"
    images.write_text(
        f"11\t{valid}\n"
        "12\tnot-base64\n"
        f"13\t{_encoded_image(size=(100, 100))}\n"
        f"14\t{valid}\n",
        encoding="ascii",
    )
    titles = {image_id: "red dress" for image_id in (11, 12, 13, 14, 15)}
    lock = _source_lock(texts, images)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    report = clean_dataset(
        image_tsv=images,
        text_source=texts,
        titles=titles,
        **_formal_args(
            tmp_path,
            lock,
            texts,
            images,
            candidate_count=5,
        ),
        output_dir=product_root,
        parquet_path=parquet_path,
    )
    assert (report.kept, report.damaged, report.too_small, report.duplicates) == (
        1,
        1,
        1,
        1,
    )
    reasons = {
        12: "invalid_source_image",
        13: "below_preregistered_minimum_side",
        14: "normalized_perceptual_duplicate",
        15: "missing_source_image",
    }
    selection = _verified_selection(
        tmp_path,
        lock,
        texts,
        images,
        accepted_image_ids={11},
        rejected_reasons=reasons,
        candidate_count=5,
    )
    sidecar = product_root / "muge-provenance.jsonl"
    output = tmp_path / "five-candidate-drafts.jsonl"
    drafts = export_dataset_asset_drafts(
        parquet_path=parquet_path,
        asset_root=clean_root,
        product_image_root=product_root,
        output_path=output,
        expected_provenance_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        verified_selection=selection,
    )
    assert [draft.product_id for draft in drafts] == ["muge-11"]

    wrong_rows = list(selection.ledger.dispositions)
    wrong_rows[1] = wrong_rows[1].model_copy(update={"reason": "missing_source_image"})
    wrong_ledger = MUGEDispositionLedger.model_validate(
        {
            **selection.ledger.model_dump(mode="python"),
            "dispositions": tuple(wrong_rows),
        },
        strict=True,
    )
    wrong_content = _canonical_json(wrong_ledger.model_dump(mode="json"))
    wrong_path = tmp_path / "wrong-reason-ledger.json"
    wrong_path.write_bytes(wrong_content)
    wrong_selection = load_verified_muge_selection(
        wrong_path,
        expected_ledger_sha256=hashlib.sha256(wrong_content).hexdigest(),
        selection_plan=selection.plan,
    )
    with pytest.raises(MUGEProvenanceError, match="deterministic source-bound"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "wrong-reason-drafts.jsonl",
            expected_provenance_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            verified_selection=wrong_selection,
        )


def test_formal_export_rejects_a_constructed_selection_handle(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    clean_dataset(
        image_tsv=images,
        text_source=texts,
        titles=titles,
        **_formal_args(tmp_path, lock, texts, images),
        output_dir=product_root,
        parquet_path=parquet_path,
    )
    selection = _verified_selection(
        tmp_path,
        lock,
        texts,
        images,
        accepted_image_ids={11},
    )
    sidecar = product_root / "muge-provenance.jsonl"
    with pytest.raises(MUGEProvenanceError, match="verified selection handle"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "constructed-selection.jsonl",
            expected_provenance_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            verified_selection=replace(selection, _verification_token=None),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_revision", "main", "immutable revision"),
        ("source_revision", "unknown", "immutable revision"),
        ("source_revision", "", "immutable revision"),
        ("license_id", "unknown", "explicit and verified"),
        ("license_id", "", "explicit and verified"),
        (
            "source_url",
            "https://example.test/muge/resolve/main",
            "mutable revision",
        ),
    ],
)
def test_source_lock_rejects_mutable_or_unknown_provenance(
    tmp_path, field, value, message
):
    texts, images, _ = _formal_sources(tmp_path)
    with pytest.raises(ValidationError, match=message):
        _source_lock(texts, images, **{field: value})


def test_formal_lock_loader_rejects_missing_pin_tamper_and_noncanonical_bytes(tmp_path):
    texts, images, _ = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)
    args = _lock_args(tmp_path, lock)
    path = args["source_lock_path"]
    expected = args["expected_source_lock_sha256"]
    assert (
        load_verified_muge_source_lock(path, expected_lock_sha256=expected).lock == lock
    )

    with pytest.raises(MUGEProvenanceError, match="external expected"):
        load_verified_muge_source_lock(path, expected_lock_sha256="self-computed")

    changed = lock.model_copy(
        update={"attribution": "different reviewer-approved text"}
    )
    changed_content = (
        json.dumps(
            changed.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    path.write_bytes(changed_content)
    with pytest.raises(MUGEProvenanceError, match="external expected digest"):
        load_verified_muge_source_lock(path, expected_lock_sha256=expected)

    noncanonical = changed_content.rstrip() + b" \n"
    path.write_bytes(noncanonical)
    with pytest.raises(MUGEProvenanceError, match="canonical JSON"):
        load_verified_muge_source_lock(
            path, expected_lock_sha256=hashlib.sha256(noncanonical).hexdigest()
        )


def test_license_review_rejects_a_constructed_ungranted_source_handle(tmp_path):
    texts, images, _ = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)
    arguments = _lock_args(tmp_path, lock)
    verified = load_verified_muge_source_lock(
        arguments["source_lock_path"],
        expected_lock_sha256=arguments["expected_source_lock_sha256"],
    )
    with pytest.raises(MUGEProvenanceError, match="verified source lock"):
        load_verified_muge_license_review(
            arguments["license_review_path"],
            expected_review_sha256=arguments["expected_license_review_sha256"],
            source_lock=replace(verified, _verification_token=None),
            license_evidence_path=arguments["license_evidence_path"],
        )


def test_formal_lock_loader_rejects_duplicate_keys_and_symlink(tmp_path):
    texts, images, _ = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)
    args = _lock_args(tmp_path, lock)
    path = args["source_lock_path"]
    content = path.read_bytes()
    duplicate = b'{"schema_version":1,' + content[1:]
    path.write_bytes(duplicate)
    with pytest.raises(MUGEProvenanceError, match="strict schema"):
        load_verified_muge_source_lock(
            path, expected_lock_sha256=hashlib.sha256(duplicate).hexdigest()
        )

    path.write_bytes(content)
    link = tmp_path / "muge-lock-link.json"
    try:
        os.symlink(path, link)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(MUGEProvenanceError, match="regular non-symlink"):
        load_verified_muge_source_lock(
            link, expected_lock_sha256=hashlib.sha256(content).hexdigest()
        )


def test_formal_clean_rejects_lock_toctou_before_publish(tmp_path, monkeypatch):
    texts, images, titles = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)
    lock_args = _formal_args(tmp_path, lock, texts, images)
    lock_path = lock_args["source_lock_path"]
    original_verify = muge_module._verify_snapshot
    checks = 0

    def mutate_on_final_check(snapshot, label):
        nonlocal checks
        if label == "MUGE source lock":
            checks += 1
            if checks == 2:
                lock_path.write_bytes(lock_path.read_bytes() + b" ")
        return original_verify(snapshot, label)

    monkeypatch.setattr(muge_module, "_verify_snapshot", mutate_on_final_check)
    with pytest.raises(MUGEProvenanceError, match="changed during"):
        clean_dataset(
            image_tsv=images,
            text_source=texts,
            titles=titles,
            **lock_args,
            output_dir=tmp_path / "clean" / "product_images" / "muge",
            parquet_path=tmp_path / "clean" / "products.parquet",
        )


def test_in_process_source_lock_cannot_request_formal_clean(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    with pytest.raises(MUGEProvenanceError, match="in-process"):
        clean_dataset(
            image_tsv=images,
            text_source=texts,
            titles=titles,
            source_lock=_source_lock(texts, images),
            output_dir=tmp_path / "images",
            parquet_path=tmp_path / "products.parquet",
        )


def test_formal_clean_rejects_wrong_external_source_lock_before_publish(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    lock = _source_lock(texts, images, images_sha256="0" * 64)
    output_dir = tmp_path / "images"

    with pytest.raises(MUGEProvenanceError, match="external source lock"):
        clean_dataset(
            image_tsv=images,
            text_source=texts,
            titles=titles,
            **_formal_args(tmp_path, lock, texts, images),
            output_dir=output_dir,
            parquet_path=tmp_path / "products.parquet",
        )

    assert not output_dir.exists()


def test_formal_clean_reads_and_locks_license_evidence_bytes(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    lock = _source_lock(texts, images)
    arguments = _formal_args(tmp_path, lock, texts, images)
    Path(arguments["license_evidence_path"]).write_bytes(b"substituted license\n")

    with pytest.raises(MUGEProvenanceError, match="license evidence"):
        clean_dataset(
            image_tsv=images,
            text_source=texts,
            titles=titles,
            **arguments,
            output_dir=tmp_path / "images",
            parquet_path=tmp_path / "products.parquet",
        )


def test_formal_clean_rejects_empty_license_evidence_even_when_hash_is_locked(
    tmp_path,
):
    texts, images, titles = _formal_sources(tmp_path)
    lock = _source_lock(
        texts,
        images,
        license_evidence_sha256=hashlib.sha256(b"").hexdigest(),
    )
    arguments = _formal_args(tmp_path, lock, texts, images)
    Path(arguments["license_evidence_path"]).write_bytes(b"")

    with pytest.raises(MUGEProvenanceError, match="must not be empty"):
        clean_dataset(
            image_tsv=images,
            text_source=texts,
            titles=titles,
            **arguments,
            output_dir=tmp_path / "images",
            parquet_path=tmp_path / "products.parquet",
        )


def test_license_review_is_human_external_and_permissions_cannot_self_upgrade(
    tmp_path,
):
    texts, images, titles = _formal_sources(tmp_path)
    with pytest.raises(ValidationError):
        _source_lock(texts, images, cloud_upload_allowed=True)

    lock = _source_lock(texts, images)
    arguments = _formal_args(tmp_path, lock, texts, images)
    review_path = Path(arguments["license_review_path"])
    forged = json.loads(review_path.read_text(encoding="utf-8"))
    forged["source_lock_sha256"] = "f" * 64
    review_path.write_bytes(
        (json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )
    arguments["expected_license_review_sha256"] = hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest()
    with pytest.raises(MUGEProvenanceError, match="not bound"):
        clean_dataset(
            image_tsv=images,
            text_source=texts,
            titles=titles,
            **arguments,
            output_dir=tmp_path / "images",
            parquet_path=tmp_path / "products.parquet",
        )


def test_formal_export_revalidates_lock_and_rejects_coordinated_sidecar_rewrite(
    tmp_path,
):
    texts, images, titles = _formal_sources(tmp_path)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    lock = _source_lock(texts, images)
    clean_dataset(
        image_tsv=images,
        text_source=texts,
        titles=titles,
        **_formal_args(tmp_path, lock, texts, images),
        output_dir=product_root,
        parquet_path=parquet_path,
    )

    forged_lock = MUGESourceLock.model_construct(
        **{**lock.model_dump(mode="python"), "source_revision": "main"}
    )
    with pytest.raises(MUGEProvenanceError, match="strict schema"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "forged-lock.jsonl",
            **_export_args(tmp_path, forged_lock, product_root),
        )

    sidecar = product_root / "muge-provenance.jsonl"
    expected_provenance_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    rewritten = json.loads(sidecar.read_text(encoding="utf-8"))
    rewritten["source_revision"] = "attacker-revision"
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
    with pytest.raises(MUGEProvenanceError, match="external expected digest"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "rewritten-sidecar.jsonl",
            **{
                **_export_args(tmp_path, lock, product_root),
                "expected_provenance_sha256": expected_provenance_sha256,
            },
        )


def test_draft_coordinated_rehash_cannot_invent_a_source(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    lock = _source_lock(texts, images)
    clean_dataset(
        image_tsv=images,
        text_source=texts,
        titles=titles,
        **_formal_args(tmp_path, lock, texts, images),
        output_dir=product_root,
        parquet_path=parquet_path,
    )
    output = tmp_path / "muge-drafts.jsonl"
    export_args = _export_args(tmp_path, lock, product_root)
    export_dataset_asset_drafts(
        parquet_path=parquet_path,
        asset_root=clean_root,
        product_image_root=product_root,
        output_path=output,
        **export_args,
    )
    forged = json.loads(output.read_text(encoding="utf-8"))
    forged["source_revision"] = "invented-source-revision"
    output.write_bytes(
        (json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )

    with pytest.raises(MUGEProvenanceError, match="source-bound recomputation"):
        load_verified_muge_draft_bundle(
            output,
            expected_bundle_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            **export_args,
        )


def test_legacy_clean_is_diagnostic_and_cannot_be_formally_exported(tmp_path):
    _, images, titles = _formal_sources(tmp_path)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    clean_dataset(
        image_tsv=images,
        titles=titles,
        output_dir=product_root,
        parquet_path=parquet_path,
    )
    marker = json.loads(
        (product_root / "muge-diagnostic.json").read_text(encoding="utf-8")
    )
    assert marker["eligible_for_formal_export"] is False

    texts = tmp_path / "train_texts.jsonl"
    texts.write_text(
        json.dumps({"text_id": 7, "text": "red dress", "image_ids": [11]}) + "\n",
        encoding="utf-8",
    )
    lock = _source_lock(texts, images)
    with pytest.raises(MUGEProvenanceError, match="diagnostic"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "drafts.jsonl",
            **_export_args(tmp_path, lock, product_root),
        )


def test_formal_export_rejects_modified_final_image_bytes(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    lock = _source_lock(texts, images)
    clean_dataset(
        image_tsv=images,
        text_source=texts,
        titles=titles,
        **_formal_args(tmp_path, lock, texts, images),
        output_dir=product_root,
        parquet_path=parquet_path,
    )
    (product_root / "muge-11.jpg").write_bytes(b"replacement")

    with pytest.raises(MUGEProvenanceError, match="final-byte digest"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=product_root,
            output_path=tmp_path / "drafts.jsonl",
            **_export_args(tmp_path, lock, product_root),
        )


def test_exact_query_materialization_is_marked_diagnostic_and_not_exportable(tmp_path):
    texts, images, titles = _formal_sources(tmp_path)
    clean_root = tmp_path / "clean"
    product_root = clean_root / "product_images" / "muge"
    parquet_path = clean_root / "products.parquet"
    lock = _source_lock(texts, images)
    report = clean_dataset(
        image_tsv=images,
        text_source=texts,
        titles=titles,
        **_formal_args(tmp_path, lock, texts, images),
        output_dir=product_root,
        parquet_path=parquet_path,
    )
    query_root = clean_root / "query_images" / "exact_match"
    assert materialize_query_images(report.image_paths, query_root) == 1
    marker = json.loads(
        (query_root / "muge-exact-query-diagnostic.json").read_text(encoding="utf-8")
    )
    assert marker["eligible_for_formal_export"] is False
    assert marker["eligible_as_independent_positive"] is False
    assert marker["relations"][0]["relation"] == "exact-byte-copy"

    with pytest.raises(MUGEProvenanceError, match="diagnostic-only"):
        export_dataset_asset_drafts(
            parquet_path=parquet_path,
            asset_root=clean_root,
            product_image_root=query_root,
            output_path=tmp_path / "query-drafts.jsonl",
            **_export_args(tmp_path, lock, query_root),
        )
