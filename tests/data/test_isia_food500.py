import io
import json
import hashlib
import zipfile

from PIL import Image, ImageDraw

from skillchain.data.isia_food500 import discover_local_records, materialize, sha256_file


def _image_bytes(index: int) -> bytes:
    image = Image.new("RGB", (320, 280), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((index * 20, 0, index * 20 + 40, 279), fill="black")
    draw.ellipse((80, index * 20, 220, index * 20 + 100), fill="red")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _partial_volume(path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        index = 1
        for category in ("Biryani", "Bibimbap"):
            for item in range(2):
                target.writestr(
                    f"ISIA_Food500/images/{category}/{category}_{item:04d}.jpg",
                    _image_bytes(index),
                )
                index += 1
    path.write_bytes(b"continuation-from-previous-volume" + archive.getvalue())


def test_discover_local_records_finds_complete_images_after_volume_continuation(tmp_path):
    archive = tmp_path / "ISIA_Food500.zip"
    _partial_volume(archive)

    records = discover_local_records(archive)

    assert len(records) == 4
    assert records[0].category == "Biryani"
    assert records[-1].category == "Bibimbap"


def test_materialize_balances_categories_and_writes_source_manifest(tmp_path):
    archive = tmp_path / "ISIA_Food500.zip"
    _partial_volume(archive)

    report = materialize(archive, tmp_path / "utility_food", limit=4)

    assert report.kept == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "utility_food" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["category"] for row in rows] == [
        "Bibimbap",
        "Biryani",
        "Bibimbap",
        "Biryani",
    ]
    assert {row["license"] for row in rows} == {"not-specified-by-publisher"}
    assert all(row["source_member"].endswith(".jpg") for row in rows)
    assert len(list((tmp_path / "utility_food").glob("isia-*.jpg"))) == 4


def test_discover_local_records_rejects_zip_bomb_ratio(tmp_path):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        target.writestr(
            "ISIA_Food500/images/Fake/Fake_0001.jpg",
            b"0" * 1_000_000,
        )

    assert discover_local_records(archive) == []


def test_sha256_file_returns_stable_digest(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()
