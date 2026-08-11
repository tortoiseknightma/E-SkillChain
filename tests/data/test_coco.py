import io
import json
import zipfile

from PIL import Image

import pytest

from skillchain.data.coco import extract_annotations, materialize, select_candidates


def _annotations():
    return {
        "images": [
            {"id": 10, "file_name": "000000000010.jpg", "width": 640, "height": 480},
            {"id": 20, "file_name": "000000000020.jpg", "width": 640, "height": 480},
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 27, "name": "handbag", "supercategory": "accessory"},
            {"id": 44, "name": "bottle", "supercategory": "kitchen"},
        ],
        "annotations": [
            {"image_id": 10, "category_id": 27, "iscrowd": 0, "bbox": [0, 0, 40, 40]},
            {"image_id": 10, "category_id": 44, "iscrowd": 0, "bbox": [0, 0, 30, 30]},
            {"image_id": 10, "category_id": 44, "iscrowd": 0, "bbox": [0, 0, 25, 25]},
            {"image_id": 10, "category_id": 44, "iscrowd": 1, "bbox": [0, 0, 100, 100]},
            {"image_id": 20, "category_id": 1, "iscrowd": 0, "bbox": [0, 0, 100, 100]},
            {"image_id": 20, "category_id": 27, "iscrowd": 0, "bbox": [0, 0, 10, 10]},
        ],
    }


def test_select_candidates_requires_three_visible_product_instances():
    selected = select_candidates(
        _annotations(), min_instances=3, min_bbox_side=20, limit=800
    )

    assert selected == [
        {
            "image_id": 10,
            "file_name": "000000000010.jpg",
            "width": 640,
            "height": 480,
            "instance_count": 3,
            "categories": {"bottle": 2, "handbag": 1},
        }
    ]


def test_materialize_reads_selected_images_from_zip_and_writes_manifest(tmp_path):
    archive = tmp_path / "val2017.zip"
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), "blue").save(buffer, format="JPEG")
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("val2017/000000000010.jpg", buffer.getvalue())
    destination = tmp_path / "multi_product"

    report = materialize(
        _annotations(), archive, destination, min_instances=3, min_bbox_side=20
    )

    assert report.kept == 1
    assert report.damaged == 0
    assert (destination / "coco-10.jpg").is_file()
    row = json.loads((destination / "manifest.jsonl").read_text(encoding="utf-8"))
    assert row["image"] == "coco-10.jpg"
    assert row["categories"] == {"bottle": 2, "handbag": 1}


def test_materialize_backfills_after_a_candidate_fails_image_quality(tmp_path):
    data = _annotations()
    data["annotations"].extend(
        [
            {"image_id": 20, "category_id": 44, "iscrowd": 0, "bbox": [0, 0, 30, 30]},
            {"image_id": 20, "category_id": 44, "iscrowd": 0, "bbox": [0, 0, 30, 30]},
            {"image_id": 20, "category_id": 44, "iscrowd": 0, "bbox": [0, 0, 30, 30]},
        ]
    )
    archive = tmp_path / "val2017.zip"
    with zipfile.ZipFile(archive, "w") as target:
        for image_id, size in ((10, (100, 100)), (20, (640, 480))):
            buffer = io.BytesIO()
            Image.new("RGB", size, "blue").save(buffer, format="JPEG")
            target.writestr(f"val2017/{image_id:012d}.jpg", buffer.getvalue())

    report = materialize(data, archive, tmp_path / "output", limit=1)

    assert report.kept == 1
    assert report.too_small == 1
    assert (tmp_path / "output" / "coco-20.jpg").is_file()


def test_extract_annotations_creates_clean_ready_path(tmp_path):
    archive = tmp_path / "annotations.zip"
    payload = json.dumps(_annotations()).encode("utf-8")
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("annotations/instances_val2017.json", payload)
    output = tmp_path / "extracted" / "annotations" / "instances_val2017.json"

    extract_annotations(archive, output)

    assert json.loads(output.read_text(encoding="utf-8"))["images"][0]["id"] == 10


def test_materialize_failure_preserves_previous_valid_outputs(tmp_path):
    destination = tmp_path / "multi_product"
    destination.mkdir()
    old_image = destination / "coco-999.jpg"
    old_image.write_bytes(b"old-image")
    manifest = destination / "manifest.jsonl"
    manifest.write_text('{"old":true}\n', encoding="utf-8")
    broken_archive = tmp_path / "broken.zip"
    broken_archive.write_bytes(b"not-a-zip")

    with pytest.raises(zipfile.BadZipFile):
        materialize(_annotations(), broken_archive, destination)

    assert old_image.read_bytes() == b"old-image"
    assert manifest.read_text(encoding="utf-8") == '{"old":true}\n'


def test_materialize_midstream_failure_does_not_publish_partial_image_set(tmp_path, monkeypatch):
    data = _annotations()
    data["annotations"].extend(
        [
            {"image_id": 20, "category_id": 44, "iscrowd": 0, "bbox": [0, 0, 30, 30]},
            {"image_id": 20, "category_id": 44, "iscrowd": 0, "bbox": [0, 0, 30, 30]},
            {"image_id": 20, "category_id": 44, "iscrowd": 0, "bbox": [0, 0, 30, 30]},
        ]
    )
    archive = tmp_path / "val2017.zip"
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), "blue").save(buffer, format="JPEG")
    second_buffer = io.BytesIO()
    second = Image.new("RGB", (640, 480), "red")
    for x in range(320):
        for y in range(480):
            second.putpixel((x, y), (0, 0, 0))
    second.save(second_buffer, format="JPEG")
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("val2017/000000000010.jpg", buffer.getvalue())
        target.writestr("val2017/000000000020.jpg", second_buffer.getvalue())
    destination = tmp_path / "multi_product"
    destination.mkdir()
    (destination / "coco-old.jpg").write_bytes(b"old-image")
    (destination / "manifest.jsonl").write_text('{"old":true}\n', encoding="utf-8")
    original_save = Image.Image.save
    calls = 0

    def fail_on_second_save(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated image write failure")
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_on_second_save)

    with pytest.raises(RuntimeError, match="simulated"):
        materialize(data, archive, destination, limit=2)

    assert (destination / "coco-old.jpg").read_bytes() == b"old-image"
    assert not (destination / "coco-10.jpg").exists()
    assert not (destination / "coco-20.jpg").exists()
