import json

import pytest
from PIL import Image

from skillchain.data.utility import _validate_provenance, compose_sources


def _source(root, name, rows):
    root.mkdir()
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(rows):
            filename = f"{name}-{index}.jpg"
            Image.new("RGB", (300, 300), (index * 40, 0, 0)).save(root / filename)
            manifest.write(json.dumps({"image": filename, **row}) + "\n")


def test_compose_sources_atomically_merges_food_and_documents(tmp_path):
    food = tmp_path / "food"
    docs = tmp_path / "docs"
    food_provenance = {
        "source": "isia-food500",
        "license": "not-specified-by-publisher",
        "source_page": "http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html",
        "source_member": "ISIA_Food500/images/Dish/item.jpg",
        "license_note": "publisher did not specify a license",
        "distribution": "local-only-not-redistributed",
    }
    _source(
        food,
        "food",
        [{"category": "dish", **food_provenance}, {"category": "meal", **food_provenance}],
    )
    _source(
        docs,
        "doc",
        [
            {
                "document_category": "receipt",
                "source": "wikimedia-commons",
                "license": "CC0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "description_url": "https://commons.wikimedia.org/wiki/File:Receipt.jpg",
                "original_url": "https://upload.wikimedia.org/receipt.jpg",
                "source_url": "https://upload.wikimedia.org/thumb/receipt.jpg",
                "attribution": "author",
            }
        ],
    )

    report = compose_sources(
        food,
        docs,
        tmp_path / "utility",
        food_count=2,
        document_count=1,
    )

    assert report.total == 3
    rows = [
        json.loads(line)
        for line in (tmp_path / "utility" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["utility_kind"] for row in rows] == ["food", "food", "document"]
    assert len(list((tmp_path / "utility").glob("*.jpg"))) == 3


def test_compose_sources_rejects_missing_provenance_before_replacing_destination(tmp_path):
    food = tmp_path / "food"
    docs = tmp_path / "docs"
    destination = tmp_path / "utility"
    _source(food, "food", [{"category": "dish"}])
    _source(
        docs,
        "doc",
        [
            {
                "source": "wikimedia-commons",
                "license": "Public domain",
                "description_url": "https://commons.wikimedia.org/wiki/File:Doc.jpg",
            }
        ],
    )
    destination.mkdir()
    (destination / "sentinel.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        compose_sources(food, docs, destination, food_count=1, document_count=1)

    assert (destination / "sentinel.txt").read_text(encoding="utf-8") == "old"


def test_final_provenance_validation_rejects_forged_licenses():
    fake_food = {
        "source": "isia-food500",
        "license": "CC BY 4.0",
        "source_page": "http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html",
        "source_member": "ISIA_Food500/images/Dish/item.jpg",
        "license_note": "pretend licensed",
        "distribution": "local-only-not-redistributed",
    }
    fake_document = {
        "source": "wikimedia-commons",
        "license": "proprietary",
        "license_url": "https://example.test/license",
        "attribution": "author",
        "description_url": "https://commons.wikimedia.org/wiki/File:Doc.jpg",
        "original_url": "https://upload.wikimedia.org/doc.jpg",
        "source_url": "https://upload.wikimedia.org/thumb/doc.jpg",
    }

    with pytest.raises(ValueError, match="未知许可"):
        _validate_provenance(fake_food, "food")
    with pytest.raises(ValueError, match="许可"):
        _validate_provenance(fake_document, "document")
