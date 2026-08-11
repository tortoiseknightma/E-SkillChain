import json
from pathlib import Path
import tarfile

import pytest
from PIL import Image

from skillchain.data.recipe1m_plus_selection import (
    RecipeSelectionError,
    build_image_fallback,
    build_selection,
    promote_completed_images,
    verify_selected_images,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_layers(path: Path, layer1: object, layer2: object) -> None:
    stage = path.parent / "stage"
    stage.mkdir()
    _write_json(stage / "layer1.json", layer1)
    _write_json(stage / "layer2.json", layer2)
    with tarfile.open(path, "w:gz") as archive:
        archive.add(stage / "layer1.json", arcname="layer1.json")
        archive.add(stage / "layer2.json", arcname="layer2.json")


def _mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "reviewed_by": "fixture reviewer",
        "reviewed_at": "2026-08-03",
        "max_total_images": 3,
        "entities": [
            {
                "entity_id": "dish_fixture",
                "recipe_title_aliases": ["Fixture Dish"],
                "max_recipes": 2,
                "max_images_per_recipe": 1,
            }
        ],
    }


def test_selection_joins_valid_reviewed_recipe_pairs_without_urls_in_inventory(tmp_path):
    archive = tmp_path / "layers.tar.gz"
    _write_layers(
        archive,
        [
            {"id": "r2", "title": "Fixture Dish", "partition": "train"},
            {"id": "r1", "title": "Fixture Dish", "partition": "val"},
            {"id": "ignored", "title": "Other", "partition": "train"},
        ],
        [
            {"id": "r1", "images": [{"id": "r1a.jpg", "url": "https://example.test/r1a"}]},
            {"id": "r2", "images": [{"id": "r2a.jpg", "url": "https://example.test/r2a"}]},
        ],
    )
    det_ingrs = tmp_path / "det.json"
    _write_json(
        det_ingrs,
        [
            {"id": "r1", "valid": [True]},
            {"id": "r2", "valid": [False]},
        ],
    )
    layer2_plus = tmp_path / "layer2+.json"
    _write_json(
        layer2_plus,
        [
            {"id": "r1", "images": [{"id": "r1a.jpg", "url": "https://example.test/r1a"}]},
            {"id": "r2", "images": [{"id": "r2a.jpg", "url": "https://example.test/r2a"}]},
        ],
    )
    mapping = tmp_path / "map.json"
    _write_json(mapping, _mapping())
    inventory_path = tmp_path / "inventory.json"
    aria2_path = tmp_path / "selected.txt"

    inventory = build_selection(
        layers_archive=archive,
        det_ingrs=det_ingrs,
        layer2_plus=layer2_plus,
        reviewed_mapping=mapping,
        inventory_path=inventory_path,
        aria2_input_path=aria2_path,
    )

    assert inventory["recipe_count"] == 1
    assert inventory["image_count"] == 1
    assert inventory["entities"] == [
        {
            "entity_id": "dish_fixture",
            "recipes": [
                {
                    "recipe_id": "r1",
                    "title": "Fixture Dish",
                    "partition": "val",
                    "image_ids": ["r1a.jpg"],
                }
            ],
        }
    ]
    assert "https://" not in inventory_path.read_text(encoding="utf-8")
    assert "https://example.test/r1a" in aria2_path.read_text(encoding="utf-8")
    assert f"dir={tmp_path / 'selected_images'}" in aria2_path.read_text(
        encoding="utf-8"
    )
    assert "out=r1a.jpg.part" in aria2_path.read_text(encoding="utf-8")


def test_selection_rejects_an_unmapped_or_invalid_reviewed_entity(tmp_path):
    archive = tmp_path / "layers.tar.gz"
    _write_layers(
        archive,
        [{"id": "r1", "title": "Other", "partition": "train"}],
        [{"id": "r1", "images": [{"id": "r1.jpg", "url": "https://example.test/r1"}]}],
    )
    det_ingrs = tmp_path / "det.json"
    _write_json(det_ingrs, [{"id": "r1", "valid": [True]}])
    layer2_plus = tmp_path / "layer2+.json"
    _write_json(
        layer2_plus,
        [{"id": "r1", "images": [{"id": "r1.jpg", "url": "https://example.test/r1"}]}],
    )
    mapping = tmp_path / "map.json"
    _write_json(mapping, _mapping())

    with pytest.raises(RecipeSelectionError, match="no valid matching"):
        build_selection(
            layers_archive=archive,
            det_ingrs=det_ingrs,
            layer2_plus=layer2_plus,
            reviewed_mapping=mapping,
            inventory_path=tmp_path / "inventory.json",
            aria2_input_path=tmp_path / "selected.txt",
        )


def test_selection_accepts_recipe1m_plus_only_image_links(tmp_path):
    archive = tmp_path / "layers.tar.gz"
    _write_layers(
        archive,
        [{"id": "r1", "title": "Fixture Dish", "partition": "train"}],
        [],
    )
    det_ingrs = tmp_path / "det.json"
    _write_json(det_ingrs, [{"id": "r1", "valid": [True]}])
    layer2_plus = tmp_path / "layer2+.json"
    _write_json(
        layer2_plus,
        [{"id": "r1", "images": [{"id": "r1.jpg", "url": "https://example.test/r1"}]}],
    )
    mapping = tmp_path / "map.json"
    _write_json(mapping, _mapping())

    inventory = build_selection(
        layers_archive=archive,
        det_ingrs=det_ingrs,
        layer2_plus=layer2_plus,
        reviewed_mapping=mapping,
        inventory_path=tmp_path / "inventory.json",
        aria2_input_path=tmp_path / "selected.txt",
    )

    assert inventory["layer2_linked_recipe_count"] == 0
    assert inventory["layer2_plus_only_recipe_count"] == 1


def test_selection_can_exclude_unavailable_recipe_ids(tmp_path):
    archive = tmp_path / "layers.tar.gz"
    _write_layers(
        archive,
        [
            {"id": "r1", "title": "Fixture Dish", "partition": "train"},
            {"id": "r2", "title": "Fixture Dish", "partition": "val"},
        ],
        [],
    )
    det_ingrs = tmp_path / "det.json"
    _write_json(det_ingrs, [{"id": "r1", "valid": [True]}, {"id": "r2", "valid": [True]}])
    layer2_plus = tmp_path / "layer2+.json"
    _write_json(
        layer2_plus,
        [
            {"id": "r1", "images": [{"id": "r1.jpg", "url": "https://example.test/r1"}]},
            {"id": "r2", "images": [{"id": "r2.jpg", "url": "https://example.test/r2"}]},
        ],
    )
    mapping = tmp_path / "map.json"
    _write_json(mapping, _mapping())

    inventory = build_selection(
        layers_archive=archive,
        det_ingrs=det_ingrs,
        layer2_plus=layer2_plus,
        reviewed_mapping=mapping,
        inventory_path=tmp_path / "inventory.json",
        aria2_input_path=tmp_path / "selected.txt",
        excluded_recipe_ids={"r1"},
    )

    assert inventory["entities"][0]["recipes"][0]["recipe_id"] == "r2"


def test_image_promotion_and_verification_keep_invalid_partials(tmp_path):
    root = tmp_path / "selected"
    staging = root / "selected_images"
    staging.mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(staging / "good.jpg.part", format="JPEG")
    (staging / "bad.jpg.part").write_bytes(b"not an image")
    inventory = root / "inventory.json"
    _write_json(
        inventory,
        {
            "entities": [
                {
                    "entity_id": "fixture",
                    "recipes": [{"image_ids": ["good.jpg", "bad.jpg"]}],
                }
            ]
        },
    )

    promoted = promote_completed_images(root)
    report = verify_selected_images(inventory)

    assert promoted.promoted == 1
    assert promoted.invalid == 1
    assert (root / "images" / "good.jpg").is_file()
    assert (staging / "bad.jpg.part").is_file()
    assert report.expected == 2
    assert report.verified == 1
    assert report.missing == 1


def test_fallback_replaces_only_a_missing_image_with_same_recipe_alternative(tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    inventory = root / "inventory.json"
    _write_json(
        inventory,
        {
            "schema_version": 1,
            "recipe_count": 1,
            "image_count": 1,
            "entities": [
                {
                    "entity_id": "fixture",
                    "recipes": [
                        {
                            "recipe_id": "r1",
                            "title": "Fixture Dish",
                            "partition": "train",
                            "image_ids": ["old.jpg"],
                        }
                    ],
                }
            ],
        },
    )
    layer2_plus = tmp_path / "layer2+.json"
    _write_json(
        layer2_plus,
        [
            {
                "id": "r1",
                "images": [
                    {"id": "old.jpg", "url": "https://example.test/old"},
                    {"id": "alternate.jpg", "url": "https://example.test/alternate"},
                ],
            }
        ],
    )
    fallback_inventory = root / "inventory-v2.json"
    aria2_input = tmp_path / "fallback.txt"

    result = build_image_fallback(
        inventory_path=inventory,
        layer2_plus=layer2_plus,
        fallback_inventory_path=fallback_inventory,
        aria2_input_path=aria2_input,
    )

    assert result["fallback_replaced_recipe_count"] == 1
    assert result["entities"][0]["recipes"][0]["image_ids"] == ["alternate.jpg"]
    assert "https://" not in fallback_inventory.read_text(encoding="utf-8")
    assert "https://example.test/alternate" in aria2_input.read_text(encoding="utf-8")
