"""Build a bounded, reviewed Recipe1M+ image/recipe selection.

The Recipe1M+ release is intentionally not mirrored wholesale.  This module
joins its three metadata layers only after a human has supplied exact,
reviewed title aliases.  The public inventory contains recipe and image IDs,
never the upstream image URLs.  A separate aria2 input is a local-only
transport artifact and must live outside the repository.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import tarfile

import ijson
from PIL import Image


class RecipeSelectionError(RuntimeError):
    """Raised when a reviewed bounded selection cannot be built safely."""


@dataclass(frozen=True)
class ReviewedEntity:
    entity_id: str
    aliases: frozenset[str]
    max_recipes: int
    max_images_per_recipe: int


@dataclass(frozen=True)
class RecipeImageReport:
    expected: int
    verified: int
    missing: int
    invalid: int
    promoted: int = 0
    active_partials: int = 0


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def _normalize_title(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeSelectionError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: object, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise RecipeSelectionError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _load_reviewed_entities(path: Path) -> tuple[dict[str, ReviewedEntity], int, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecipeSelectionError(f"cannot read reviewed mapping: {path}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RecipeSelectionError("reviewed mapping must use schema_version 1")
    _require_string(raw.get("reviewed_by"), "reviewed_by")
    _require_string(raw.get("reviewed_at"), "reviewed_at")
    max_total_images = _require_positive_int(
        raw.get("max_total_images"), "max_total_images", maximum=5_000
    )
    values = raw.get("entities")
    if not isinstance(values, list) or not values:
        raise RecipeSelectionError("reviewed mapping must contain at least one entity")

    aliases: dict[str, ReviewedEntity] = {}
    entities: dict[str, ReviewedEntity] = {}
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise RecipeSelectionError(f"entities[{index}] must be an object")
        entity_id = _require_string(item.get("entity_id"), f"entities[{index}].entity_id")
        if entity_id in entities:
            raise RecipeSelectionError(f"duplicate entity_id: {entity_id}")
        raw_aliases = item.get("recipe_title_aliases")
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise RecipeSelectionError(
                f"entities[{index}].recipe_title_aliases must be a non-empty array"
            )
        normalized = frozenset(_normalize_title(alias) for alias in raw_aliases)
        if not normalized or "" in normalized:
            raise RecipeSelectionError(f"entities[{index}] contains an empty title alias")
        entity = ReviewedEntity(
            entity_id=entity_id,
            aliases=normalized,
            max_recipes=_require_positive_int(
                item.get("max_recipes"), f"entities[{index}].max_recipes", maximum=10
            ),
            max_images_per_recipe=_require_positive_int(
                item.get("max_images_per_recipe"),
                f"entities[{index}].max_images_per_recipe",
                maximum=3,
            ),
        )
        for alias in entity.aliases:
            owner = aliases.get(alias)
            if owner is not None:
                raise RecipeSelectionError(
                    f"reviewed title alias belongs to both {owner.entity_id} and {entity_id}"
                )
            aliases[alias] = entity
        entities[entity_id] = entity
    return entities, max_total_images, hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_archive_items(archive_path: Path, member_name: str) -> Iterable[Mapping[str, object]]:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.name == member_name]
            if len(members) != 1 or not members[0].isfile():
                raise RecipeSelectionError(
                    f"{archive_path} must contain regular {member_name} exactly once"
                )
            source = archive.extractfile(members[0])
            assert source is not None
            for item in ijson.items(source, "item"):
                if not isinstance(item, dict):
                    raise RecipeSelectionError(f"{member_name} must be an array of objects")
                yield item
    except (OSError, tarfile.TarError, ijson.JSONError) as error:
        raise RecipeSelectionError(f"cannot read {member_name} from {archive_path}") from error


def _iter_json_items(path: Path) -> Iterable[Mapping[str, object]]:
    try:
        with path.open("rb") as source:
            for item in ijson.items(source, "item"):
                if not isinstance(item, dict):
                    raise RecipeSelectionError(f"{path} must be an array of objects")
                yield item
    except (OSError, ijson.JSONError) as error:
        raise RecipeSelectionError(f"cannot read JSON array: {path}") from error


def _valid_ingredients(row: Mapping[str, object]) -> bool:
    values = row.get("valid")
    return isinstance(values, list) and bool(values) and all(value is True for value in values)


def _image_rows(row: Mapping[str, object], *, label: str) -> list[tuple[str, str]]:
    values = row.get("images")
    if not isinstance(values, list):
        raise RecipeSelectionError(f"{label}.images must be an array")
    images: dict[str, str] = {}
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise RecipeSelectionError(f"{label}.images[{index}] must be an object")
        image_id = _require_string(item.get("id"), f"{label}.images[{index}].id")
        if Path(image_id).name != image_id or not image_id.lower().endswith(".jpg"):
            raise RecipeSelectionError(f"{label}.images[{index}].id is not a safe JPEG name")
        url = _require_string(item.get("url"), f"{label}.images[{index}].url")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise RecipeSelectionError(f"{label}.images[{index}] has an unusable URL")
        previous = images.get(image_id)
        if previous is not None and previous != url:
            raise RecipeSelectionError(f"{label} repeats image ID with different URLs")
        images[image_id] = url
    return sorted(images.items())


def _write_create_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        try:
            with path.open("xb") as target:
                target.write(content)
        except FileExistsError:
            existing = path.read_bytes()
        else:
            return
    if existing != content:
        raise RecipeSelectionError(
            f"refusing to overwrite a different existing artifact: {path}"
        )


def _is_valid_jpeg(path: Path) -> bool:
    if not path.is_file() or not path.name.lower().endswith((".jpg", ".jpg.part")):
        return False
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                return False
            image.load()
    except OSError:
        return False
    return True


def promote_completed_images(selection_root: Path) -> RecipeImageReport:
    """Atomically publish complete JPEG partials without deleting failures."""

    root = Path(selection_root)
    staging = root / "selected_images"
    published = root / "images"
    published.mkdir(parents=True, exist_ok=True)
    promoted = invalid = active_partials = 0
    for partial in sorted(staging.glob("*.jpg.part")):
        control = partial.with_name(partial.name + ".aria2")
        if control.exists():
            active_partials += 1
            continue
        if not _is_valid_jpeg(partial):
            invalid += 1
            continue
        target = published / partial.name.removesuffix(".part")
        if target.exists():
            raise RecipeSelectionError(
                f"refusing to overwrite published image: {target.name}"
            )
        os.replace(partial, target)
        promoted += 1
    return RecipeImageReport(
        expected=0,
        verified=0,
        missing=0,
        invalid=invalid,
        promoted=promoted,
        active_partials=active_partials,
    )


def verify_selected_images(inventory_path: Path) -> RecipeImageReport:
    """Verify every image ID in a URL-free inventory is published as a JPEG."""

    try:
        inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecipeSelectionError(f"cannot read selection inventory: {inventory_path}") from error
    entities = inventory.get("entities") if isinstance(inventory, dict) else None
    if not isinstance(entities, list):
        raise RecipeSelectionError("selection inventory has no entities array")
    image_ids: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict) or not isinstance(entity.get("recipes"), list):
            raise RecipeSelectionError("selection inventory contains an invalid entity")
        for recipe in entity["recipes"]:
            if not isinstance(recipe, dict) or not isinstance(recipe.get("image_ids"), list):
                raise RecipeSelectionError("selection inventory contains an invalid recipe")
            for image_id in recipe["image_ids"]:
                image_ids.add(_require_string(image_id, "inventory image_id"))
    published = Path(inventory_path).parent / "images"
    verified = missing = invalid = 0
    for image_id in image_ids:
        path = published / image_id
        if not path.exists():
            missing += 1
        elif _is_valid_jpeg(path):
            verified += 1
        else:
            invalid += 1
    return RecipeImageReport(
        expected=len(image_ids),
        verified=verified,
        missing=missing,
        invalid=invalid,
    )


def build_image_fallback(
    *,
    inventory_path: Path,
    layer2_plus: Path,
    fallback_inventory_path: Path,
    aria2_input_path: Path,
) -> dict[str, object]:
    """Replace only missing images with alternatives from the same recipe.

    This is intentionally a new inventory rather than an overwrite: the
    parent selection remains an auditable record of the unavailable URLs.
    """

    source_path = Path(inventory_path)
    try:
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise RecipeSelectionError(f"cannot read selection inventory: {source_path}") from error
    if not isinstance(source, dict) or not isinstance(source.get("entities"), list):
        raise RecipeSelectionError("selection inventory has no entities array")
    inventory = json.loads(json.dumps(source))
    published = source_path.parent / "images"
    current_ids: set[str] = set()
    missing_recipes: set[str] = set()
    for entity in inventory["entities"]:
        if not isinstance(entity, dict) or not isinstance(entity.get("recipes"), list):
            raise RecipeSelectionError("selection inventory contains an invalid entity")
        for recipe in entity["recipes"]:
            if not isinstance(recipe, dict):
                raise RecipeSelectionError("selection inventory contains an invalid recipe")
            recipe_id = _require_string(recipe.get("recipe_id"), "inventory recipe_id")
            raw_ids = recipe.get("image_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                raise RecipeSelectionError("selection inventory recipe has no image IDs")
            image_ids = [_require_string(value, "inventory image_id") for value in raw_ids]
            current_ids.update(image_ids)
            if not any(_is_valid_jpeg(published / image_id) for image_id in image_ids):
                missing_recipes.add(recipe_id)
    if not missing_recipes:
        raise RecipeSelectionError("selection has no missing recipe images to replace")

    alternatives: dict[str, tuple[str, str]] = {}
    for row in _iter_json_items(Path(layer2_plus)):
        recipe_id = row.get("id")
        if recipe_id not in missing_recipes:
            continue
        recipe_id = str(recipe_id)
        if recipe_id in alternatives:
            raise RecipeSelectionError(f"layer2+ repeats missing recipe ID: {recipe_id}")
        candidates = [
            image for image in _image_rows(row, label="layer2+") if image[0] not in current_ids
        ]
        if candidates:
            alternatives[recipe_id] = candidates[0]
    missing_alternatives = missing_recipes.difference(alternatives)
    if missing_alternatives:
        raise RecipeSelectionError("some missing recipes have no alternate layer2+ image")

    pending_urls: dict[str, str] = {}
    for entity in inventory["entities"]:
        for recipe in entity["recipes"]:
            recipe_id = recipe["recipe_id"]
            if recipe_id not in alternatives:
                continue
            image_id, url = alternatives[recipe_id]
            recipe["image_ids"] = [image_id]
            previous = pending_urls.get(image_id)
            if previous is not None and previous != url:
                raise RecipeSelectionError("fallback image ID resolves to different URLs")
            pending_urls[image_id] = url

    inventory["fallback_of_inventory_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    inventory["fallback_replaced_recipe_count"] = len(alternatives)
    inventory["image_count"] = sum(
        len(recipe["image_ids"])
        for entity in inventory["entities"]
        for recipe in entity["recipes"]
    )
    _write_create_only(Path(fallback_inventory_path), _canonical_bytes(inventory))

    selection_root = Path(fallback_inventory_path).parent
    image_directory = selection_root / "selected_images"
    lines: list[str] = []
    for image_id, url in sorted(pending_urls.items()):
        if _is_valid_jpeg(selection_root / "images" / image_id):
            continue
        lines.extend(
            [
                url,
                f"  dir={image_directory}",
                f"  out={image_id}.part",
                "  continue=true",
                "  auto-file-renaming=false",
                "",
            ]
        )
    _write_create_only(Path(aria2_input_path), "\n".join(lines).encode("utf-8"))
    return inventory


def build_selection(
    *,
    layers_archive: Path,
    det_ingrs: Path,
    layer2_plus: Path,
    reviewed_mapping: Path,
    inventory_path: Path,
    aria2_input_path: Path,
    excluded_recipe_ids: set[str] | None = None,
    supersedes_inventory_sha256: str | None = None,
    extra_inventory_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a deterministic ID inventory and a local-only aria2 input.

    The inventory deliberately excludes all image URLs.  The aria2 input is
    needed solely for the approved local acquisition and must not be committed
    or displayed.
    """

    entities, max_total_images, mapping_sha256 = _load_reviewed_entities(
        Path(reviewed_mapping)
    )
    excluded_ids = frozenset(excluded_recipe_ids or ())
    alias_to_entity = {
        alias: entity for entity in entities.values() for alias in entity.aliases
    }
    candidates: dict[str, list[dict[str, str]]] = {key: [] for key in entities}
    candidate_ids: set[str] = set()
    for row in _iter_archive_items(Path(layers_archive), "layer1.json"):
        recipe_id = _require_string(row.get("id"), "layer1.id")
        if recipe_id in excluded_ids:
            continue
        entity = alias_to_entity.get(_normalize_title(row.get("title", "")))
        if entity is None:
            continue
        if recipe_id in candidate_ids:
            raise RecipeSelectionError(f"layer1 repeats candidate recipe ID: {recipe_id}")
        candidate_ids.add(recipe_id)
        candidates[entity.entity_id].append(
            {
                "recipe_id": recipe_id,
                "title": _require_string(row.get("title"), "layer1.title"),
                "partition": _require_string(row.get("partition"), "layer1.partition"),
            }
        )

    valid_ids: set[str] = set()
    for row in _iter_json_items(Path(det_ingrs)):
        recipe_id = row.get("id")
        if recipe_id in candidate_ids and _valid_ingredients(row):
            valid_ids.add(str(recipe_id))

    valid_candidates: dict[str, list[dict[str, str]]] = {}
    valid_candidate_ids: set[str] = set()
    for entity_id in entities:
        rows = [row for row in candidates[entity_id] if row["recipe_id"] in valid_ids]
        if not rows:
            raise RecipeSelectionError(
                f"reviewed entity has no valid matching Recipe1M+ recipe: {entity_id}"
            )
        valid_candidates[entity_id] = rows
        valid_candidate_ids.update(row["recipe_id"] for row in rows)

    layer2_ids: set[str] = set()
    for row in _iter_archive_items(Path(layers_archive), "layer2.json"):
        recipe_id = row.get("id")
        if recipe_id not in valid_candidate_ids:
            continue
        if _image_rows(row, label="layer2"):
            layer2_ids.add(str(recipe_id))
    plus_images: dict[str, list[tuple[str, str]]] = {}
    for row in _iter_json_items(Path(layer2_plus)):
        recipe_id = row.get("id")
        if recipe_id not in valid_candidate_ids:
            continue
        recipe_id = str(recipe_id)
        if recipe_id in plus_images:
            raise RecipeSelectionError(f"layer2+ repeats candidate recipe ID: {recipe_id}")
        images = _image_rows(row, label="layer2+")
        if images:
            plus_images[recipe_id] = images

    selected: dict[str, list[dict[str, str]]] = {}
    selected_ids: set[str] = set()
    for entity_id, entity in entities.items():
        rows = sorted(
            (
                row
                for row in valid_candidates[entity_id]
                if row["recipe_id"] in plus_images
            ),
            key=lambda row: row["recipe_id"],
        )[: entity.max_recipes]
        if not rows:
            raise RecipeSelectionError(
                f"reviewed entity has no valid Recipe1M+ recipe with an image: {entity_id}"
            )
        selected[entity_id] = rows
        selected_ids.update(row["recipe_id"] for row in rows)

    downloaded_images: dict[str, str] = {}
    inventory_entities: list[dict[str, object]] = []
    for entity_id, entity in entities.items():
        recipes: list[dict[str, object]] = []
        for recipe in selected[entity_id]:
            images = plus_images[recipe["recipe_id"]][: entity.max_images_per_recipe]
            image_ids = [image_id for image_id, url in images]
            for image_id, url in images:
                previous = downloaded_images.get(image_id)
                if previous is not None and previous != url:
                    raise RecipeSelectionError(
                        "selected image ID resolves to different layer2+ URLs"
                    )
                downloaded_images[image_id] = url
            recipes.append({**recipe, "image_ids": image_ids})
        inventory_entities.append({"entity_id": entity_id, "recipes": recipes})

    if len(downloaded_images) > max_total_images:
        raise RecipeSelectionError(
            f"selection has {len(downloaded_images)} images, above reviewed cap "
            f"{max_total_images}"
        )
    inventory: dict[str, object] = {
        "schema_version": 1,
        "mapping_sha256": mapping_sha256,
        "source_files": {
            "layers_archive_bytes": Path(layers_archive).stat().st_size,
            "det_ingrs_bytes": Path(det_ingrs).stat().st_size,
            "layer2_plus_bytes": Path(layer2_plus).stat().st_size,
        },
        "entities": inventory_entities,
        "recipe_count": len(selected_ids),
        "image_count": len(downloaded_images),
        "layer2_linked_recipe_count": len(selected_ids.intersection(layer2_ids)),
        "layer2_plus_only_recipe_count": len(selected_ids.difference(layer2_ids)),
    }
    if supersedes_inventory_sha256 is not None:
        inventory["supersedes_inventory_sha256"] = supersedes_inventory_sha256
    if extra_inventory_fields is not None:
        inventory.update(extra_inventory_fields)
    _write_create_only(Path(inventory_path), _canonical_bytes(inventory))

    selection_root = Path(inventory_path).parent
    image_directory = selection_root / "selected_images"
    published_directory = selection_root / "images"
    lines: list[str] = []
    for image_id, url in sorted(downloaded_images.items()):
        if _is_valid_jpeg(published_directory / image_id):
            continue
        lines.extend(
            [
                url,
                f"  dir={image_directory}",
                f"  out={image_id}.part",
                "  continue=true",
                "  auto-file-renaming=false",
                "",
            ]
        )
    _write_create_only(Path(aria2_input_path), "\n".join(lines).encode("utf-8"))
    return inventory


def repair_missing_recipes(
    *,
    inventory_path: Path,
    layers_archive: Path,
    det_ingrs: Path,
    layer2_plus: Path,
    reviewed_mapping: Path,
    repaired_inventory_path: Path,
    aria2_input_path: Path,
    prior_inventory_path: Path | None = None,
) -> dict[str, object]:
    """Replace only recipe IDs whose selected images are not valid JPEGs."""

    source_path = Path(inventory_path)
    try:
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise RecipeSelectionError(f"cannot read selection inventory: {source_path}") from error
    entities = source.get("entities") if isinstance(source, dict) else None
    if not isinstance(entities, list):
        raise RecipeSelectionError("selection inventory has no entities array")
    published = source_path.parent / "images"
    unavailable: set[str] = set()
    current_recipe_ids: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict) or not isinstance(entity.get("recipes"), list):
            raise RecipeSelectionError("selection inventory contains an invalid entity")
        for recipe in entity["recipes"]:
            if not isinstance(recipe, dict) or not isinstance(recipe.get("image_ids"), list):
                raise RecipeSelectionError("selection inventory contains an invalid recipe")
            recipe_id = _require_string(recipe.get("recipe_id"), "inventory recipe_id")
            current_recipe_ids.add(recipe_id)
            image_ids = [
                _require_string(image_id, "inventory image_id")
                for image_id in recipe["image_ids"]
            ]
            if not image_ids or not any(
                _is_valid_jpeg(published / image_id) for image_id in image_ids
            ):
                unavailable.add(recipe_id)
    if not unavailable:
        raise RecipeSelectionError("selection has no missing recipes to replace")
    excluded = set(unavailable)
    raw_excluded = source.get("excluded_recipe_ids", [])
    if not isinstance(raw_excluded, list):
        raise RecipeSelectionError("selection inventory has invalid excluded_recipe_ids")
    excluded.update(
        _require_string(recipe_id, "inventory excluded_recipe_id")
        for recipe_id in raw_excluded
    )
    if prior_inventory_path is not None:
        try:
            prior = json.loads(Path(prior_inventory_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RecipeSelectionError(
                f"cannot read prior selection inventory: {prior_inventory_path}"
            ) from error
        prior_entities = prior.get("entities") if isinstance(prior, dict) else None
        if not isinstance(prior_entities, list):
            raise RecipeSelectionError("prior selection inventory has no entities array")
        prior_recipe_ids: set[str] = set()
        for entity in prior_entities:
            if not isinstance(entity, dict) or not isinstance(entity.get("recipes"), list):
                raise RecipeSelectionError("prior selection inventory contains an invalid entity")
            for recipe in entity["recipes"]:
                if not isinstance(recipe, dict):
                    raise RecipeSelectionError("prior selection inventory contains an invalid recipe")
                prior_recipe_ids.add(
                    _require_string(recipe.get("recipe_id"), "prior inventory recipe_id")
                )
        excluded.update(prior_recipe_ids.difference(current_recipe_ids))
    inventory = build_selection(
        layers_archive=layers_archive,
        det_ingrs=det_ingrs,
        layer2_plus=layer2_plus,
        reviewed_mapping=reviewed_mapping,
        inventory_path=repaired_inventory_path,
        aria2_input_path=aria2_input_path,
        excluded_recipe_ids=excluded,
        supersedes_inventory_sha256=hashlib.sha256(source_bytes).hexdigest(),
        extra_inventory_fields={
            "replaced_recipe_count": len(unavailable),
            "excluded_recipe_ids": sorted(excluded),
        },
    )
    return inventory
