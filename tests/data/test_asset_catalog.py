"""Contract tests for the content-aware dataset asset catalog.

These tests deliberately exercise only the public catalog API.  In particular,
the leakage component is expected to be the transitive closure of *typed* exact
content, perceptual-neighbour, source-record, product and derivation edges.
"""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

import imagehash
import numpy as np
import pytest
from PIL import Image, ImageOps
from pydantic import ValidationError

from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)


def _write_pattern(path: Path, seed: int) -> Path:
    """Write deterministic non-flat pixels so pHash comparisons are meaningful."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pixels = bytes(rng.randrange(256) for _ in range(64 * 64 * 3))
    Image.frombytes("RGB", (64, 64), pixels).save(path, format="PNG")
    return path


def _draft(
    local_path: str,
    *,
    dataset: str = "dataset-a",
    revision: str = "revision-1",
    record: str = "record-1",
    product_id: str | None = None,
    parents: list[str] | None = None,
) -> DatasetAssetDraft:
    return DatasetAssetDraft(
        source_dataset=dataset,
        source_revision=revision,
        source_record_id=record,
        local_path=local_path,
        product_id=product_id,
        license_id="CC0-1.0",
        source_url=f"https://example.test/{dataset}/{record}",
        attribution="test fixture",
        cloud_upload_allowed=True,
        public_demo_allowed=True,
        derivation_parent_asset_ids=parents or [],
    )


def _inventory(
    asset_root: Path,
    local_path: str,
    *,
    seed: int,
    dataset: str = "dataset-a",
    revision: str = "revision-1",
    record: str = "record-1",
    product_id: str | None = None,
    parents: list[str] | None = None,
):
    _write_pattern(asset_root / Path(local_path), seed)
    return inventory_dataset_asset(
        _draft(
            local_path,
            dataset=dataset,
            revision=revision,
            record=record,
            product_id=product_id,
            parents=parents,
        ),
        asset_root,
    )


def _policy(threshold: int = 0) -> NearDuplicatePolicy:
    return NearDuplicatePolicy(
        policy_version=f"test-phash-hamming-{threshold}-v1",
        max_phash_hamming_distance=threshold,
    )


def _publish_and_load(
    assets,
    output_dir: Path,
    asset_root: Path,
    *,
    threshold: int = 0,
    verify_files: bool = True,
):
    publish_asset_catalog(
        assets,
        output_dir,
        asset_root,
        _policy(threshold),
        coverage_roots=["."],
    )
    return load_asset_catalog(output_dir, asset_root, verify_files=verify_files)


def _phash_distance(left, right) -> int:
    return int(imagehash.hex_to_hash(left.phash) - imagehash.hex_to_hash(right.phash))


def _related_image_arrays() -> list[np.ndarray]:
    """Return a smooth deterministic path through pHash space."""

    rng = np.random.default_rng(20260719)
    first = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    second = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return [
        ((1.0 - alpha) * first.astype(float) + alpha * second.astype(float)).astype(
            np.uint8
        )
        for alpha in np.linspace(0.0, 1.0, 21)
    ]


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_inventory_hashes_final_bytes_and_decoded_final_image(tmp_path: Path):
    asset_root = tmp_path / "assets"
    path = _write_pattern(asset_root / "images" / "sample.png", seed=11)

    asset = inventory_dataset_asset(
        _draft("images/sample.png", record="sample"),
        asset_root,
    )

    assert asset.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as opened:
        expected_phash = str(
            imagehash.phash(ImageOps.exif_transpose(opened).convert("RGB"))
        )
    assert asset.phash == expected_phash
    assert asset.local_path == "images/sample.png"
    assert asset.asset_id


def test_catalog_resolves_ids_paths_components_and_verifies_references(tmp_path: Path):
    asset_root = tmp_path / "assets"
    asset = _inventory(
        asset_root,
        "images/a.png",
        seed=21,
        record="a",
    )
    catalog = _publish_and_load([asset], tmp_path / "catalog", asset_root)

    assert catalog.resolve_path("images/a.png").asset_id == asset.asset_id
    assert catalog.resolve_path(Path("images/a.png")).sha256 == asset.sha256
    assert catalog.resolve_asset_id(asset.asset_id).local_path == "images/a.png"
    component = catalog.component_for_asset(asset.asset_id)
    assert component
    assert (
        catalog.verify_reference(
            asset.asset_id,
            "images/a.png",
            leakage_group_id=component,
        ).asset_id
        == asset.asset_id
    )

    with pytest.raises((KeyError, ValueError)):
        catalog.resolve_asset_id("asset.does-not-exist")
    with pytest.raises((KeyError, ValueError)):
        catalog.resolve_path("images/missing.png")
    with pytest.raises(ValueError):
        catalog.verify_reference(asset.asset_id, "images/not-a.png")
    with pytest.raises(ValueError):
        catalog.verify_reference(
            asset.asset_id,
            "images/a.png",
            leakage_group_id="component.wrong",
        )


def test_catalog_path_lookup_preserves_portable_canonical_case(tmp_path: Path):
    asset_root = tmp_path / "assets"
    asset = _inventory(
        asset_root,
        "Images/MixedCase.PNG",
        seed=22,
        record="mixed-case",
    )
    catalog = _publish_and_load([asset], tmp_path / "catalog", asset_root)

    assert catalog.resolve_path("Images/MixedCase.PNG").asset_id == asset.asset_id
    with pytest.raises(ValueError):
        catalog.resolve_path("images/mixedcase.png")


def test_identical_final_bytes_join_one_component_across_paths_and_sources(
    tmp_path: Path,
):
    asset_root = tmp_path / "assets"
    first_path = _write_pattern(asset_root / "one" / "a.png", seed=31)
    second_path = asset_root / "two" / "b.png"
    second_path.parent.mkdir(parents=True)
    shutil.copyfile(first_path, second_path)
    first = inventory_dataset_asset(
        _draft("one/a.png", dataset="source-a", record="a"), asset_root
    )
    second = inventory_dataset_asset(
        _draft("two/b.png", dataset="source-b", record="b"), asset_root
    )

    catalog = _publish_and_load([first, second], tmp_path / "catalog", asset_root)

    assert first.asset_id != second.asset_id
    assert first.sha256 == second.sha256
    assert catalog.component_for_asset(first.asset_id) == catalog.component_for_asset(
        second.asset_id
    )


def test_manifest_self_hash_round_trips_two_digit_component_size(tmp_path: Path):
    asset_root = tmp_path / "assets"
    source = _write_pattern(asset_root / "images" / "image-00.png", seed=32)
    assets = []
    for index in range(10):
        local_path = f"images/image-{index:02d}.png"
        destination = asset_root / local_path
        if index:
            shutil.copyfile(source, destination)
        assets.append(
            inventory_dataset_asset(
                _draft(local_path, record=f"record-{index:02d}"),
                asset_root,
            )
        )

    catalog = _publish_and_load(assets, tmp_path / "catalog", asset_root)

    assert catalog.manifest.component_size_histogram == {10: 1}
    assert catalog.manifest.component_count == 1


def test_phash_distance_at_threshold_joins_but_threshold_plus_one_does_not(
    tmp_path: Path,
):
    asset_root = tmp_path / "assets"
    related = _related_image_arrays()
    for name, pixels in (("a", related[0]), ("b", related[5])):
        path = asset_root / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path, format="PNG")
    first = inventory_dataset_asset(
        _draft("a.png", dataset="source-a", record="a"), asset_root
    )
    second = inventory_dataset_asset(
        _draft("b.png", dataset="source-b", record="b"), asset_root
    )
    distance = _phash_distance(first, second)
    assert 0 < distance <= 16
    assert first.sha256 != second.sha256

    inclusive = _publish_and_load(
        [first, second],
        tmp_path / "inclusive",
        asset_root,
        threshold=distance,
    )
    exclusive = _publish_and_load(
        [first, second],
        tmp_path / "exclusive",
        asset_root,
        threshold=distance - 1,
    )

    assert inclusive.component_for_asset(
        first.asset_id
    ) == inclusive.component_for_asset(second.asset_id)
    assert exclusive.component_for_asset(
        first.asset_id
    ) != exclusive.component_for_asset(second.asset_id)


def _find_transitive_phash_chain() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Find deterministic A~B~C hashes where A and C have no direct edge."""

    arrays = _related_image_arrays()
    hashes = [imagehash.phash(Image.fromarray(array)) for array in arrays]
    for middle_index, middle_hash in enumerate(hashes):
        for left_index, left_hash in enumerate(hashes):
            if left_index == middle_index:
                continue
            left_distance = left_hash - middle_hash
            for right_index, right_hash in enumerate(hashes):
                if right_index in {left_index, middle_index}:
                    continue
                right_distance = middle_hash - right_hash
                threshold = max(left_distance, right_distance)
                if 0 < threshold <= 16 and left_hash - right_hash > threshold:
                    return (
                        arrays[left_index],
                        arrays[middle_index],
                        arrays[right_index],
                        threshold,
                    )
    raise AssertionError("deterministic pHash fixtures did not contain a chain")


def test_near_duplicate_edges_are_closed_transitively(tmp_path: Path):
    asset_root = tmp_path / "assets"
    left_pixels, middle_pixels, right_pixels, threshold = _find_transitive_phash_chain()
    assets = []
    for name, pixels in (
        ("left", left_pixels),
        ("middle", middle_pixels),
        ("right", right_pixels),
    ):
        path = asset_root / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path, format="PNG")
        assets.append(
            inventory_dataset_asset(
                _draft(
                    f"{name}.png",
                    dataset=f"source-{name}",
                    record=name,
                ),
                asset_root,
            )
        )
    left, middle, right = assets
    assert _phash_distance(left, middle) <= threshold
    assert _phash_distance(middle, right) <= threshold
    assert _phash_distance(left, right) > threshold

    catalog = _publish_and_load(
        assets,
        tmp_path / "catalog",
        asset_root,
        threshold=threshold,
    )

    assert {catalog.component_for_asset(asset.asset_id) for asset in assets} == {
        catalog.component_for_asset(left.asset_id)
    }


def test_source_record_identity_ignores_revision_but_asset_identity_does_not(
    tmp_path: Path,
):
    asset_root = tmp_path / "assets"
    old = _inventory(
        asset_root,
        "old.png",
        seed=51,
        dataset="mutable-source",
        revision="rev-old",
        record="record-7",
    )
    new = _inventory(
        asset_root,
        "new.png",
        seed=52,
        dataset="mutable-source",
        revision="rev-new",
        record="record-7",
    )
    assert old.asset_id != new.asset_id
    assert old.phash != new.phash

    catalog = _publish_and_load([old, new], tmp_path / "catalog", asset_root)

    assert catalog.component_for_asset(old.asset_id) == catalog.component_for_asset(
        new.asset_id
    )


def test_product_edges_are_scoped_by_source_dataset(tmp_path: Path):
    asset_root = tmp_path / "assets"
    first_view = _inventory(
        asset_root,
        "first.png",
        seed=61,
        dataset="shop-a",
        record="record-a",
        product_id="product-1",
    )
    second_view = _inventory(
        asset_root,
        "second.png",
        seed=62,
        dataset="shop-a",
        record="record-b",
        product_id="product-1",
    )
    other_dataset = _inventory(
        asset_root,
        "other.png",
        seed=63,
        dataset="shop-b",
        record="record-c",
        product_id="product-1",
    )
    catalog = _publish_and_load(
        [first_view, second_view, other_dataset],
        tmp_path / "catalog",
        asset_root,
    )

    first_component = catalog.component_for_asset(first_view.asset_id)
    assert first_component == catalog.component_for_asset(second_view.asset_id)
    assert first_component != catalog.component_for_asset(other_dataset.asset_id)


def test_derivation_parent_edges_join_visually_different_assets(tmp_path: Path):
    asset_root = tmp_path / "assets"
    parent = _inventory(
        asset_root,
        "parent.png",
        seed=71,
        dataset="source-a",
        record="parent",
    )
    child = _inventory(
        asset_root,
        "child.png",
        seed=72,
        dataset="source-b",
        record="child",
        parents=[parent.asset_id],
    )
    assert parent.phash != child.phash

    catalog = _publish_and_load([parent, child], tmp_path / "catalog", asset_root)

    assert catalog.component_for_asset(parent.asset_id) == catalog.component_for_asset(
        child.asset_id
    )


def test_equal_raw_tokens_in_different_edge_namespaces_do_not_join(tmp_path: Path):
    asset_root = tmp_path / "assets"
    source_record = _inventory(
        asset_root,
        "record.png",
        seed=81,
        dataset="same-dataset",
        record="shared-token",
    )
    product = _inventory(
        asset_root,
        "product.png",
        seed=82,
        dataset="same-dataset",
        record="different-record",
        product_id="shared-token",
    )
    assert source_record.phash != product.phash

    catalog = _publish_and_load(
        [source_record, product], tmp_path / "catalog", asset_root
    )

    assert catalog.component_for_asset(
        source_record.asset_id
    ) != catalog.component_for_asset(product.asset_id)


def test_catalog_bytes_and_hash_are_deterministic_under_input_reordering(
    tmp_path: Path,
):
    asset_root = tmp_path / "assets"
    assets = [
        _inventory(
            asset_root,
            f"images/{index}.png",
            seed=90 + index,
            dataset=f"source-{index}",
            record=f"record-{index}",
        )
        for index in range(3)
    ]
    first_dir = tmp_path / "catalog-first"
    second_dir = tmp_path / "catalog-second"
    first = _publish_and_load(assets, first_dir, asset_root)
    second = _publish_and_load(list(reversed(assets)), second_dir, asset_root)

    assert first.manifest.catalog_sha256 == second.manifest.catalog_sha256
    assert _tree_snapshot(first_dir) == _tree_snapshot(second_dir)


def test_catalog_rejects_tampered_asset_file_catalog_bytes_and_declared_hash(
    tmp_path: Path,
):
    asset_root = tmp_path / "assets"
    asset = _inventory(
        asset_root,
        "image.png",
        seed=101,
        dataset="source-a",
        record="a",
    )

    bad_hash = asset.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="(?i)sha|hash|摘要"):
        publish_asset_catalog(
            [bad_hash],
            tmp_path / "bad-hash-catalog",
            asset_root,
            _policy(),
            coverage_roots=["."],
        )

    output_dir = tmp_path / "catalog"
    catalog = _publish_and_load([asset], output_dir, asset_root)
    (asset_root / "image.png").write_bytes(b"tampered asset bytes")
    with pytest.raises(ValueError, match="(?i)sha|hash|tamper|摘要"):
        load_asset_catalog(output_dir, asset_root, verify_files=True)

    # The manifest self-hash protects the descriptors, and this descriptor
    # protects the canonical asset JSONL bytes.
    catalog_path = output_dir / catalog.manifest.assets.path
    assert hashlib.sha256(catalog_path.read_bytes()).hexdigest() == (
        catalog.manifest.assets.sha256
    )
    catalog_path.write_bytes(catalog_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="(?i)catalog|sha|hash|摘要"):
        load_asset_catalog(output_dir, asset_root, verify_files=False)


def test_publish_rejects_casefold_duplicate_paths(tmp_path: Path):
    asset_root = tmp_path / "assets"
    first = _inventory(
        asset_root,
        "Images/Asset.PNG",
        seed=111,
        dataset="source-a",
        record="a",
    )
    second = first.model_copy(
        update={
            "asset_id": first.asset_id + ".other",
            "source_dataset": "source-b",
            "source_record_id": "b",
            "local_path": "images/asset.png",
        }
    )

    with pytest.raises(ValueError, match="(?i)path|duplicate|case|路径|重复"):
        publish_asset_catalog(
            [first, second],
            tmp_path / "catalog",
            asset_root,
            _policy(),
            coverage_roots=["."],
        )


def test_load_rejects_noncanonical_manifest_bytes(tmp_path: Path):
    asset_root = tmp_path / "assets"
    asset = _inventory(
        asset_root,
        "image.png",
        seed=109,
        dataset="source-a",
        record="a",
    )
    output_dir = tmp_path / "catalog"
    _publish_and_load([asset], output_dir, asset_root)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="canonical|manifest"):
        load_asset_catalog(output_dir, asset_root, verify_files=False)


def test_inventory_fails_closed_for_missing_or_out_of_root_files(tmp_path: Path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()

    with pytest.raises((FileNotFoundError, ValueError)):
        inventory_dataset_asset(_draft("missing.png", record="missing"), asset_root)

    outside = tmp_path / "outside.png"
    _write_pattern(outside, seed=121)
    with pytest.raises((ValidationError, ValueError)):
        draft = _draft("../outside.png", record="outside")
        inventory_dataset_asset(draft, asset_root)

    with pytest.raises((ValidationError, ValueError)):
        draft = _draft(outside.resolve().as_posix(), record="absolute")
        inventory_dataset_asset(draft, asset_root)


def test_formal_catalog_requires_nonempty_and_complete_coverage(tmp_path: Path):
    asset_root = tmp_path / "assets"
    asset = _inventory(
        asset_root,
        "query_images/a.png",
        seed=131,
        record="a",
    )

    with pytest.raises(ValueError, match="coverage"):
        publish_asset_catalog(
            [asset],
            tmp_path / "empty-coverage",
            asset_root,
            _policy(),
            coverage_roots=[],
        )

    (asset_root / "unrelated").mkdir()
    with pytest.raises(ValueError, match="coverage|不在"):
        publish_asset_catalog(
            [asset],
            tmp_path / "unrelated-coverage",
            asset_root,
            _policy(),
            coverage_roots=["unrelated"],
        )


def test_coverage_rejects_decodable_images_outside_supported_policy(tmp_path: Path):
    asset_root = tmp_path / "assets"
    asset = _inventory(asset_root, "registered.png", seed=141, record="registered")
    unsupported = asset_root / "unregistered.gif"
    Image.new("RGB", (16, 16), "red").save(unsupported, format="GIF")

    with pytest.raises(ValueError, match="格式|format|未支持"):
        publish_asset_catalog(
            [asset],
            tmp_path / "catalog",
            asset_root,
            _policy(),
            coverage_roots=["."],
        )


def test_coverage_detects_casefold_collision_before_set_comparison(tmp_path: Path):
    asset_root = tmp_path / "assets"
    upper_path = _write_pattern(asset_root / "A.png", seed=151)
    lower_path = _write_pattern(asset_root / "a.png", seed=152)
    if upper_path.samefile(lower_path):
        pytest.skip("filesystem is case-insensitive")
    asset = inventory_dataset_asset(_draft("A.png", record="upper"), asset_root)

    with pytest.raises(ValueError, match="casefold|冲突"):
        publish_asset_catalog(
            [asset],
            tmp_path / "catalog",
            asset_root,
            _policy(),
            coverage_roots=["."],
        )


def test_derivation_parent_order_is_canonicalized(tmp_path: Path):
    asset_root = tmp_path / "assets"
    first = _inventory(asset_root, "first.png", seed=161, record="first")
    second = _inventory(asset_root, "second.png", seed=162, record="second")
    child = _inventory(
        asset_root,
        "child.png",
        seed=163,
        record="child",
        parents=[second.asset_id, first.asset_id],
    )

    assert child.derivation_parent_asset_ids == sorted(
        [first.asset_id, second.asset_id]
    )
