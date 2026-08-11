from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from skillchain.data.asset_catalog import (
    AssetNotCatalogedError,
    DatasetAssetDraft,
    NearDuplicatePolicy,
    QueryAssetReference,
    assert_query_gallery_eligible,
    audit_query_gallery_eligibility,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)


def _write_pattern(
    path: Path,
    pattern: str,
    *,
    compress_level: int = 6,
) -> None:
    image = Image.new("RGB", (64, 64), "white")
    pixels = image.load()
    assert pixels is not None
    for y in range(64):
        for x in range(64):
            black = (
                (pattern == "diagonal" and x < y)
                or (pattern == "checker" and (x // 8 + y // 8) % 2 == 0)
                or (pattern == "vertical" and x < 32)
            )
            if black:
                pixels[x, y] = (0, 0, 0)
    image.save(path, format="PNG", compress_level=compress_level)


def _build_catalog(
    tmp_path: Path,
    specs: list[tuple[str, str, str | None, int]],
):
    """Build a real frozen catalog; the tuple is name/pattern/product/compression."""

    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    assets = []
    asset_ids: dict[str, str] = {}
    for name, pattern, product_id, compress_level in specs:
        local_path = f"{name}.png"
        _write_pattern(
            asset_root / local_path,
            pattern,
            compress_level=compress_level,
        )
        asset = inventory_dataset_asset(
            DatasetAssetDraft(
                source_dataset="query-gallery-fixture",
                source_revision="fixture-revision-v1",
                source_record_id=name,
                local_path=local_path,
                product_id=product_id,
                license_id="test-only",
            ),
            asset_root,
        )
        assets.append(asset)
        asset_ids[name] = asset.asset_id

    output_dir = tmp_path / "catalog"
    publish_asset_catalog(
        assets,
        output_dir,
        asset_root,
        NearDuplicatePolicy(
            policy_version="query-gallery-test-v1",
            max_phash_hamming_distance=0,
        ),
        coverage_roots=["."],
    )
    return (
        load_asset_catalog(output_dir, asset_root, verify_files=True),
        asset_ids,
    )


def test_exact_match_rejects_query_bytes_exposed_in_gallery(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [("query", "diagonal", "product-1", 6)],
    )
    query = QueryAssetReference(
        query_id="q-exact-self",
        intent="exact_match",
        asset_id=ids["query"],
    )

    report = audit_query_gallery_eligibility(
        [query],
        [ids["query"]],
        catalog,
    )
    assert report.exact_or_near_duplicate_violations == {query.query_id: [ids["query"]]}
    assert report.missing_exact_match_positive == [query.query_id]
    assert report.violation_count == 2
    with pytest.raises(ValueError, match="q-exact-self|gallery|eligible"):
        assert_query_gallery_eligible(report)


def test_exact_match_rejects_near_duplicate_gallery_positive(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [
            ("query", "diagonal", "product-1", 0),
            ("reencoded", "diagonal", "product-1", 9),
        ],
    )
    query_asset = catalog.resolve_asset_id(ids["query"])
    reencoded_asset = catalog.resolve_asset_id(ids["reencoded"])
    assert query_asset.sha256 != reencoded_asset.sha256
    assert query_asset.phash == reencoded_asset.phash
    query = QueryAssetReference(
        query_id="q-exact-near-duplicate",
        intent="exact_match",
        asset_id=ids["query"],
    )

    report = audit_query_gallery_eligibility(
        [query],
        [ids["reencoded"]],
        catalog,
    )
    assert report.exact_or_near_duplicate_violations == {
        query.query_id: [ids["reencoded"]]
    }
    assert report.missing_exact_match_positive == [query.query_id]
    assert report.violation_count == 2
    with pytest.raises(ValueError, match="q-exact-near-duplicate|gallery|eligible"):
        assert_query_gallery_eligible(report)


def test_exact_match_accepts_distinct_non_duplicate_view_of_same_product(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [
            ("query", "diagonal", "product-1", 6),
            ("positive", "checker", "product-1", 6),
            ("distractor", "vertical", "product-2", 6),
        ],
    )
    query_asset = catalog.resolve_asset_id(ids["query"])
    positive_asset = catalog.resolve_asset_id(ids["positive"])
    assert query_asset.sha256 != positive_asset.sha256
    assert query_asset.phash != positive_asset.phash
    query = QueryAssetReference(
        query_id="q-exact-valid",
        intent="exact_match",
        asset_id=ids["query"],
    )

    report = audit_query_gallery_eligibility(
        [query],
        [ids["positive"], ids["distractor"]],
        catalog,
    )
    assert report.exact_or_near_duplicate_violations == {}
    assert report.missing_exact_match_positive == []
    assert report.divergent_product_exclusions == {}
    assert report.violation_count == 0
    assert_query_gallery_eligible(report)


def test_exact_match_rejects_gallery_without_distinct_product_positive(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [
            ("query", "diagonal", "product-1", 6),
            ("unrelated", "checker", "product-2", 6),
        ],
    )
    query = QueryAssetReference(
        query_id="q-exact-no-positive",
        intent="exact_match",
        asset_id=ids["query"],
    )

    report = audit_query_gallery_eligibility(
        [query],
        [ids["unrelated"]],
        catalog,
    )
    assert report.exact_or_near_duplicate_violations == {}
    assert report.missing_exact_match_positive == [query.query_id]
    assert report.violation_count == 1
    with pytest.raises(ValueError, match="q-exact-no-positive|gallery|eligible"):
        assert_query_gallery_eligible(report)


def test_divergent_recommendation_reports_same_product_exclusions(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [
            ("query", "diagonal", "product-1", 6),
            ("same-product", "checker", "product-1", 6),
            ("candidate", "vertical", "product-2", 6),
        ],
    )
    query = QueryAssetReference(
        query_id="q-divergent",
        intent="divergent_rec",
        asset_id=ids["query"],
    )

    report = audit_query_gallery_eligibility(
        [query],
        [ids["same-product"], ids["candidate"]],
        catalog,
    )
    assert report.exact_or_near_duplicate_violations == {}
    assert report.missing_exact_match_positive == []
    assert report.divergent_product_exclusions == {
        query.query_id: [ids["same-product"]]
    }
    with pytest.raises(ValueError, match="q-divergent|gallery|eligible"):
        assert_query_gallery_eligible(report)

    filtered = audit_query_gallery_eligibility(
        [query],
        [ids["candidate"]],
        catalog,
    )
    assert filtered.exact_or_near_duplicate_violations == {}
    assert filtered.missing_exact_match_positive == []
    assert filtered.divergent_product_exclusions == {}
    assert_query_gallery_eligible(filtered)


def test_gallery_audit_fails_closed_for_unknown_asset_reference(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [("query", "diagonal", "product-1", 6)],
    )

    with pytest.raises(AssetNotCatalogedError):
        audit_query_gallery_eligibility(
            [
                QueryAssetReference(
                    query_id="q-missing",
                    intent="exact_match",
                    asset_id="asset.missing",
                )
            ],
            [ids["query"]],
            catalog,
        )


def test_gallery_audit_rejects_same_source_record_even_when_pixels_differ(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    _write_pattern(asset_root / "query.png", "diagonal")
    _write_pattern(asset_root / "gallery.png", "checker")
    query_asset = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="mutable-source",
            source_revision="revision-a",
            source_record_id="record-7",
            local_path="query.png",
            license_id="test-only",
        ),
        asset_root,
    )
    gallery_asset = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="mutable-source",
            source_revision="revision-b",
            source_record_id="record-7",
            local_path="gallery.png",
            license_id="test-only",
        ),
        asset_root,
    )
    catalog_dir = tmp_path / "catalog"
    publish_asset_catalog(
        [query_asset, gallery_asset],
        catalog_dir,
        asset_root,
        NearDuplicatePolicy(
            policy_version="source-record-gate-test-v1",
            max_phash_hamming_distance=0,
        ),
        coverage_roots=["."],
    )
    catalog = load_asset_catalog(catalog_dir, asset_root)

    report = audit_query_gallery_eligibility(
        [
            QueryAssetReference(
                query_id="q-source-record",
                intent="multi_product",
                asset_id=query_asset.asset_id,
            )
        ],
        [gallery_asset.asset_id],
        catalog,
    )

    assert report.exact_or_near_duplicate_violations == {
        "q-source-record": [gallery_asset.asset_id]
    }
    with pytest.raises(ValueError):
        assert_query_gallery_eligible(report)


def test_gallery_audit_rejects_transitive_derivation_lineage(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    for name, pattern in (
        ("parent", "diagonal"),
        ("middle", "checker"),
        ("child", "vertical"),
    ):
        _write_pattern(asset_root / f"{name}.png", pattern)
    parent = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="lineage-source",
            source_revision="revision-a",
            source_record_id="parent",
            local_path="parent.png",
            license_id="test-only",
        ),
        asset_root,
    )
    middle = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="lineage-source",
            source_revision="revision-a",
            source_record_id="middle",
            local_path="middle.png",
            derivation_parent_asset_ids=[parent.asset_id],
            license_id="test-only",
        ),
        asset_root,
    )
    child = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="lineage-source",
            source_revision="revision-a",
            source_record_id="child",
            local_path="child.png",
            derivation_parent_asset_ids=[middle.asset_id],
            license_id="test-only",
        ),
        asset_root,
    )
    catalog_dir = tmp_path / "catalog"
    publish_asset_catalog(
        [parent, middle, child],
        catalog_dir,
        asset_root,
        NearDuplicatePolicy(
            policy_version="lineage-gate-test-v1",
            max_phash_hamming_distance=0,
        ),
        coverage_roots=["."],
    )
    catalog = load_asset_catalog(catalog_dir, asset_root)

    report = audit_query_gallery_eligibility(
        [
            QueryAssetReference(
                query_id="q-lineage",
                intent="multi_product",
                asset_id=parent.asset_id,
            )
        ],
        [child.asset_id],
        catalog,
    )

    assert report.exact_or_near_duplicate_violations == {"q-lineage": [child.asset_id]}
    with pytest.raises(ValueError):
        assert_query_gallery_eligible(report)


def test_gallery_audit_closes_mixed_source_and_derivation_relations(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    for name, pattern in (
        ("query", "diagonal"),
        ("bridge", "checker"),
        ("gallery", "vertical"),
    ):
        _write_pattern(asset_root / f"{name}.png", pattern)
    query_asset = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="mixed-source",
            source_revision="revision-a",
            source_record_id="shared-record",
            local_path="query.png",
            license_id="test-only",
        ),
        asset_root,
    )
    bridge = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="mixed-source",
            source_revision="revision-b",
            source_record_id="shared-record",
            local_path="bridge.png",
            license_id="test-only",
        ),
        asset_root,
    )
    gallery_asset = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="other-source",
            source_revision="revision-a",
            source_record_id="gallery-record",
            local_path="gallery.png",
            derivation_parent_asset_ids=[bridge.asset_id],
            license_id="test-only",
        ),
        asset_root,
    )
    catalog_dir = tmp_path / "catalog"
    publish_asset_catalog(
        [query_asset, bridge, gallery_asset],
        catalog_dir,
        asset_root,
        NearDuplicatePolicy(
            policy_version="mixed-closure-gate-test-v1",
            max_phash_hamming_distance=0,
        ),
        coverage_roots=["."],
    )
    catalog = load_asset_catalog(catalog_dir, asset_root)

    report = audit_query_gallery_eligibility(
        [
            QueryAssetReference(
                query_id="q-mixed-closure",
                intent="multi_product",
                asset_id=query_asset.asset_id,
            )
        ],
        [gallery_asset.asset_id],
        catalog,
    )

    assert report.exact_or_near_duplicate_violations == {
        "q-mixed-closure": [gallery_asset.asset_id]
    }
    with pytest.raises(ValueError):
        assert_query_gallery_eligible(report)


def test_divergent_recommendation_requires_product_identity(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [
            ("query", "diagonal", None, 6),
            ("candidate", "checker", "product-2", 6),
        ],
    )
    query = QueryAssetReference(
        query_id="q-divergent-no-product",
        intent="divergent_rec",
        asset_id=ids["query"],
    )

    report = audit_query_gallery_eligibility(
        [query],
        [ids["candidate"]],
        catalog,
    )

    assert report.missing_divergent_product_identity == [query.query_id]
    assert report.violation_count == 1
    with pytest.raises(ValueError, match="missing_divergent_product_identity"):
        assert_query_gallery_eligible(report)


def test_divergent_recommendation_rejects_gallery_with_unknown_product(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [
            ("query", "diagonal", "product-1", 6),
            ("unknown", "checker", None, 6),
            ("candidate", "vertical", "product-2", 6),
        ],
    )
    query = QueryAssetReference(
        query_id="q-divergent-unknown-gallery",
        intent="divergent_rec",
        asset_id=ids["query"],
    )

    report = audit_query_gallery_eligibility(
        [query],
        [ids["unknown"], ids["candidate"]],
        catalog,
    )

    assert report.divergent_unknown_product_exclusions == {
        query.query_id: [ids["unknown"]]
    }
    with pytest.raises(ValueError, match="divergent_unknown_product"):
        assert_query_gallery_eligible(report)


def test_gallery_audit_rechecks_asset_bytes_after_metadata_only_load(tmp_path):
    catalog, ids = _build_catalog(
        tmp_path,
        [
            ("query", "diagonal", "product-1", 6),
            ("candidate", "checker", "product-2", 6),
        ],
    )
    metadata_only = load_asset_catalog(
        catalog.root,
        catalog.asset_root,
        verify_files=False,
    )
    (catalog.asset_root / "query.png").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="verify_files|verified"):
        audit_query_gallery_eligibility(
            [
                QueryAssetReference(
                    query_id="q-recheck",
                    intent="multi_product",
                    asset_id=ids["query"],
                )
            ],
            [ids["candidate"]],
            metadata_only,
        )
    with pytest.raises(ValueError, match="SHA|pHash|hash|摘要|解码"):
        audit_query_gallery_eligibility(
            [
                QueryAssetReference(
                    query_id="q-recheck-verified",
                    intent="multi_product",
                    asset_id=ids["query"],
                )
            ],
            [ids["candidate"]],
            catalog,
        )
