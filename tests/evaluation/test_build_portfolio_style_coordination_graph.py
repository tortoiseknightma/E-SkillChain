from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
from pathlib import Path
import tarfile
from typing import Any

from PIL import Image
import pytest

from scripts import build_portfolio_style_coordination_graph as graph_builder
from skillchain.evaluation import portfolio_core_runtime_sources as core_sources
from skillchain.tools.portfolio_runtime import (
    PortfolioRuntimeError,
    PortfolioRuntimeSources,
    _PortfolioMetadataIndex,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    sha256_bytes,
)


def _image(path: Path, color: tuple[int, int, int]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    Image.new("RGB", (24, 32), color).save(buffer, format="PNG")
    content = buffer.getvalue()
    path.write_bytes(content)
    return sha256_bytes(content), len(content)


def _asset_id(identity: str) -> str:
    return f"asset.v2.{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _selection(
    *,
    path: str,
    product_id: str,
    source_dataset: str,
    source_record_id: str,
    sha256: str,
    size: int,
) -> dict[str, Any]:
    return {
        "capability_bindings": [],
        "destination_path": path,
        "draft": {
            "product_id": product_id,
            "source_dataset": source_dataset,
            "source_record_id": source_record_id,
        },
        "expected_bytes": size,
        "expected_sha256": sha256,
    }


def _catalog(
    *,
    path: str,
    product_id: str,
    source_dataset: str,
    source_record_id: str,
    sha256: str,
) -> dict[str, Any]:
    return {
        "asset_id": _asset_id(path),
        "local_path": path,
        "product_id": product_id,
        "sha256": sha256,
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
    }


def _candidate(
    *,
    path: str,
    item_id: str,
    image_id: str,
    sha256: str,
    category: str,
    audience: str,
    colors: list[str],
    tags: list[str],
    eligibility: str,
    product_type: str,
    compatible: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "asset_id": _asset_id(path),
        "audience": audience,
        "category_l1": category,
        "color_families": colors,
        "confidence": 0.9 if eligibility == "neutral_universal" else 0.86,
        "display_title": f"Reviewed {item_id}",
        "eligibility_mode": eligibility,
        "feature_tags": tags,
        "image_path": path,
        "image_sha256": sha256,
        "product_id": f"abo:{item_id}",
        "source_metadata": {
            "archive_member": "listings/metadata/listings_a.json.gz",
            "item_id": item_id,
            "main_image_id": image_id,
            "product_type": product_type,
        },
        "source_record_id": f"listing:{item_id}/image:{image_id}",
    }
    if compatible is not None:
        value["compatible_anchor_color_families"] = compatible
    return value


def _write_archive(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = gzip.compress(canonical_jsonl_bytes(rows), mtime=0)
    member = tarfile.TarInfo("listings/metadata/listings_a.json.gz")
    member.size = len(payload)
    member.mode = 0o644
    member.mtime = 0
    with tarfile.open(path, mode="w") as archive:
        archive.addfile(member, BytesIO(payload))
    return sha256_bytes(path.read_bytes())


def _write_seed(path: Path, seed: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(seed))


def _fixture(tmp_path: Path) -> dict[str, Any]:
    asset_root = tmp_path / "assets"
    selections: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []

    assets = [
        (
            "query_images/style/blue-dress.png",
            "fashioniq:BLUE",
            "fashioniq",
            "dress:BLUE",
            (20, 80, 220),
        ),
        (
            "query_images/style/red-dress.png",
            "fashioniq:RED",
            "fashioniq",
            "dress:RED",
            (220, 30, 30),
        ),
        # A valid FashionIQ non-dress selection proves that anchors are
        # category-scoped without consulting query or split artifacts.
        (
            "query_images/style/blue-shirt.png",
            "fashioniq:SHIRT",
            "fashioniq",
            "shirt:SHIRT",
            (21, 80, 220),
        ),
        (
            "query_images/exact/black-bag.png",
            "abo:BLACKBAG",
            "abo",
            "listing:BLACKBAG/image:IMG-BAG",
            (15, 15, 15),
        ),
        (
            "query_images/exact/turquoise-sandal.png",
            "abo:TURQSANDAL",
            "abo",
            "listing:TURQSANDAL/image:IMG-SANDAL",
            (20, 180, 180),
        ),
        (
            "query_images/exact/mens-loafer.png",
            "abo:MENSLOAFER",
            "abo",
            "listing:MENSLOAFER/image:IMG-LOAFER",
            (16, 15, 15),
        ),
    ]
    hashes: dict[str, str] = {}
    for path, product_id, source_dataset, source_record_id, color in assets:
        digest, size = _image(asset_root / Path(path), color)
        hashes[path] = digest
        selections.append(
            _selection(
                path=path,
                product_id=product_id,
                source_dataset=source_dataset,
                source_record_id=source_record_id,
                sha256=digest,
                size=size,
            )
        )
        catalog.append(
            _catalog(
                path=path,
                product_id=product_id,
                source_dataset=source_dataset,
                source_record_id=source_record_id,
                sha256=digest,
            )
        )

    candidates = [
        _candidate(
            path="query_images/exact/black-bag.png",
            item_id="BLACKBAG",
            image_id="IMG-BAG",
            sha256=hashes["query_images/exact/black-bag.png"],
            category="bag",
            audience="women",
            colors=["black"],
            tags=["camera_bag", "crossbody", "small"],
            eligibility="neutral_universal",
            product_type="HANDBAG",
        ),
        _candidate(
            path="query_images/exact/turquoise-sandal.png",
            item_id="TURQSANDAL",
            image_id="IMG-SANDAL",
            sha256=hashes["query_images/exact/turquoise-sandal.png"],
            category="footwear",
            audience="women",
            colors=["turquoise"],
            tags=["sandal", "flat"],
            eligibility="compatible_palette",
            compatible=["turquoise", "blue", "green"],
            product_type="SANDAL",
        ),
        _candidate(
            path="query_images/exact/mens-loafer.png",
            item_id="MENSLOAFER",
            image_id="IMG-LOAFER",
            sha256=hashes["query_images/exact/mens-loafer.png"],
            category="footwear",
            audience="men",
            colors=["black"],
            tags=["loafer", "slip_on", "low_heel"],
            eligibility="neutral_universal",
            product_type="SHOES",
        ),
    ]
    listing_rows = [
        {
            "item_id": candidate["source_metadata"]["item_id"],
            "main_image_id": candidate["source_metadata"]["main_image_id"],
            "product_type": [{"value": candidate["source_metadata"]["product_type"]}],
        }
        for candidate in candidates
    ]
    archive_path = tmp_path / "abo-listings.tar"
    archive_sha256 = _write_archive(archive_path, listing_rows)
    seed = {
        "abo_listings_archive_sha256": archive_sha256,
        "candidates": candidates,
        "kind": graph_builder.SEED_KIND,
        "policy_version": "portfolio-style-coordination-candidate-review-v1",
        "review_boundary": {
            "coordination_claim": "fixture visual review",
            "native_outfit_cooccurrence_claimed": False,
            "review_method": "fixture exact-image review",
        },
        "schema_version": 1,
    }

    selection_path = tmp_path / "selection-manifest.json"
    selection_path.write_bytes(canonical_json_bytes({"selections": selections}))
    catalog_path = tmp_path / "runtime-assets.jsonl"
    catalog_path.write_bytes(canonical_jsonl_bytes(catalog))
    seed_path = tmp_path / "candidate-seed.json"
    _write_seed(seed_path, seed)
    return {
        "selection_manifest_path": selection_path,
        "runtime_catalog_assets_path": catalog_path,
        "asset_root": asset_root,
        "abo_listings_tar_path": archive_path,
        "candidate_seed_path": seed_path,
        "seed": seed,
        "listing_rows": listing_rows,
    }


def _build(inputs: dict[str, Any]) -> dict[str, Any]:
    return graph_builder.build_graph(
        selection_manifest_path=inputs["selection_manifest_path"],
        runtime_catalog_assets_path=inputs["runtime_catalog_assets_path"],
        asset_root=inputs["asset_root"],
        abo_listings_tar_path=inputs["abo_listings_tar_path"],
        candidate_seed_path=inputs["candidate_seed_path"],
    )


def _runtime_index_for_graph(
    tmp_path: Path,
    inputs: dict[str, Any],
    graph: dict[str, Any],
) -> _PortfolioMetadataIndex:
    graph_path = tmp_path / "style-coordination-graph.json"
    graph_path.write_bytes(canonical_json_bytes(graph))
    empty_jsonl_paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("dataset-assets", "rpc-scenes", "inaturalist", "recipes")
    }
    for path in empty_jsonl_paths.values():
        path.write_bytes(canonical_jsonl_bytes([]))
    return _PortfolioMetadataIndex(
        PortfolioRuntimeSources(
            selection_manifest=inputs["selection_manifest_path"],
            dataset_assets=empty_jsonl_paths["dataset-assets"],
            runtime_catalog_assets=inputs["runtime_catalog_assets_path"],
            rpc_scenes=empty_jsonl_paths["rpc-scenes"],
            inaturalist_manifest=empty_jsonl_paths["inaturalist"],
            recipe_evidence=empty_jsonl_paths["recipes"],
            asset_root=inputs["asset_root"],
            style_coordination_graph=graph_path,
        )
    )


def _with_fresh_graph_hash(graph: dict[str, Any]) -> dict[str, Any]:
    unsigned = deepcopy(graph)
    unsigned.pop("graph_sha256")
    return {
        **unsigned,
        "graph_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }


def _field_names(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(
            *(_field_names(child) for child in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_field_names(child) for child in value), set())
    return set()


def test_graph_is_deterministic_query_blind_self_hashed_and_create_only(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)

    first = _build(inputs)
    second = _build(inputs)

    assert first == second
    assert first["kind"] == graph_builder.GRAPH_KIND
    assert len(first["edges"]) == 3
    assert {edge["anchor"]["source_record_id"] for edge in first["edges"]} == {
        "dress:BLUE",
        "dress:RED",
    }
    assert all(edge["anchor"]["category_l1"] == "dress" for edge in first["edges"])
    assert all(edge["candidate"]["category_l1"] != "dress" for edge in first["edges"])
    assert all(edge["candidate"]["audience"] != "men" for edge in first["edges"])
    turquoise_edges = [
        edge
        for edge in first["edges"]
        if edge["candidate"]["product_id"] == "abo:TURQSANDAL"
    ]
    assert len(turquoise_edges) == 1
    assert turquoise_edges[0]["anchor"]["source_record_id"] == "dress:BLUE"
    assert turquoise_edges[0]["facets"][-1]["value"] == "compatible_palette_rule"
    assert not (_field_names(first) & graph_builder.FORBIDDEN_FIELD_NAMES)

    unsigned = dict(first)
    observed_sha256 = unsigned.pop("graph_sha256")
    assert observed_sha256 == sha256_bytes(canonical_json_bytes(unsigned))

    output = tmp_path / "published" / "coordination-graph.json"
    published = graph_builder.publish_graph(
        selection_manifest_path=inputs["selection_manifest_path"],
        runtime_catalog_assets_path=inputs["runtime_catalog_assets_path"],
        asset_root=inputs["asset_root"],
        abo_listings_tar_path=inputs["abo_listings_tar_path"],
        candidate_seed_path=inputs["candidate_seed_path"],
        output_path=output,
    )
    assert published == first
    assert parse_canonical_json(output.read_bytes(), label="published graph") == first
    with pytest.raises(FileExistsError):
        graph_builder.publish_graph(
            selection_manifest_path=inputs["selection_manifest_path"],
            runtime_catalog_assets_path=inputs["runtime_catalog_assets_path"],
            asset_root=inputs["asset_root"],
            abo_listings_tar_path=inputs["abo_listings_tar_path"],
            candidate_seed_path=inputs["candidate_seed_path"],
            output_path=output,
        )


def test_built_graph_is_accepted_by_core_validator_and_runtime_loader(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    graph = _build(inputs)

    core_sources._validate_style_coordination_graph(graph)
    index = _runtime_index_for_graph(tmp_path, inputs, graph)

    assert sum(map(len, index.style_coordination_edges.values())) == len(graph["edges"])
    assert set(index.style_coordination_edges) == {
        edge["anchor"]["image_sha256"] for edge in graph["edges"]
    }


def test_built_graph_annotation_policy_drift_fails_closed_in_both_consumers(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    graph = _build(inputs)
    graph["edges"][0]["annotation_policy_version"] = "drifted-review-policy-v1"
    graph = _with_fresh_graph_hash(graph)

    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="annotation policy is invalid",
    ):
        core_sources._validate_style_coordination_graph(graph)
    with pytest.raises(PortfolioRuntimeError, match="edge 1 identity is invalid"):
        _runtime_index_for_graph(tmp_path, inputs, graph)


def test_candidate_seed_policy_drift_fails_before_graph_build(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["seed"]["policy_version"] = "drifted-review-policy-v1"
    _write_seed(inputs["candidate_seed_path"], inputs["seed"])

    with pytest.raises(
        graph_builder.StyleCoordinationGraphError,
        match="unsupported coordination candidate seed policy_version",
    ):
        _build(inputs)


def test_built_graph_catalog_binding_drift_fails_closed_in_runtime_loader(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    graph = _build(inputs)
    graph["source_bindings"]["runtime_catalog_assets_sha256"] = "f" * 64
    graph = _with_fresh_graph_hash(graph)

    # The portable Core schema validator validates binding shape and self-hash;
    # the runtime loader owns the exact comparison with the selected source bytes.
    core_sources._validate_style_coordination_graph(graph)
    with pytest.raises(
        PortfolioRuntimeError,
        match="source bindings differ from runtime sources",
    ):
        _runtime_index_for_graph(tmp_path, inputs, graph)


@pytest.mark.parametrize("forbidden", sorted(graph_builder.FORBIDDEN_FIELD_NAMES))
def test_seed_rejects_every_forbidden_field(tmp_path: Path, forbidden: str) -> None:
    inputs = _fixture(tmp_path)
    inputs["seed"]["candidates"][0][forbidden] = "forbidden"
    _write_seed(inputs["candidate_seed_path"], inputs["seed"])

    with pytest.raises(
        graph_builder.StyleCoordinationGraphError, match="forbidden field"
    ):
        _build(inputs)


@pytest.mark.parametrize("drift", ["unknown_path", "image_sha256"])
def test_unknown_or_drifted_candidate_fails_closed(tmp_path: Path, drift: str) -> None:
    inputs = _fixture(tmp_path)
    candidate = inputs["seed"]["candidates"][0]
    if drift == "unknown_path":
        candidate["image_path"] = "query_images/exact/unknown.png"
    else:
        candidate["image_sha256"] = "f" * 64
    _write_seed(inputs["candidate_seed_path"], inputs["seed"])

    with pytest.raises(
        graph_builder.StyleCoordinationGraphError,
        match="unknown selection/catalog image|candidate selection/catalog binding drift",
    ):
        _build(inputs)


def test_same_category_candidate_is_rejected(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["seed"]["candidates"][0]["category_l1"] = "dress"
    _write_seed(inputs["candidate_seed_path"], inputs["seed"])

    with pytest.raises(
        graph_builder.StyleCoordinationGraphError, match="same/unsupported category"
    ):
        _build(inputs)


def test_duplicate_candidate_is_rejected(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["seed"]["candidates"].append(deepcopy(inputs["seed"]["candidates"][0]))
    _write_seed(inputs["candidate_seed_path"], inputs["seed"])

    with pytest.raises(
        graph_builder.StyleCoordinationGraphError, match="duplicate asset/product/path"
    ):
        _build(inputs)


def test_missing_bound_image_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    (inputs["asset_root"] / "query_images/exact/black-bag.png").unlink()

    with pytest.raises(
        graph_builder.StyleCoordinationGraphError, match="image .*cannot be inspected"
    ):
        _build(inputs)


def test_abo_source_metadata_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["listing_rows"][0]["product_type"] = [{"value": "WRONG"}]
    archive_sha256 = _write_archive(
        inputs["abo_listings_tar_path"],
        inputs["listing_rows"],
    )
    inputs["seed"]["abo_listings_archive_sha256"] = archive_sha256
    _write_seed(inputs["candidate_seed_path"], inputs["seed"])

    with pytest.raises(
        graph_builder.StyleCoordinationGraphError, match="source metadata mismatch"
    ):
        _build(inputs)
