from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from skillchain.evaluation import portfolio_core_runtime_sources as core_sources
from skillchain.evaluation.portfolio_inputs import (
    CORE_PORTFOLIO_PROCESSOR_ORDER,
    ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
)
from skillchain.tools.portfolio_runtime import build_portfolio_tool_runtime
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_core_runtime_sources_accept_forward_role_processor_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = SimpleNamespace(
        remote_runtimes=tuple(
            SimpleNamespace(processor=processor)
            for processor in ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER
        )
    )
    monkeypatch.setattr(
        core_sources,
        "require_verified_portfolio_core_inputs",
        lambda candidate: candidate,
    )

    assert core_sources._require_core_inputs(value) is value


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    asset_root = tmp_path / "asset-root"
    selection_path = asset_root / "core" / "selection-manifest.json"
    dataset_assets_path = asset_root / "core" / "dataset-assets.jsonl"
    catalog_dir = tmp_path / "runtime-catalog"
    catalog_assets_path = catalog_dir / "assets.jsonl"
    rpc_image_content = b"selected-rpc-image-bytes"
    _write(
        asset_root / "query_images" / "multi_product" / "rpc-core-1.jpg",
        rpc_image_content,
    )
    rpc_image_sha256 = sha256_bytes(rpc_image_content)
    style_anchor_content = b"selected-style-anchor-image-bytes"
    style_anchor_path = "query_images/divergent_rec/fashioniq-anchor.jpg"
    _write(asset_root / style_anchor_path, style_anchor_content)
    style_anchor_sha256 = sha256_bytes(style_anchor_content)
    coordination_candidate_content = b"selected-coordination-candidate-image-bytes"
    coordination_candidate_path = "query_images/exact/abo-candidate.jpg"
    _write(asset_root / coordination_candidate_path, coordination_candidate_content)
    coordination_candidate_sha256 = sha256_bytes(coordination_candidate_content)

    inaturalist_rows = (
        {
            "api_photo_url": "https://example.test/square.jpg",
            "attribution": "CC0",
            "common_name": "Test flower",
            "iconic_taxon": "Plantae",
            "image": "inat-11.jpg",
            "license": "cc0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "observation_id": 10,
            "observation_url": "https://example.test/observations/10",
            "original_url": "https://example.test/original.jpg",
            "photo_id": 11,
            "scientific_name": "Planta testensis",
            "source_url": "https://example.test/medium.jpg",
        },
    )
    # Frozen public-source manifests are exact-hash-bound strict JSONL, but
    # they are not required to use the repository's canonical whitespace.
    inaturalist_content = (
        json.dumps(inaturalist_rows[0], ensure_ascii=False) + "\n"
    ).encode("utf-8")
    inaturalist_path = _write(
        tmp_path / "inaturalist" / "manifest.jsonl", inaturalist_content
    )
    inaturalist_sha256 = sha256_bytes(inaturalist_content)

    recipe_rows = (
        {
            "author": "test-author",
            "category": "Biryani",
            "dish": "biryani",
            "license_id": "LicenseRef-Test",
            "name": "Test biryani",
            "recipeIngredient": ["rice", "spice"],
            "recipeInstructions": ["cook"],
            "source_archive_sha256": "9" * 64,
            "source_record_id": "recipes.json:1",
            "source_uri": "https://example.test/recipe/1",
        },
    )
    recipe_content = (json.dumps(recipe_rows[0], ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    recipe_path = _write(tmp_path / "recipes.jsonl", recipe_content)
    caption_content = canonical_json_bytes([])
    caption_path = _write(
        tmp_path / "captions" / "cap.dress.test.json", caption_content
    )

    selection = {
        "dataset_asset_count": 5,
        "dataset_assets_sha256": "pending",
        "formal_eligible": False,
        "formal_status": "non_formal",
        "policy_version": "portfolio-core-asset-preparation-v1",
        "schema_version": 1,
        "selections": [
            {
                "candidate_id": "rpc.core.1",
                "candidate_sha256": "1" * 64,
                "capability_bindings": [
                    {
                        "canonical_capability": "product.multi_search",
                        "canonical_intent": "multi_product",
                    }
                ],
                "destination_path": "query_images/multi_product/rpc-core-1.jpg",
                "draft": {
                    "cloud_upload_allowed": False,
                    "local_path": "query_images/multi_product/rpc-core-1.jpg",
                    "public_demo_allowed": False,
                    "source_dataset": "rpc",
                    "source_record_id": "val2019:7",
                    "source_revision": "rpc-revision-v1",
                },
                "expected_sha256": rpc_image_sha256,
                "source_id": "rpc",
            },
            {
                "candidate_id": "inat.core.1",
                "candidate_sha256": "2" * 64,
                "capability_bindings": [
                    {
                        "canonical_capability": "knowledge.visual_encyclopedia",
                        "canonical_intent": "encyclopedia",
                    }
                ],
                "destination_path": "query_images/encyclopedia/inat-core-1.jpg",
                "draft": {
                    "cloud_upload_allowed": False,
                    "local_path": "query_images/encyclopedia/inat-core-1.jpg",
                    "public_demo_allowed": False,
                    "source_dataset": "inaturalist",
                    "source_record_id": "photo:11",
                    "source_revision": f"manifest-{inaturalist_sha256}",
                },
                "expected_sha256": "b" * 64,
                "source_id": "inaturalist",
            },
            {
                "candidate_id": "food.core.1",
                "candidate_sha256": "3" * 64,
                "capability_bindings": [
                    {
                        "canonical_capability": "utility.recipe_guidance",
                        "canonical_intent": "utility",
                    }
                ],
                "destination_path": "query_images/utility/food-core-1.jpg",
                "draft": {
                    "cloud_upload_allowed": False,
                    "local_path": "query_images/utility/food-core-1.jpg",
                    "public_demo_allowed": False,
                    "source_dataset": "isia_food500",
                    "source_record_id": "ISIA_Food500/images/Biryani/one.jpg",
                    "source_revision": "archive-" + "8" * 64,
                },
                "expected_sha256": "c" * 64,
                "source_id": "isia_food500",
            },
            {
                "candidate_id": "fashioniq.core.anchor",
                "candidate_sha256": "4" * 64,
                "capability_bindings": [
                    {
                        "canonical_capability": "product.style_recommendation",
                        "canonical_intent": "divergent_rec",
                    }
                ],
                "destination_path": style_anchor_path,
                "draft": {
                    "cloud_upload_allowed": False,
                    "local_path": style_anchor_path,
                    "product_id": "fashioniq:anchor-1",
                    "public_demo_allowed": False,
                    "source_dataset": "fashioniq",
                    "source_record_id": "dress:anchor-1",
                    "source_revision": "fashioniq-revision-v1",
                },
                "expected_sha256": style_anchor_sha256,
                "source_id": "fashioniq",
            },
            {
                "candidate_id": "abo.core.coordination-candidate",
                "candidate_sha256": "5" * 64,
                "capability_bindings": [
                    {
                        "canonical_capability": "product.exact_search",
                        "canonical_intent": "exact_search",
                    }
                ],
                "destination_path": coordination_candidate_path,
                "draft": {
                    "cloud_upload_allowed": False,
                    "local_path": coordination_candidate_path,
                    "product_id": "abo:candidate-1",
                    "public_demo_allowed": False,
                    "source_dataset": "abo",
                    "source_record_id": "abo-listing:candidate-1",
                    "source_revision": "abo-revision-v1",
                },
                "expected_sha256": coordination_candidate_sha256,
                "source_id": "abo",
            },
        ],
    }
    dataset_assets_content = canonical_jsonl_bytes(
        (
            {"asset": "rpc"},
            {"asset": "inat"},
            {"asset": "food"},
            {"asset": "fashioniq-anchor"},
            {"asset": "abo-coordination-candidate"},
        )
    )
    selection["dataset_assets_sha256"] = sha256_bytes(dataset_assets_content)
    selection_content = canonical_json_bytes(selection)
    _write(selection_path, selection_content)
    _write(dataset_assets_path, dataset_assets_content)
    catalog_assets_content = canonical_jsonl_bytes(
        (
            {"asset_id": "asset-rpc", "local_path": "rpc.jpg"},
            {"asset_id": "asset-inat", "local_path": "inat.jpg"},
            {"asset_id": "asset-food", "local_path": "food.jpg"},
            {
                "asset_id": "asset-style-anchor",
                "local_path": style_anchor_path,
                "product_id": "fashioniq:anchor-1",
                "sha256": style_anchor_sha256,
                "source_dataset": "fashioniq",
                "source_record_id": "dress:anchor-1",
            },
            {
                "asset_id": "asset-coordination-candidate",
                "local_path": coordination_candidate_path,
                "product_id": "abo:candidate-1",
                "sha256": coordination_candidate_sha256,
                "source_dataset": "abo",
                "source_record_id": "abo-listing:candidate-1",
            },
        )
    )
    _write(catalog_assets_path, catalog_assets_content)

    coordination_graph = {
        "edges": [
            {
                "anchor": {
                    "asset_id": "asset-style-anchor",
                    "category_l1": "dress",
                    "color_families": ["black"],
                    "image_path": style_anchor_path,
                    "image_sha256": style_anchor_sha256,
                    "product_id": "fashioniq:anchor-1",
                    "source_dataset": "fashioniq",
                    "source_record_id": "dress:anchor-1",
                },
                "annotation_policy_version": (
                    "portfolio-style-coordination-candidate-review-v1"
                ),
                "candidate": {
                    "asset_id": "asset-coordination-candidate",
                    "audience": "women",
                    "category_l1": "footwear",
                    "color_families": ["black"],
                    "display_title": "Black low-block-heel ankle boot",
                    "feature_tags": ["ankle_boot", "low_block_heel"],
                    "image_path": coordination_candidate_path,
                    "image_sha256": coordination_candidate_sha256,
                    "product_id": "abo:candidate-1",
                    "source_dataset": "abo",
                    "source_record_id": "abo-listing:candidate-1",
                },
                "confidence": 0.9,
                "edge_id": "pending",
                "facets": [
                    {
                        "confidence": 1.0,
                        "facet": "category",
                        "value": "footwear",
                    },
                    {"confidence": 0.95, "facet": "palette", "value": "black"},
                    {
                        "confidence": 0.95,
                        "facet": "verified_attributes",
                        "value": "ankle_boot,low_block_heel",
                    },
                    {
                        "confidence": 0.9,
                        "facet": "coordination_rule",
                        "value": "neutral_palette_rule",
                    },
                ],
                "relation_kind": "portfolio_curated_coordination_rule",
            }
        ],
        "kind": "portfolio-style-coordination-graph",
        "policy_version": "portfolio-style-coordination-graph-v1",
        "schema_version": 1,
        "source_bindings": {
            "abo_listings_archive_sha256": "6" * 64,
            "candidate_seed_sha256": "7" * 64,
            "runtime_catalog_assets_sha256": sha256_bytes(catalog_assets_content),
            "selection_manifest_sha256": sha256_bytes(selection_content),
        },
    }
    graph_edge = coordination_graph["edges"][0]
    graph_edge["edge_id"] = "style.edge.v1." + sha256_bytes(
        canonical_json_bytes(
            {
                "anchor_asset_id": graph_edge["anchor"]["asset_id"],
                "annotation_policy_version": graph_edge["annotation_policy_version"],
                "candidate_asset_id": graph_edge["candidate"]["asset_id"],
                "relation_kind": graph_edge["relation_kind"],
                "rule": "neutral_palette_rule",
            }
        )
    )
    coordination_graph["graph_sha256"] = sha256_bytes(
        canonical_json_bytes(coordination_graph)
    )
    coordination_graph_content = canonical_json_bytes(coordination_graph)
    coordination_graph_path = _write(
        tmp_path / "style-coordination-graph.json", coordination_graph_content
    )

    annotation = {
        "annotations": [
            {
                "bbox": [0, 0, 20, 20],
                "category_id": 1,
                "id": 100,
                "image_id": 7,
                "iscrowd": 0,
            },
            {
                "bbox": [25, 10, 30, 25],
                "category_id": 2,
                "id": 101,
                "image_id": 7,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "bottle"},
            {"id": 2, "name": "box"},
        ],
        "images": [{"file_name": "scene-7.jpg", "height": 100, "id": 7, "width": 100}],
    }
    annotation_content = json.dumps(
        annotation, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    rpc_archive = tmp_path / "rpc.zip"
    with zipfile.ZipFile(rpc_archive, "w") as archive:
        archive.writestr("instances_val2019.json", annotation_content)
        archive.writestr("unrelated.bin", b"aaaa")
    rpc_archive_content = rpc_archive.read_bytes()
    rpc_adapter = {
        "archives": [
            {
                "bytes": len(rpc_archive_content),
                "logical_path": "rpc/archive.zip",
                "sha256": sha256_bytes(rpc_archive_content),
            }
        ],
        "cross_intent_count": 0,
        "formal_eligible": False,
        "formal_status": "non_formal",
        "include_count": 1,
        "policy_version": "portfolio-core-selected-source-adapters-v1",
        "raw_mutation_performed": False,
        "reserve_count": 0,
        "schema_version": 1,
        "source_id": "rpc",
        "source_revision": "rpc-revision-v1",
        "track": "portfolio",
    }
    rpc_adapter_content = canonical_json_bytes(rpc_adapter)
    rpc_adapter_path = _write(tmp_path / "rpc-adapter.json", rpc_adapter_content)

    selection_sha256 = sha256_bytes(selection_content)
    dataset_assets_sha256 = sha256_bytes(dataset_assets_content)
    catalog_sha256 = "d" * 64
    receipt_file_sha256 = "e" * 64
    receipt = SimpleNamespace(
        base_selection_manifest_sha256=selection_sha256,
        base_dataset_assets_sha256=dataset_assets_sha256,
        output_catalog_sha256=catalog_sha256,
    )
    remote_runtimes = tuple(
        SimpleNamespace(
            processor=processor,
            receipt=receipt,
            receipt_file_sha256=receipt_file_sha256,
        )
        for processor in CORE_PORTFOLIO_PROCESSOR_ORDER
    )
    remote_files = SimpleNamespace(
        selection_manifest=selection_path,
        dataset_assets=dataset_assets_path,
        output_catalog_dir=catalog_dir,
    )
    private_asset_id = "private-core-asset"
    private_image_path = "private/core/image.jpg"
    public_input = canonical_json_bytes(
        {"asset_id": "asset-token-opaque", "text": "public text", "turns": []}
    ).decode("utf-8")
    verified = SimpleNamespace(
        files=SimpleNamespace(
            remote_files=remote_files,
            asset_root=asset_root,
        ),
        remote_runtimes=remote_runtimes,
        expected_output_catalog_sha256=catalog_sha256,
        expected_plan_sha256="f" * 64,
        expected_query_artifact_sha256="0" * 64,
        queries=(
            SimpleNamespace(asset_id=private_asset_id, image_path=private_image_path),
        ),
        assistant_queries=(SimpleNamespace(public_input_json=public_input),),
    )
    monkeypatch.setattr(
        core_sources,
        "require_verified_portfolio_core_inputs",
        lambda value: value,
    )
    auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        rpc_archive=rpc_archive,
        rpc_adapter_manifest=rpc_adapter_path,
        expected_rpc_adapter_manifest_sha256=sha256_bytes(rpc_adapter_content),
        expected_rpc_annotation_sha256=sha256_bytes(annotation_content),
        inaturalist_manifest=inaturalist_path,
        expected_inaturalist_manifest_sha256=inaturalist_sha256,
        recipe_evidence=recipe_path,
        expected_recipe_evidence_sha256=sha256_bytes(recipe_content),
        fashioniq_captions=(
            core_sources.BoundPortfolioRuntimeFile(
                path=caption_path,
                expected_sha256=sha256_bytes(caption_content),
            ),
        ),
        style_coordination_graph=core_sources.BoundPortfolioRuntimeFile(
            path=coordination_graph_path,
            expected_sha256=sha256_bytes(coordination_graph_content),
        ),
    )
    return verified, auxiliary


def _replace_coordination_graph(
    tmp_path: Path,
    auxiliary: core_sources.PortfolioCoreRuntimeAuxiliaryFiles,
    graph: dict[str, object],
    *,
    name: str,
) -> core_sources.PortfolioCoreRuntimeAuxiliaryFiles:
    content = canonical_json_bytes(graph)
    path = _write(tmp_path / name, content)
    return core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{
            **auxiliary.__dict__,
            "style_coordination_graph": core_sources.BoundPortfolioRuntimeFile(
                path=path,
                expected_sha256=sha256_bytes(content),
            ),
        }
    )


def test_materializes_core_runtime_sources_with_opaque_assistant_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "runtime-sources"

    result = core_sources.materialize_verified_portfolio_core_runtime_sources(
        verified, auxiliary, output_dir=output
    )

    assert result.provider_call_count == 0
    assert result.formal_eligible is False
    assert result.rpc_scene_count == 1
    assert result.inaturalist_row_count == 1
    assert result.recipe_row_count == 1
    assert result.sources.selection_manifest == (
        verified.files.remote_files.selection_manifest.absolute()
    )
    assert result.sources.runtime_catalog_assets == (
        verified.files.remote_files.output_catalog_dir.absolute() / "assets.jsonl"
    )
    assert result.sources.asset_root == verified.files.asset_root.absolute()
    assert result.sources.fashioniq_captions == (
        output / "fashioniq-captions" / "cap.dress.test.json",
    )
    assert result.sources.style_coordination_graph == (
        output / core_sources.CORE_RUNTIME_STYLE_COORDINATION_GRAPH
    )
    assert auxiliary.style_coordination_graph is not None
    assert result.source_file_sha256s[-1] == (
        auxiliary.style_coordination_graph.expected_sha256
    )

    rpc_scene = json.loads(result.sources.rpc_scenes.read_text(encoding="utf-8"))
    assert rpc_scene["source_record_id"] == "val2019:7"
    assert [item["category_name"] for item in rpc_scene["instances"]] == [
        "bottle",
        "box",
    ]
    inaturalist = json.loads(
        result.sources.inaturalist_manifest.read_text(encoding="utf-8")
    )
    assert inaturalist["image"] == "inat-core-1.jpg"
    assert inaturalist["source_image"] == "inat-11.jpg"

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["assistant_projection"] == {
        "field_allowlist": ["asset_id", "text", "turns"],
        "private_labels_exposed": False,
        "query_count_verified": 1,
    }
    assert receipt["auxiliary_inputs"]["rpc"]["archive_full_sha256_reverified"] is False
    assert receipt["auxiliary_inputs"]["rpc"]["selected_image_bytes_verified"] is True
    graph_record = receipt["auxiliary_inputs"]["style_coordination_graph"]
    assert graph_record["output_path"] == (
        core_sources.CORE_RUNTIME_STYLE_COORDINATION_GRAPH
    )
    assert graph_record["edge_count"] == 1
    assert (
        receipt["outputs"][core_sources.CORE_RUNTIME_STYLE_COORDINATION_GRAPH]["sha256"]
        == graph_record["sha256"]
    )
    assert receipt["provider_call_count"] == 0
    assert (
        core_sources.require_verified_portfolio_core_runtime_sources(result) is result
    )

    # The existing Portfolio runtime can consume the derived Core projections.
    runtime = build_portfolio_tool_runtime(result.sources)
    assert "val2019:7" in runtime.index.rpc_scenes
    assert runtime.index.inaturalist[0]["photo_id"] == 11
    assert runtime.index.runtime_data_sha256 == receipt["runtime_data_sha256"]


def test_runtime_source_publication_is_create_only_and_reload_is_digest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "runtime-sources"
    result = core_sources.materialize_verified_portfolio_core_runtime_sources(
        verified, auxiliary, output_dir=output
    )

    with pytest.raises(FileExistsError):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified, auxiliary, output_dir=output
        )
    original_digest = core_sources.stable_file_digest

    def reject_raw_archive(path, *, label):
        if Path(path).absolute() == auxiliary.rpc_archive.absolute():
            raise AssertionError("downstream load re-read the raw RPC archive")
        return original_digest(path, label=label)

    monkeypatch.setattr(core_sources, "stable_file_digest", reject_raw_archive)
    loaded = core_sources.load_verified_portfolio_core_runtime_sources(
        verified,
        output_dir=output,
        expected_receipt_file_sha256=result.receipt_file_sha256,
    )
    assert loaded.source_file_sha256s == result.source_file_sha256s
    assert loaded.sources.style_coordination_graph == (
        output / core_sources.CORE_RUNTIME_STYLE_COORDINATION_GRAPH
    )

    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="external digest mismatch",
    ):
        core_sources.load_verified_portfolio_core_runtime_sources(
            verified,
            output_dir=output,
            expected_receipt_file_sha256="1" * 64,
        )

    result.sources.rpc_scenes.write_bytes(canonical_jsonl_bytes(({"tampered": True},)))
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="differs from receipt",
    ):
        core_sources.load_verified_portfolio_core_runtime_sources(
            verified,
            output_dir=output,
            expected_receipt_file_sha256=result.receipt_file_sha256,
        )


def test_style_coordination_graph_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "runtime-sources"
    result = core_sources.materialize_verified_portfolio_core_runtime_sources(
        verified, auxiliary, output_dir=output
    )
    graph_path = result.sources.style_coordination_graph
    assert graph_path is not None
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["edges"][0]["candidate"]["display_title"] += " changed"
    graph.pop("graph_sha256")
    graph["graph_sha256"] = sha256_bytes(canonical_json_bytes(graph))
    graph_path.write_bytes(canonical_json_bytes(graph))

    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="differs from receipt",
    ):
        core_sources.load_verified_portfolio_core_runtime_sources(
            verified,
            output_dir=output,
            expected_receipt_file_sha256=result.receipt_file_sha256,
        )


def test_style_coordination_graph_forbidden_field_and_self_hash_fail_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    assert auxiliary.style_coordination_graph is not None
    original = json.loads(
        auxiliary.style_coordination_graph.path.read_text(encoding="utf-8")
    )

    wrong_digest_auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{
            **auxiliary.__dict__,
            "style_coordination_graph": core_sources.BoundPortfolioRuntimeFile(
                path=auxiliary.style_coordination_graph.path,
                expected_sha256="8" * 64,
            ),
        }
    )
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="style coordination graph SHA-256 mismatch",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            wrong_digest_auxiliary,
            output_dir=tmp_path / "wrong-digest-style-runtime",
        )

    forbidden = json.loads(json.dumps(original))
    forbidden["edges"][0]["candidate"]["query_text"] = "private query"
    forbidden.pop("graph_sha256")
    forbidden["graph_sha256"] = sha256_bytes(canonical_json_bytes(forbidden))
    forbidden_auxiliary = _replace_coordination_graph(
        tmp_path,
        auxiliary,
        forbidden,
        name="forbidden-style-graph.json",
    )
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="forbidden field: query_text",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            forbidden_auxiliary,
            output_dir=tmp_path / "forbidden-style-runtime",
        )

    invalid_self_hash = json.loads(json.dumps(original))
    invalid_self_hash["edges"][0]["confidence"] = 0.7
    invalid_self_hash_auxiliary = _replace_coordination_graph(
        tmp_path,
        auxiliary,
        invalid_self_hash,
        name="invalid-self-hash-style-graph.json",
    )
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="self hash is invalid",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            invalid_self_hash_auxiliary,
            output_dir=tmp_path / "invalid-self-hash-style-runtime",
        )


def test_style_coordination_graph_must_bind_verified_core_primary_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    assert auxiliary.style_coordination_graph is not None
    graph = json.loads(
        auxiliary.style_coordination_graph.path.read_text(encoding="utf-8")
    )
    graph["source_bindings"]["runtime_catalog_assets_sha256"] = "f" * 64
    graph.pop("graph_sha256")
    graph["graph_sha256"] = sha256_bytes(canonical_json_bytes(graph))
    drifted = _replace_coordination_graph(
        tmp_path,
        auxiliary,
        graph,
        name="drifted-source-binding-style-graph.json",
    )

    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="source bindings differ from verified Core inputs",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            drifted,
            output_dir=tmp_path / "drifted-source-binding-runtime",
        )


def test_style_coordination_graph_is_optional_for_legacy_runtime_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    legacy_auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{**auxiliary.__dict__, "style_coordination_graph": None}
    )
    output = tmp_path / "legacy-runtime-sources"
    result = core_sources.materialize_verified_portfolio_core_runtime_sources(
        verified,
        legacy_auxiliary,
        output_dir=output,
    )

    assert result.sources.style_coordination_graph is None
    assert not (output / core_sources.CORE_RUNTIME_STYLE_COORDINATION_GRAPH).exists()
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert "style_coordination_graph" not in receipt["auxiliary_inputs"]
    assert core_sources.CORE_RUNTIME_STYLE_COORDINATION_GRAPH not in receipt["outputs"]

    loaded = core_sources.load_verified_portfolio_core_runtime_sources(
        verified,
        output_dir=output,
        expected_receipt_file_sha256=result.receipt_file_sha256,
    )
    assert loaded.sources.style_coordination_graph is None
    assert loaded.source_file_sha256s == result.source_file_sha256s


def test_recipe_coverage_and_private_dto_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    wrong_recipe = canonical_jsonl_bytes(
        (
            {
                "category": "Other",
                "license_id": "LicenseRef-Test",
                "name": "Other recipe",
                "recipeIngredient": ["x"],
                "recipeInstructions": ["y"],
                "source_archive_sha256": "9" * 64,
                "source_record_id": "recipes.json:2",
                "source_uri": "https://example.test/recipe/2",
            },
        )
    )
    wrong_path = _write(tmp_path / "wrong-recipes.jsonl", wrong_recipe)
    wrong_auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{
            **auxiliary.__dict__,
            "recipe_evidence": wrong_path,
            "expected_recipe_evidence_sha256": sha256_bytes(wrong_recipe),
        }
    )
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="does not cover",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            wrong_auxiliary,
            output_dir=tmp_path / "wrong-runtime",
        )

    private_public = canonical_json_bytes(
        {
            "asset_id": "asset-token",
            "canonical_capability": "product.multi_search",
            "text": "request",
            "turns": [],
        }
    ).decode("utf-8")
    verified.assistant_queries = (  # type: ignore[misc]
        SimpleNamespace(public_input_json=private_public),
    )
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="non-opaque",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            auxiliary,
            output_dir=tmp_path / "private-runtime",
        )


def test_rpc_projection_ignores_unconsumed_archive_bytes_but_binds_annotation_and_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    with zipfile.ZipFile(auxiliary.rpc_archive) as archive:
        annotation = archive.read("instances_val2019.json")

    unrelated_changed = tmp_path / "rpc-unrelated-changed.zip"
    with zipfile.ZipFile(unrelated_changed, "w") as archive:
        archive.writestr("instances_val2019.json", annotation)
        archive.writestr("unrelated.bin", b"bbbb")
    assert unrelated_changed.stat().st_size == auxiliary.rpc_archive.stat().st_size
    changed_auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{**auxiliary.__dict__, "rpc_archive": unrelated_changed}
    )
    accepted = core_sources.materialize_verified_portfolio_core_runtime_sources(
        verified,
        changed_auxiliary,
        output_dir=tmp_path / "unrelated-byte-change",
    )
    assert accepted.rpc_scene_count == 1
    receipt = json.loads(accepted.receipt_path.read_text(encoding="utf-8"))
    assert (
        receipt["verification_boundary"]["rpc_full_archive_sha256_reverified"] is False
    )

    annotation_changed = tmp_path / "rpc-annotation-changed.zip"
    changed_annotation = annotation.replace(b'"bottle"', b'"cattle"')
    assert len(changed_annotation) == len(annotation)
    with zipfile.ZipFile(annotation_changed, "w") as archive:
        archive.writestr("instances_val2019.json", changed_annotation)
        archive.writestr("unrelated.bin", b"aaaa")
    assert annotation_changed.stat().st_size == auxiliary.rpc_archive.stat().st_size
    annotation_auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{**auxiliary.__dict__, "rpc_archive": annotation_changed}
    )
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="annotation member SHA-256 mismatch",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            annotation_auxiliary,
            output_dir=tmp_path / "annotation-byte-change",
        )

    selected_image = (
        verified.files.asset_root / "query_images" / "multi_product" / "rpc-core-1.jpg"
    )
    selected_image.write_bytes(b"changed-selected-rpc-image")
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="selected image SHA-256 mismatch",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            auxiliary,
            output_dir=tmp_path / "selected-image-change",
        )


def test_rpc_projection_accepts_hash_bound_extracted_annotation_without_opening_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    with zipfile.ZipFile(auxiliary.rpc_archive) as archive:
        annotation = archive.read("instances_val2019.json")
    annotation_path = _write(tmp_path / "instances_val2019.json", annotation)
    extracted_auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{**auxiliary.__dict__, "rpc_annotation_file": annotation_path}
    )

    def reject_archive_open(*args, **kwargs):
        raise AssertionError("hash-bound annotation path reopened the large archive")

    monkeypatch.setattr(core_sources.zipfile, "ZipFile", reject_archive_open)
    result = core_sources.materialize_verified_portfolio_core_runtime_sources(
        verified,
        extracted_auxiliary,
        output_dir=tmp_path / "extracted-annotation-runtime",
    )

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["auxiliary_inputs"]["rpc"]["annotation_source_mode"] == (
        "externally_hash_bound_extracted_annotation"
    )


def test_malformed_auxiliary_digest_fails_before_expensive_core_reverification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    malformed = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{**auxiliary.__dict__, "expected_rpc_annotation_sha256": "not-a-digest"}
    )

    def reject_core_reverification(value):
        raise AssertionError("malformed auxiliary reached Core reverification")

    monkeypatch.setattr(
        core_sources, "_require_core_inputs", reject_core_reverification
    )
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="RPC annotation SHA-256 must be lowercase SHA-256",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            malformed,
            output_dir=tmp_path / "malformed-auxiliary-runtime",
        )

    invalid_recipe = b"not-json\n"
    invalid_recipe_path = _write(tmp_path / "invalid-recipe.jsonl", invalid_recipe)
    invalid_file = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{
            **auxiliary.__dict__,
            "recipe_evidence": invalid_recipe_path,
            "expected_recipe_evidence_sha256": sha256_bytes(invalid_recipe),
        }
    )
    with pytest.raises(core_sources.PortfolioCoreRuntimeSourceError):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            invalid_file,
            output_dir=tmp_path / "invalid-auxiliary-file-runtime",
        )


def test_recipe_evidence_accepts_only_strict_objects_with_blank_separators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, auxiliary = _fixture(tmp_path, monkeypatch)
    original = auxiliary.recipe_evidence.read_bytes()
    blank_separated = original.replace(b"\n", b"\n \t\n")
    blank_path = _write(tmp_path / "blank-separated-recipes.jsonl", blank_separated)
    blank_auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{
            **auxiliary.__dict__,
            "recipe_evidence": blank_path,
            "expected_recipe_evidence_sha256": sha256_bytes(blank_separated),
        }
    )
    result = core_sources.materialize_verified_portfolio_core_runtime_sources(
        verified,
        blank_auxiliary,
        output_dir=tmp_path / "blank-separated-runtime",
    )
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert (
        receipt["auxiliary_inputs"]["recipe_evidence"]["ignored_blank_line_count"] == 1
    )

    duplicated = original + b"\n" + original
    duplicate_path = _write(tmp_path / "duplicate-recipes.jsonl", duplicated)
    duplicate_auxiliary = core_sources.PortfolioCoreRuntimeAuxiliaryFiles(
        **{
            **auxiliary.__dict__,
            "recipe_evidence": duplicate_path,
            "expected_recipe_evidence_sha256": sha256_bytes(duplicated),
        }
    )
    with pytest.raises(
        core_sources.PortfolioCoreRuntimeSourceError,
        match="source_record_id is duplicated",
    ):
        core_sources.materialize_verified_portfolio_core_runtime_sources(
            verified,
            duplicate_auxiliary,
            output_dir=tmp_path / "duplicate-recipe-runtime",
        )
