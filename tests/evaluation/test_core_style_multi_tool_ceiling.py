from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from scripts import build_core_style_multi_tool_ceiling as ceiling
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _query(
    ordinal: int,
    *,
    capability: str,
    image_path: str,
    text: str,
    split: str = "dev_mini",
) -> dict[str, object]:
    return {
        "query_id": f"q-{ordinal:04d}",
        "asset_id": f"asset-{ordinal:04d}",
        "canonical_capability": capability,
        "image_path": image_path,
        "text": text,
        "split": split,
    }


def _selection(
    image_path: str,
    *,
    source_dataset: str,
    source_record_id: str,
    product_id: str | None = None,
) -> dict[str, object]:
    return {
        "destination_path": image_path,
        "draft": {
            "source_dataset": source_dataset,
            "source_record_id": source_record_id,
            "product_id": product_id or f"product:{source_record_id}",
        },
    }


def _coordination_graph(
    *,
    selection_content: bytes,
    cross_queries: list[dict[str, object]],
    cross_selections: list[dict[str, object]],
) -> dict[str, object]:
    edges: list[dict[str, object]] = []
    for index, (query, selection) in enumerate(
        zip(cross_queries[:10], cross_selections[:10], strict=True)
    ):
        draft = selection["draft"]
        assert isinstance(draft, dict)
        candidate_specs = (
            (
                "footwear",
                "Reviewed black ankle boot",
                ["ankle_boot", "block_heel"],
                ["black"],
            ),
            (
                "footwear",
                "Reviewed turquoise flat sandal",
                ["sandal", "flat"],
                ["turquoise"],
            ),
            (
                "bag",
                "Reviewed cognac crossbody camera bag",
                ["camera_bag", "crossbody", "small"],
                ["cognac"],
            ),
            (
                "jewelry",
                "Reviewed silver drop earrings",
                ["earrings", "drop", "silver_tone"],
                ["silver"],
            ),
        )
        for candidate_index, (
            category,
            title,
            feature_tags,
            color_families,
        ) in enumerate(candidate_specs):
            edge_ordinal = index * len(candidate_specs) + candidate_index
            anchor = {
                "asset_id": query["asset_id"],
                "product_id": draft["product_id"],
                "source_dataset": "fashioniq",
                "source_record_id": draft["source_record_id"],
                "image_sha256": f"{index + 1:064x}",
                "image_path": selection["destination_path"],
                "category_l1": "dress",
                "color_families": ["blue"],
            }
            candidate = {
                "asset_id": f"candidate-asset-{index:02d}-{candidate_index}",
                "product_id": f"abo:candidate-{index:02d}-{candidate_index}",
                "source_dataset": "abo",
                "source_record_id": (
                    f"listing:CANDIDATE{index:02d}{candidate_index}/image:main"
                ),
                "image_sha256": f"{100 + edge_ordinal:064x}",
                "image_path": (
                    f"query_images/candidate-{index:02d}-{candidate_index}.jpg"
                ),
                "category_l1": category,
                "display_title": title,
                "audience": "women",
                "feature_tags": feature_tags,
                "color_families": color_families,
            }
            confidence = 0.9 - candidate_index * 0.01
            rule = "neutral_palette_rule"
            identity = {
                "anchor_asset_id": anchor["asset_id"],
                "annotation_policy_version": (
                    ceiling.STYLE_COORDINATION_ANNOTATION_POLICY
                ),
                "candidate_asset_id": candidate["asset_id"],
                "relation_kind": ceiling.STYLE_COORDINATION_RELATION,
                "rule": rule,
            }
            edge_id = "style.edge.v1." + sha256_bytes(canonical_json_bytes(identity))
            edges.append(
                {
                    "edge_id": edge_id,
                    "anchor": anchor,
                    "candidate": candidate,
                    "relation_kind": ceiling.STYLE_COORDINATION_RELATION,
                    "confidence": confidence,
                    "facets": [
                        {"facet": "category", "value": category, "confidence": 1.0},
                        {
                            "facet": "palette",
                            "value": ",".join(color_families),
                            "confidence": 0.95,
                        },
                        {
                            "facet": "verified_attributes",
                            "value": ",".join(feature_tags),
                            "confidence": 0.95,
                        },
                        {
                            "facet": "coordination_rule",
                            "value": rule,
                            "confidence": confidence,
                        },
                    ],
                    "annotation_policy_version": (
                        ceiling.STYLE_COORDINATION_ANNOTATION_POLICY
                    ),
                }
            )
    edges.sort(key=lambda edge: str(edge["edge_id"]))
    unsigned: dict[str, object] = {
        "kind": ceiling.STYLE_COORDINATION_GRAPH_KIND,
        "schema_version": 1,
        "policy_version": ceiling.STYLE_COORDINATION_GRAPH_POLICY,
        "source_bindings": {
            "selection_manifest_sha256": sha256_bytes(selection_content),
            "runtime_catalog_assets_sha256": "a" * 64,
            "abo_listings_archive_sha256": "b" * 64,
            "candidate_seed_sha256": "c" * 64,
        },
        "edges": edges,
    }
    return {
        **unsigned,
        "graph_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }


def _resign_coordination_graph(graph: dict[str, object]) -> None:
    unsigned = {key: value for key, value in graph.items() if key != "graph_sha256"}
    graph["graph_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))


def _fixture(tmp_path: Path) -> dict[str, Path]:
    queries: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    captions_dir = tmp_path / "captions"
    images_dir = tmp_path / "images"
    captions_dir.mkdir()

    dress_rows: list[dict[str, object]] = []
    cross_queries: list[dict[str, object]] = []
    cross_selections: list[dict[str, object]] = []
    ordinal = 1
    for index in range(20):
        anchor_id = f"anchor-{index:02d}"
        target_id = f"target-{index:02d}"
        image_path = f"query_images/style/style-{index:02d}.jpg"
        same_category_text = {
            0: "\u627e\u7c7b\u4f3c\u914d\u8272\u7684\u8fde\u8863\u88d9",
            1: ("\u627e\u51e0\u4ef6\u76f8\u8fd1\u9020\u578b\u7684\u88d9\u5b50"),
        }.get(index, "find a similar dress")
        queries.append(
            _query(
                ordinal,
                capability="product.style_recommendation",
                image_path=image_path,
                text=same_category_text,
            )
        )
        selections.append(
            _selection(
                image_path,
                source_dataset="fashioniq",
                source_record_id=f"dress:{anchor_id}",
            )
        )
        dress_rows.append(
            {
                "candidate": anchor_id,
                "target": target_id,
                "captions": ["shorter sleeves", "darker blue"],
            }
        )
        target_path = images_dir / "dress" / f"{target_id}.jpg"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(f"image:{target_id}".encode())
        ordinal += 1

    for index in range(20):
        image_path = f"query_images/style/cross-{index:02d}.jpg"
        text = {
            0: "\u683c\u7eb9\u88d9\u548b\u914d",
            1: ("\u4e0d\u914d\u978b\u4e86\uff1b\u6539\u642d\u659c\u630e\u5305"),
            2: "\u7ed9\u8fd9\u6761\u88d9\u5b50\u642d\u767d\u8272\u978b",
            3: ("\u7ed9\u8fd9\u6761\u88d9\u5b50\u642d\u5e73\u5e95\u51c9\u978b"),
            11: ("\u7ed9\u8fd9\u6761\u683c\u7eb9\u88d9\u914d\u5916\u5957"),
            12: "\u683c\u7eb9\u88d9\u548b\u914d",
        }.get(index, "\u600e\u4e48\u642d\u914d\u978b\u5b50")
        cross_query = _query(
            ordinal,
            capability="product.style_recommendation",
            image_path=image_path,
            text=text,
        )
        cross_selection = (
            _selection(
                image_path,
                source_dataset="abo",
                source_record_id="listing:ABO-NO-EDGE/image:main",
            )
            if index == 12
            else _selection(
                image_path,
                source_dataset="fashioniq",
                source_record_id=f"dress:cross-anchor-{index:02d}",
            )
        )
        queries.append(cross_query)
        selections.append(cross_selection)
        cross_queries.append(cross_query)
        cross_selections.append(cross_selection)
        ordinal += 1

    rpc_images: list[dict[str, object]] = []
    rpc_annotations: list[dict[str, object]] = []
    annotation_id = 1
    for index in range(40):
        image_id = 1_000 + index
        image_path = f"query_images/multi/rpc-{index:02d}.jpg"
        queries.append(
            _query(
                ordinal,
                capability="product.multi_search",
                image_path=image_path,
                text="find every product",
            )
        )
        selections.append(
            _selection(
                image_path,
                source_dataset="rpc",
                source_record_id=f"val2019:{image_id}",
            )
        )
        rpc_images.append({"id": image_id, "file_name": f"{image_id}.jpg"})
        for category_id in (1, 1, 2):
            rpc_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [index, category_id, 10, 20],
                }
            )
            annotation_id += 1
        ordinal += 1
    # Unknown categories are retained as typed unresolved objects and must not
    # be converted into speculative candidates.
    rpc_annotations.append(
        {
            "id": annotation_id,
            "image_id": 1_000,
            "category_id": 999,
            "bbox": [0, 0, 1, 1],
        }
    )

    while ordinal <= ceiling.CORE_QUERY_COUNT:
        queries.append(
            _query(
                ordinal,
                capability="product.exact_match",
                image_path=f"query_images/exact/filler-{ordinal:04d}.jpg",
                text="exact item",
            )
        )
        ordinal += 1

    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_bytes(b"".join(canonical_json_bytes(query) for query in queries))
    selection_path = tmp_path / "selection-manifest.json"
    selection_content = canonical_json_bytes({"selections": selections})
    selection_path.write_bytes(selection_content)
    coordination_graph_path = tmp_path / "style-coordination-graph.json"
    coordination_graph_path.write_bytes(
        canonical_json_bytes(
            _coordination_graph(
                selection_content=selection_content,
                cross_queries=cross_queries,
                cross_selections=cross_selections,
            )
        )
    )

    for category in ceiling.FASHIONIQ_CATEGORIES:
        for split_name in ceiling.FASHIONIQ_SPLITS:
            rows = dress_rows if (category, split_name) == ("dress", "train") else []
            (captions_dir / f"cap.{category}.{split_name}.json").write_bytes(
                canonical_json_bytes(rows)
            )

    annotation_payload = {
        "images": rpc_images,
        "annotations": rpc_annotations,
        "categories": [
            {"id": 1, "name": "bottle"},
            {"id": 2, "name": "bag"},
        ],
    }
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            ceiling.RPC_ANNOTATION_MEMBER,
            canonical_json_bytes(annotation_payload),
        )
    adapter_path = tmp_path / "adapter-run.json"
    adapter_path.write_bytes(
        canonical_json_bytes(
            {
                "archives": [
                    {
                        "logical_path": "rpc/archive.zip",
                        "bytes": archive_path.stat().st_size,
                        "sha256": "a" * 64,
                    }
                ]
            }
        )
    )
    return {
        "queries_path": queries_path,
        "selection_path": selection_path,
        "style_coordination_graph_path": coordination_graph_path,
        "fashioniq_captions_dir": captions_dir,
        "fashioniq_images_dir": images_dir,
        "rpc_archive_path": archive_path,
        "rpc_adapter_manifest_path": adapter_path,
    }


def test_build_receipt_splits_graph_mapped_and_uncovered_cross_requests(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)

    first = ceiling.build_receipt(**inputs)
    second = ceiling.build_receipt(**inputs)

    assert first == second
    assert first["gate0_passed"] is True
    assert first["execution"]["provider_calls_performed"] == 0
    assert first["claim_boundary"]["model_effect_claimed"] is False
    assert len(first["style"]["sample_ids"]["same_category_eligible"]) == 20
    assert {"q-0001", "q-0002"}.issubset(
        first["style"]["sample_ids"]["same_category_eligible"]
    )
    assert len(first["style"]["sample_ids"]["cross_category_coordination_mapped"]) == 9
    assert (
        len(first["style"]["sample_ids"]["cross_category_coordination_uncovered"]) == 11
    )
    mapped = first["style"]["cross_category_coordination_mapped"]
    assert all(row["hits"] for row in mapped["results"])
    assert all(
        hit["category_l1"] in ceiling.STYLE_COORDINATION_CATEGORIES
        and hit["provenance"]["relation"] == ceiling.STYLE_COORDINATION_RELATION
        for row in mapped["results"]
        for hit in row["hits"]
    )
    uncovered = first["style"]["cross_category_coordination_uncovered"]
    assert uncovered["uncovered_counted_as_hit"] is False
    assert uncovered["eligible_hit_numerator_contribution"] == 0
    assert uncovered["eligible_hit_denominator_contribution"] == 0
    assert all(row["hits"] == [] for row in uncovered["results"])
    assert any(
        row["semantic_evidence"]["anchor_dataset"] == "abo"
        for row in uncovered["results"]
    )
    assert any(
        row["requested_target_categories"] == [] and row["allowed_target_categories"]
        for row in mapped["results"]
    )
    assert first["sampling_policy"]["style"]["test_frozen_text_used"] is False
    pool = first["style"]["pool_counts"]
    assert pool["runtime_coordination_prefilter_query_count"] == (
        pool["cross_category_mapped_query_count"]
        + pool["cross_category_uncovered_query_count"]
    )
    assert pool["runtime_coordination_unconfirmed_query_count"] == 0
    classifier = first["sampling_policy"]["style"]["runtime_classifier"]
    assert classifier["runtime_policy_version"] == (
        ceiling.PORTFOLIO_TOOL_RUNTIME_POLICY
    )
    assert "tool ceiling" in first["claim_boundary"]["proves"]

    multi = first["multi"]
    assert len(multi["sample_ids"]) == 40
    assert multi["counts"] == {
        "object_count": 121,
        "matched_object_count": 120,
        "unresolved_object_count": 1,
        "candidate_count_after_category_dedup": 80,
    }
    first_scene = next(
        row for row in multi["results"] if row["source_record_id"] == "val2019:1000"
    )
    assert [candidate["quantity"] for candidate in first_scene["candidates"]] == [2, 1]
    unresolved = [
        obj
        for obj in first_scene["objects"]
        if obj["resolution_status"] == "unresolved"
    ]
    assert len(unresolved) == 1
    assert all(
        unresolved[0]["item_ref"] not in candidate["item_refs"]
        for candidate in first_scene["candidates"]
    )

    unsigned = dict(first)
    observed = unsigned.pop("receipt_sha256")
    assert observed == sha256_bytes(canonical_json_bytes(unsigned))
    assert all(check["passed"] for check in first["checks"])


def test_missing_annotation_backed_target_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    (inputs["fashioniq_images_dir"] / "dress" / "target-00.jpg").unlink()

    with pytest.raises(ceiling.GateEvidenceError, match="only 19 unique assets"):
        ceiling.build_receipt(**inputs)


def test_coordination_graph_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    graph_path = inputs["style_coordination_graph_path"]
    graph = json.loads(graph_path.read_bytes())
    graph["edges"][0]["confidence"] = 0.1
    graph_path.write_bytes(canonical_json_bytes(graph))

    with pytest.raises(ceiling.GateEvidenceError, match="self hash"):
        ceiling.build_receipt(**inputs)


def test_coordination_graph_fake_edge_id_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    graph_path = inputs["style_coordination_graph_path"]
    graph = json.loads(graph_path.read_bytes())
    graph["edges"][0]["edge_id"] = "style.edge.v1." + "f" * 64
    graph["edges"].sort(key=lambda edge: edge["edge_id"])
    _resign_coordination_graph(graph)
    graph_path.write_bytes(canonical_json_bytes(graph))

    with pytest.raises(ceiling.GateEvidenceError, match="identity is invalid"):
        ceiling.build_receipt(**inputs)


def test_coordination_graph_missing_required_facet_fails_closed(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    graph_path = inputs["style_coordination_graph_path"]
    graph = json.loads(graph_path.read_bytes())
    graph["edges"][0]["facets"].pop(1)
    _resign_coordination_graph(graph)
    graph_path.write_bytes(canonical_json_bytes(graph))

    with pytest.raises(ceiling.GateEvidenceError, match="facet contract is invalid"):
        ceiling.build_receipt(**inputs)


def test_ceiling_runtime_parity_with_literal_chinese_queries(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    selection_content, selection_rows = ceiling._load_selection(
        inputs["selection_path"]
    )
    by_destination = ceiling._selection_by_destination(selection_rows)
    _, coordination_edges, _ = ceiling._load_style_coordination_graph(
        inputs["style_coordination_graph_path"],
        selection_content=selection_content,
        by_destination=by_destination,
    )
    rows = [
        json.loads(line) for line in inputs["queries_path"].read_bytes().splitlines()
    ]
    base = next(
        row
        for row in rows
        if str(row["image_path"]).startswith("query_images/style/cross-")
    )
    selection = by_destination[str(base["image_path"])]

    def matches(text: str) -> list[dict[str, object]]:
        query = {**base, "text": text}
        return ceiling._matching_coordination_edges(
            query=query,
            selection=selection,
            coordination_edges=coordination_edges,
        )

    assert ceiling._is_style_coordination_query("找类似配色的连衣裙") is False
    assert ceiling._is_style_coordination_query("找几件相近造型的裙子") is False

    generic = matches("格纹裙咋配")
    assert ceiling._is_style_coordination_query("格纹裙咋配") is True
    assert {edge["candidate"]["category_l1"] for edge in generic} == {
        "footwear",
        "bag",
        "jewelry",
    }
    assert matches("给这条格纹裙配外套") == []

    bag = matches("不配鞋了；改搭斜挎包")
    assert [edge["candidate"]["category_l1"] for edge in bag] == ["bag"]
    assert "crossbody" in bag[0]["candidate"]["feature_tags"]

    assert matches("给这条裙子搭白色鞋") == []
    sandal = matches("给这条裙子搭平底凉鞋")
    assert len(sandal) == 1
    assert sandal[0]["candidate"]["feature_tags"] == ["sandal", "flat"]

    full_outfit = matches("给这条裙子搭平底凉鞋和斜挎包")
    assert {edge["candidate"]["category_l1"] for edge in full_outfit} == {
        "footwear",
        "bag",
    }

    abo_query = {
        **base,
        "asset_id": "asset-without-graph-edge",
        "image_path": "query_images/style/abo-anchor.jpg",
        "text": "格纹裙咋配",
    }
    abo_selection = _selection(
        "query_images/style/abo-anchor.jpg",
        source_dataset="abo",
        source_record_id="listing:ABO/image:main",
    )
    assert (
        ceiling._matching_coordination_edges(
            query=abo_query,
            selection=abo_selection,
            coordination_edges=coordination_edges,
        )
        == []
    )


def test_ceiling_uses_runtime_classifier_and_exact_candidate_constraints(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    selection_content, selection_rows = ceiling._load_selection(
        inputs["selection_path"]
    )
    by_destination = ceiling._selection_by_destination(selection_rows)
    _, coordination_edges, _ = ceiling._load_style_coordination_graph(
        inputs["style_coordination_graph_path"],
        selection_content=selection_content,
        by_destination=by_destination,
    )
    rows = [
        json.loads(line) for line in inputs["queries_path"].read_bytes().splitlines()
    ]
    base = next(
        row
        for row in rows
        if str(row["image_path"]).startswith("query_images/style/cross-")
    )
    selection = by_destination[str(base["image_path"])]

    def matches(text: str) -> list[dict[str, object]]:
        return ceiling._matching_coordination_edges(
            query={**base, "text": text},
            selection=selection,
            coordination_edges=coordination_edges,
        )

    similar_color = "\u627e\u7c7b\u4f3c\u914d\u8272\u7684\u8fde\u8863\u88d9"
    similar_shape = "\u627e\u51e0\u4ef6\u76f8\u8fd1\u9020\u578b\u7684\u88d9\u5b50"
    assert ceiling._is_style_coordination_query(similar_color) is False
    assert ceiling._is_style_coordination_query(similar_shape) is False

    generic_text = "\u683c\u7eb9\u88d9\u548b\u914d"
    generic = matches(generic_text)
    assert ceiling._is_style_coordination_query(generic_text) is True
    assert {edge["candidate"]["category_l1"] for edge in generic} == {
        "footwear",
        "bag",
        "jewelry",
    }
    partial_edges = {
        asset_id: [
            edge for edge in edges if edge["candidate"]["category_l1"] != "jewelry"
        ]
        for asset_id, edges in coordination_edges.items()
    }
    assert {
        edge["candidate"]["category_l1"]
        for edge in ceiling._matching_coordination_edges(
            query={**base, "text": generic_text},
            selection=selection,
            coordination_edges=partial_edges,
        )
    } == {"footwear", "bag"}
    assert matches("\u7ed9\u8fd9\u6761\u683c\u7eb9\u88d9\u914d\u5916\u5957") == []

    bag = matches("\u4e0d\u914d\u978b\u4e86\uff1b\u6539\u642d\u659c\u630e\u5305")
    assert [edge["candidate"]["category_l1"] for edge in bag] == ["bag"]
    assert "crossbody" in bag[0]["candidate"]["feature_tags"]

    assert matches("\u7ed9\u8fd9\u6761\u88d9\u5b50\u642d\u767d\u8272\u978b") == []
    sandal = matches("\u7ed9\u8fd9\u6761\u88d9\u5b50\u642d\u5e73\u5e95\u51c9\u978b")
    assert len(sandal) == 1
    assert sandal[0]["candidate"]["feature_tags"] == ["sandal", "flat"]

    full_outfit = matches(
        "\u7ed9\u8fd9\u6761\u88d9\u5b50\u642d\u5e73\u5e95\u51c9\u978b"
        "\u548c\u659c\u630e\u5305"
    )
    assert {edge["candidate"]["category_l1"] for edge in full_outfit} == {
        "footwear",
        "bag",
    }
    assert (
        matches(
            "\u642d\u5c0f\u5305\uff1b\u978b\u8ddf\u4e0d\u8981\u8d85\u8fc7"
            "\u4e09\u5398\u7c73"
        )
        == []
    )

    abo_query = {
        **base,
        "asset_id": "asset-without-graph-edge",
        "image_path": "query_images/style/abo-anchor.jpg",
        "text": generic_text,
    }
    abo_selection = _selection(
        "query_images/style/abo-anchor.jpg",
        source_dataset="abo",
        source_record_id="listing:ABO/image:main",
    )
    assert (
        ceiling._matching_coordination_edges(
            query=abo_query,
            selection=abo_selection,
            coordination_edges=coordination_edges,
        )
        == []
    )


def test_test_frozen_style_text_is_never_evaluated(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    queries_path = inputs["queries_path"]
    rows = [json.loads(line) for line in queries_path.read_bytes().splitlines()]
    frozen = next(
        row for row in rows if row["canonical_capability"] == "product.exact_match"
    )
    frozen["canonical_capability"] = "product.style_recommendation"
    frozen["split"] = "test_frozen"
    frozen["text"] = None
    queries_path.write_bytes(b"".join(canonical_json_bytes(query) for query in rows))

    receipt = ceiling.build_receipt(**inputs)

    assert receipt["gate0_passed"] is True
