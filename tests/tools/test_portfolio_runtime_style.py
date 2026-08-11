import json
from pathlib import Path

from PIL import Image
import pytest

from skillchain.tools.portfolio_runtime import (
    PortfolioRuntimeError,
    PortfolioProductSearchService,
    PortfolioRuntimeSources,
    _PortfolioMetadataIndex,
    _is_style_coordination_query,
    _normalized_selection_row,
    _requested_style_coordination_categories,
    _style_coordination_constraint,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _png(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)
    return sha256_bytes(path.read_bytes())


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _style_service(
    tmp_path: Path, *, with_coordination_graph: bool = False
) -> tuple[PortfolioProductSearchService, Path]:
    rows = []
    colors = {"A": (180, 20, 20), "B": (230, 30, 30), "C": (20, 20, 180)}
    for product_id, color in colors.items():
        relative = f"query_images/{product_id}.png"
        digest = _png(tmp_path / relative, color)
        rows.append(
            {
                "canonical_capability": "product.style_recommendation",
                "image_path": relative,
                "image_sha256": digest,
                "product_id": f"fashioniq:{product_id}",
                "source_record_id": f"dress:{product_id}",
                "source_dataset": "fashioniq",
                "category_l1": "dress",
            }
        )
    candidate_specs = (
        {
            "name": "shoe",
            "color": (25, 25, 25),
            "product_id": "abo:SHOE",
            "category_l1": "footwear",
            "display_title": "Black low-heel ankle boot",
            "feature_tags": ["ankle_boot", "low_block_heel"],
            "color_families": ["black"],
        },
        {
            "name": "bag",
            "color": (90, 45, 20),
            "product_id": "abo:BAG",
            "category_l1": "bag",
            "display_title": "Cognac crossbody camera bag",
            "feature_tags": ["camera_bag", "crossbody", "small"],
            "color_families": ["cognac"],
        },
        {
            "name": "earrings",
            "color": (180, 180, 180),
            "product_id": "abo:EARRINGS",
            "category_l1": "jewelry",
            "display_title": "Silver-tone drop earrings",
            "feature_tags": ["earrings", "drop", "silver_tone"],
            "color_families": ["silver"],
        },
        {
            "name": "necklace",
            "color": (160, 160, 160),
            "product_id": "abo:NECKLACE",
            "category_l1": "jewelry",
            "display_title": "Silver-tone pendant necklace",
            "feature_tags": ["necklace", "pendant", "silver_tone"],
            "color_families": ["silver"],
        },
    )
    candidate_rows: list[dict[str, object]] = []
    for spec in candidate_specs:
        relative = f"query_images/{spec['name']}.png"
        digest = _png(tmp_path / relative, spec["color"])
        row = {
            "canonical_capability": "product.style_recommendation",
            "image_path": relative,
            "image_sha256": digest,
            "product_id": spec["product_id"],
            "source_record_id": f"listing:{spec['name'].upper()}/image:1",
            "source_dataset": "abo",
            "category_l1": spec["category_l1"],
            "display_title": spec["display_title"],
            "feature_tags": spec["feature_tags"],
            "color_families": spec["color_families"],
        }
        rows.append(row)
        candidate_rows.append(row)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"selections": rows}, ensure_ascii=False), encoding="utf-8"
    )
    dataset_assets = tmp_path / "dataset-assets.jsonl"
    rpc = tmp_path / "rpc.jsonl"
    inaturalist = tmp_path / "inaturalist.jsonl"
    recipes = tmp_path / "recipes.jsonl"
    for path in (dataset_assets, rpc, inaturalist, recipes):
        _jsonl(path, [])
    catalog = tmp_path / "catalog" / "assets.jsonl"
    catalog.parent.mkdir(parents=True)
    _jsonl(
        catalog,
        [
            {
                "asset_id": f"asset-{row['product_id']}",
                "product_id": row["product_id"],
                "source_dataset": row["source_dataset"],
                "source_record_id": row["source_record_id"],
                "sha256": row["image_sha256"],
                "local_path": row["image_path"],
            }
            for row in rows
        ],
    )
    captions = tmp_path / "cap.dress.test.json"
    captions.write_text(
        json.dumps(
            [
                {
                    "candidate": "A",
                    "target": "B",
                    "captions": ["brighter red", "shorter sleeves"],
                }
            ]
        ),
        encoding="utf-8",
    )
    coordination_graph = None
    if with_coordination_graph:
        anchor = rows[0]
        graph_edges: list[dict[str, object]] = []
        for candidate in candidate_rows:
            graph_edges.append(
                {
                    "edge_id": "pending",
                    "anchor": {
                        "asset_id": f"asset-{anchor['product_id']}",
                        "product_id": anchor["product_id"],
                        "source_dataset": anchor["source_dataset"],
                        "source_record_id": anchor["source_record_id"],
                        "image_sha256": anchor["image_sha256"],
                        "image_path": anchor["image_path"],
                        "category_l1": "dress",
                        "color_families": ["rose"],
                    },
                    "candidate": {
                        "asset_id": f"asset-{candidate['product_id']}",
                        "product_id": candidate["product_id"],
                        "source_dataset": candidate["source_dataset"],
                        "source_record_id": candidate["source_record_id"],
                        "image_sha256": candidate["image_sha256"],
                        "image_path": candidate["image_path"],
                        "category_l1": candidate["category_l1"],
                        "display_title": candidate["display_title"],
                        "audience": "women",
                        "feature_tags": candidate["feature_tags"],
                        "color_families": candidate["color_families"],
                    },
                    "relation_kind": "portfolio_curated_coordination_rule",
                    "confidence": 0.9,
                    "facets": [
                        {
                            "facet": "category",
                            "value": candidate["category_l1"],
                            "confidence": 1.0,
                        },
                        {
                            "facet": "palette",
                            "value": ",".join(candidate["color_families"]),
                            "confidence": 0.95,
                        },
                        {
                            "facet": "verified_attributes",
                            "value": ",".join(candidate["feature_tags"]),
                            "confidence": 0.95,
                        },
                        {
                            "facet": "coordination_rule",
                            "value": "neutral_palette_rule",
                            "confidence": 0.9,
                        },
                    ],
                    "annotation_policy_version": (
                        "portfolio-style-coordination-candidate-review-v1"
                    ),
                }
            )
        graph_without_hash = {
            "kind": "portfolio-style-coordination-graph",
            "schema_version": 1,
            "policy_version": "portfolio-style-coordination-graph-v1",
            "source_bindings": {
                "selection_manifest_sha256": sha256_bytes(selection.read_bytes()),
                "runtime_catalog_assets_sha256": sha256_bytes(catalog.read_bytes()),
                "abo_listings_archive_sha256": "a" * 64,
                "candidate_seed_sha256": "b" * 64,
            },
            "edges": graph_edges,
        }
        for edge in graph_without_hash["edges"]:
            edge["edge_id"] = "style.edge.v1." + sha256_bytes(
                canonical_json_bytes(
                    {
                        "anchor_asset_id": edge["anchor"]["asset_id"],
                        "annotation_policy_version": edge["annotation_policy_version"],
                        "candidate_asset_id": edge["candidate"]["asset_id"],
                        "relation_kind": edge["relation_kind"],
                        "rule": "neutral_palette_rule",
                    }
                )
            )
        graph_without_hash["edges"].sort(key=lambda item: item["edge_id"])
        graph = {
            **graph_without_hash,
            "graph_sha256": sha256_bytes(canonical_json_bytes(graph_without_hash)),
        }
        coordination_graph = tmp_path / "style-coordination-graph.json"
        coordination_graph.write_text(
            json.dumps(graph, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    sources = PortfolioRuntimeSources(
        selection_manifest=selection,
        dataset_assets=dataset_assets,
        runtime_catalog_assets=catalog,
        rpc_scenes=rpc,
        inaturalist_manifest=inaturalist,
        recipe_evidence=recipes,
        asset_root=tmp_path,
        fashioniq_captions=(captions,),
        style_coordination_graph=coordination_graph,
    )
    return PortfolioProductSearchService(
        _PortfolioMetadataIndex(sources)
    ), tmp_path / "query_images/A.png"


def test_core_selection_row_normalizes_capability_and_asset_root_fields() -> None:
    normalized = _normalized_selection_row(
        {
            "candidate_id": "fashioniq.core.1",
            "destination_path": "query_images/divergent_rec/a.jpg",
            "expected_sha256": "a" * 64,
            "capability_bindings": [
                {
                    "canonical_capability": "product.style_recommendation",
                    "canonical_intent": "divergent_rec",
                }
            ],
            "draft": {
                "source_dataset": "fashioniq",
                "source_record_id": "dress:A",
                "product_id": "fashioniq:A",
                "local_path": "query_images/divergent_rec/a.jpg",
            },
        }
    )

    assert normalized["capability_ids"] == ["product.style_recommendation"]
    assert normalized["category_l1"] == "dress"
    assert normalized["image_path"] == "query_images/divergent_rec/a.jpg"


def test_style_runtime_uses_fashioniq_relative_caption_evidence(tmp_path: Path) -> None:
    service, anchor = _style_service(tmp_path)

    trace = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="brighter red",
    )

    assert trace.style_submode == "same_category_alternative"
    assert [hit.product.product_id for hit in trace.hits] == ["fashioniq:B"]
    hit = trace.hits[0]
    assert hit.similarity_source == "fashioniq_relative_caption_graph"
    assert hit.facet_evidence[0].value == "brighter red; shorter sleeves"
    assert hit.facet_evidence[0].provenance == "fashioniq_relative_caption"


def test_style_runtime_fails_closed_for_cross_category_coordination(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path)

    trace = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="这件裙子搭配什么鞋？",
    )

    assert trace.style_submode == "cross_category_coordination"
    assert trace.hits == ()
    assert trace.unsupported_reason is not None


def test_style_runtime_returns_exact_curated_cross_category_candidates(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path, with_coordination_graph=True)

    trace = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="Pair this dress with shoes",
    )

    assert trace.style_submode == "cross_category_coordination"
    assert trace.unsupported_reason is None
    assert [hit.product.product_id for hit in trace.hits] == ["abo:SHOE"]
    hit = trace.hits[0]
    assert hit.product.category_l1 == "footwear"
    assert hit.similarity_source == "verified_coordination_graph"
    assert all(
        item.provenance == "portfolio_curated_coordination_rule"
        for item in hit.facet_evidence
    )


def test_style_runtime_filters_requested_coordination_family_and_subtype(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path, with_coordination_graph=True)

    trace = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="Pair this dress with a bag",
    )

    assert trace.style_submode == "cross_category_coordination"
    assert [hit.product.product_id for hit in trace.hits] == ["abo:BAG"]

    unsupported_subtype = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="Pair this dress with a handheld clutch bag",
    )
    assert unsupported_subtype.hits == ()
    assert unsupported_subtype.unsupported_reason is not None

    negated_shoe = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="Do not pair with shoes; pair with a crossbody bag",
    )
    assert [hit.product.product_id for hit in negated_shoe.hits] == ["abo:BAG"]


def test_style_runtime_applies_exact_color_feature_and_negative_constraints(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path, with_coordination_graph=True)

    exact = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="怎么搭：银色耳饰、低跟鞋",
    )
    assert {hit.product.product_id for hit in exact.hits} == {
        "abo:EARRINGS",
        "abo:SHOE",
    }

    wrong_color = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="给这条裙子搭白色低跟鞋",
    )
    assert wrong_color.hits == ()

    wrong_subtype = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="给这条裙子搭平底凉鞋",
    )
    assert wrong_subtype.hits == ()

    unverified_height = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="给这条裙子搭三厘米粗跟鞋",
    )
    assert unverified_height.hits == ()

    partial_family = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭白色低跟鞋和小包",
    )
    assert partial_family.hits == ()

    anchor_color_is_not_target_color = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="这条红色裙配啥低跟鞋",
    )
    assert [
        hit.product.product_id for hit in anchor_color_is_not_target_color.hits
    ] == ["abo:SHOE"]

    negative_subtype_keeps_family = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="怎么搭；不要细高跟鞋",
    )
    assert any(
        hit.product.category_l1 == "footwear"
        for hit in negative_subtype_keeps_family.hits
    )

    exact_height_with_positive_bag = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭小包；鞋跟不要超过三厘米",
    )
    assert exact_height_with_positive_bag.hits == ()

    safe_height_with_positive_bag = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭小包；鞋跟不要太高",
    )
    assert {hit.product.product_id for hit in safe_height_with_positive_bag.hits} == {
        "abo:BAG",
        "abo:SHOE",
    }

    multi_family_text = "搭银色耳饰、裸色低跟鞋和小包"
    assert _style_coordination_constraint(
        multi_family_text, category="jewelry"
    ).required_colors == {"silver"}
    assert _style_coordination_constraint(
        multi_family_text, category="footwear"
    ).required_colors == {"taupe"}
    assert not _style_coordination_constraint(
        multi_family_text, category="bag"
    ).required_colors

    no_necklace = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭一套耳饰；不戴项链",
    )
    assert [hit.product.product_id for hit in no_necklace.hits] == ["abo:EARRINGS"]


def test_style_runtime_scopes_mode_and_unsupported_targets_to_positive_clauses(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path, with_coordination_graph=True)

    generic = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="格纹裙咋配",
    )
    assert generic.style_submode == "cross_category_coordination"
    assert {hit.product.category_l1 for hit in generic.hits} == {
        "footwear",
        "bag",
        "jewelry",
    }

    unsupported = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="给这条格纹裙配外套",
    )
    assert unsupported.style_submode == "cross_category_coordination"
    assert unsupported.hits == ()

    mixed_unsupported = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="给这条格纹裙配低跟鞋和外套",
    )
    assert mixed_unsupported.hits == ()

    same_category = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="找类似配色的连衣裙",
    )
    assert same_category.style_submode == "same_category_alternative"
    assert same_category.hits

    no_bag = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭一套；不要包",
    )
    assert no_bag.style_submode == "cross_category_coordination"
    assert all(hit.product.category_l1 != "bag" for hit in no_bag.hits)


@pytest.mark.parametrize(
    "query_text,expected_categories",
    [
        ("按黑白配色给我鞋、包和耳饰搭配", {"footwear", "bag", "jewelry"}),
        ("给鞋、耳饰和小包的搭配", {"footwear", "bag", "jewelry"}),
        ("推荐一套完整搭配", set()),
        ("裸色平底鞋和小号草编包搭", {"footwear", "bag"}),
        ("红高跟不要了，换黑色乐福鞋", {"footwear"}),
        ("推荐一套完整配饰", set()),
        ("一套通勤搭配", set()),
        ("晚宴搭配，不要高跟", set()),
        (
            "围绕耳钉做通勤造型：短发侧别、低饱和粉妆，配灰色西装和银色细项链，别再加大件首饰。",
            {"jewelry"},
        ),
    ],
)
def test_style_runtime_recognizes_real_target_before_verb_coordination_queries(
    query_text: str,
    expected_categories: set[str],
) -> None:
    assert _is_style_coordination_query(query_text) is True
    assert set(_requested_style_coordination_categories(query_text)) == (
        expected_categories
    )


def test_style_runtime_distinguishes_anchor_mentions_and_inline_feature_negation(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path, with_coordination_graph=True)

    anchor_mentions = [
        service.trace_similar_styles(
            anchor,
            asset_id="asset-fashioniq:A",
            query_text=f"{lead}这条裙子，配什么鞋",
        )
        for lead in ("怎么搭", "如何搭", "如何搭配", "帮我搭", "帮我搭配")
    ]
    inline_negation = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭配一双不要高跟的鞋",
    )
    unsupported_suit = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text=(
            "围绕耳钉做通勤造型：短发侧别、低饱和粉妆，"
            "配灰色西装和银色细项链，别再加大件首饰。"
        ),
    )

    assert all(
        trace.style_submode == "cross_category_coordination"
        and [hit.product.product_id for hit in trace.hits] == ["abo:SHOE"]
        for trace in anchor_mentions
    )
    assert inline_negation.style_submode == "cross_category_coordination"
    assert [hit.product.product_id for hit in inline_negation.hits] == ["abo:SHOE"]
    assert unsupported_suit.style_submode == "cross_category_coordination"
    assert unsupported_suit.hits == ()


@pytest.mark.parametrize(
    "query_text",
    [
        "这个包搭载了NFC芯片吗",
        "鞋搭载缓震科技",
        "耳钉搭扣松了怎么修",
        "鞋很百搭",
    ],
)
def test_style_runtime_does_not_treat_lexical_bare_da_as_coordination(
    query_text: str,
) -> None:
    assert _is_style_coordination_query(query_text) is False


def test_style_runtime_scopes_inline_bare_family_exclusions(tmp_path: Path) -> None:
    service, anchor = _style_service(tmp_path, with_coordination_graph=True)

    for query_text in ("搭配鞋但不要包", "pair with shoes but no bag"):
        trace = service.trace_similar_styles(
            anchor,
            asset_id="asset-fashioniq:A",
            query_text=query_text,
        )
        assert trace.style_submode == "cross_category_coordination"
        assert [hit.product.product_id for hit in trace.hits] == ["abo:SHOE"]

    generic_without_bag = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭配不要包的造型",
    )
    assert generic_without_bag.style_submode == "cross_category_coordination"
    assert {hit.product.category_l1 for hit in generic_without_bag.hits} == {
        "footwear",
        "jewelry",
    }

    assert _is_style_coordination_query("分别给鞋和包搭配") is True
    assert set(_requested_style_coordination_categories("分别给鞋和包搭配")) == {
        "footwear",
        "bag",
    }


def test_style_runtime_handles_reverse_anchor_and_positive_negation_contrast(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path, with_coordination_graph=True)

    for query_text in (
        "用黑色鞋搭这条裙子",
        "鞋搭配这条裙子",
        "鞋搭这条裙子",
    ):
        trace = service.trace_similar_styles(
            anchor,
            asset_id="asset-fashioniq:A",
            query_text=query_text,
        )
        assert trace.style_submode == "cross_category_coordination"
        assert [hit.product.product_id for hit in trace.hits] == ["abo:SHOE"]

    for query_text in (
        "搭配不要包但要鞋",
        "搭配不要包改要鞋",
        "pair with no bag but shoes",
    ):
        trace = service.trace_similar_styles(
            anchor,
            asset_id="asset-fashioniq:A",
            query_text=query_text,
        )
        assert [hit.product.product_id for hit in trace.hits] == ["abo:SHOE"]

    shoe_and_earrings = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭配鞋不要包要耳饰",
    )
    shoe_without_necklace = service.trace_similar_styles(
        anchor,
        asset_id="asset-fashioniq:A",
        query_text="搭配鞋但不要项链要耳环",
    )
    assert {hit.product.product_id for hit in shoe_and_earrings.hits} == {
        "abo:SHOE",
        "abo:EARRINGS",
    }
    assert {hit.product.product_id for hit in shoe_without_necklace.hits} == {
        "abo:SHOE",
        "abo:EARRINGS",
    }


def test_style_runtime_requires_the_exact_hidden_anchor_identity(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path, with_coordination_graph=True)

    trace = service.trace_similar_styles(
        anchor,
        asset_id="asset-wrong-anchor",
        query_text="Pair this dress with shoes",
    )

    assert trace.hits == ()
    assert trace.unsupported_reason is not None


def test_style_runtime_rejects_query_specific_coordination_graph_fields(
    tmp_path: Path,
) -> None:
    service, _anchor = _style_service(tmp_path, with_coordination_graph=True)
    graph_path = service.index.sources.style_coordination_graph
    assert graph_path is not None
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["edges"][0]["query_id"] = "forbidden"
    graph_without_hash = {
        key: value for key, value in graph.items() if key != "graph_sha256"
    }
    graph["graph_sha256"] = sha256_bytes(canonical_json_bytes(graph_without_hash))
    graph_path.write_text(
        json.dumps(graph, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(PortfolioRuntimeError, match="query-specific"):
        _PortfolioMetadataIndex(service.index.sources)


def test_style_runtime_does_not_treat_negated_coordination_as_a_request(
    tmp_path: Path,
) -> None:
    service, anchor = _style_service(tmp_path)

    negated = service.trace_similar_styles(
        anchor,
        asset_id="asset-a",
        query_text="不配鞋了，给我几个同类替代款",
    )
    later_positive = service.trace_similar_styles(
        anchor,
        asset_id="asset-a",
        query_text="不配鞋了；改搭手拿包",
    )

    assert negated.style_submode == "same_category_alternative"
    assert negated.hits
    assert later_positive.style_submode == "cross_category_coordination"
    assert later_positive.hits == ()


def test_style_runtime_falls_back_to_local_features_without_graph(
    tmp_path: Path,
) -> None:
    service, _anchor = _style_service(tmp_path)

    trace = service.trace_similar_styles(
        tmp_path / "query_images/C.png",
        asset_id="asset-c",
        query_text="给我几个同类替代款",
    )

    assert trace.hits
    assert all(
        hit.similarity_source == "local_image_feature_cosine" for hit in trace.hits
    )
    assert all(
        hit.facet_evidence[0].provenance == "local_image_feature" for hit in trace.hits
    )
