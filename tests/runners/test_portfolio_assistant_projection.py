import json

from pydantic import BaseModel
import pytest

from skillchain.runners.assistant import (
    GCS_V2_MODEL_RESPONSE_CONTRACT_SHA256,
    GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION,
    _bind_model_tool_arguments,
    _gcs_v2_model_response_contract_for_tool,
    _model_query_projection,
    _model_visible_tool_output,
    _visible_projection,
    gcs_v2_model_response_contract_payload,
)
from skillchain.tools.registry import ToolCallError
from skillchain.tools.document_ocr import (
    DocumentOCRResult,
    OCRInputBinding,
    OCRLine,
)
from skillchain.tools.model_artifacts import ModelRuntimeBinding
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


class _KBHit(BaseModel):
    title: str
    text: str


@pytest.mark.parametrize(
    ("tool_name", "required_sections", "required_fragment"),
    (
        (
            "image_product_search",
            ["answer", "product_cards", "uncertainty"],
            "no supported match",
        ),
        (
            "text_product_search",
            ["answer", "product_cards", "uncertainty"],
            "no supported match",
        ),
        (
            "multi_product_search",
            ["answer", "item_mapping", "product_cards", "uncertainty"],
            "candidate-N",
        ),
        (
            "style_similar_search",
            ["answer", "diversity_rationale", "product_cards", "uncertainty"],
            "style_evidence",
        ),
        (
            "document_ocr",
            ["answer", "evidence", "uncertainty"],
            "untrusted document text",
        ),
        (
            "encyclopedia_lookup",
            ["answer", "evidence", "uncertainty"],
            "tool-call-N-source-M",
        ),
        ("recipe_lookup", ["answer", "evidence", "uncertainty"], "no supported recipe"),
    ),
)
def test_gcs_v2_response_contract_is_public_tool_selected_and_explicit(
    tool_name: str,
    required_sections: list[str],
    required_fragment: str,
) -> None:
    contract = _gcs_v2_model_response_contract_for_tool(tool_name)

    assert contract is not None
    assert contract["policy_version"] == GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION
    assert contract["required_sections"] == required_sections
    assert required_fragment in json.dumps(contract, ensure_ascii=False)
    assert _gcs_v2_model_response_contract_for_tool("object_detect") is None


def test_gcs_v2_response_contract_has_frozen_query_independent_identity() -> None:
    payload = gcs_v2_model_response_contract_payload()

    assert payload["gold_query_label_used"] is False
    assert payload["scorer_sidecar_used"] is False
    assert payload["response_rewrite_or_renderer"] == (
        "at_most_one_fixed_model_format_repair"
    )
    assert payload["repair_tool_surface"] == "none"
    assert payload["repair_image_attachment"] is False
    assert (
        sha256_bytes(canonical_json_bytes(payload))
        == GCS_V2_MODEL_RESPONSE_CONTRACT_SHA256
    )


def test_product_projection_creates_public_cards_without_internal_ids() -> None:
    cards, evidence = _visible_projection(
        "image_product_search",
        {
            "hits": [
                {
                    "score": 0.875,
                    "product": {
                        "product_id": "private-product-id",
                        "title": "Black ankle boot",
                        "category_l1": "Shoes",
                        "image_path": "private/path.jpg",
                        "source": "private-source",
                    },
                }
            ]
        },
    )

    assert len(cards) == 1
    assert cards[0].title == "Black ankle boot"
    assert cards[0].body == "Shoes"
    serialized = evidence.model_dump_json()
    assert "private-product-id" not in serialized
    assert "private/path.jpg" not in serialized
    assert "private-source" not in serialized


def test_model_product_dto_uses_public_contract_handles_only() -> None:
    raw_output = {
        "query_asset_id": "asset.v2.private-evaluation-id",
        "query_image_sha256": "private-image-digest",
        "artifact_binding": {"query_artifact_sha256": "private-query-digest"},
        "hits": [
            {
                "rank": 1,
                "score": 1.0,
                "product": {
                    "product_id": "abo:B06XBMZQ6F",
                    "title": "ABO item B06XBMZQ6F",
                    "category_l1": "catalog_product",
                    "image_path": "private/catalog/path.jpg",
                    "source": "abo",
                },
            }
        ],
    }

    public = _model_visible_tool_output(
        "image_product_search",
        raw_output,
        call_index=2,
    )

    assert public == {
        "result_kind": "product_candidates",
        "candidates": [
            {
                "evidence_reference": "tool-call-2-evidence-1",
                "product_id": "tool-call-2-product-1",
                "title": "商品候选 1",
                "category": "catalog_product",
                "score": 1.0,
            }
        ],
    }
    serialized = json.dumps(public, ensure_ascii=False)
    for hidden in (
        "asset.v2.private-evaluation-id",
        "private-image-digest",
        "private-query-digest",
        "abo:B06XBMZQ6F",
        "B06XBMZQ6F",
        "private/catalog/path.jpg",
    ):
        assert hidden not in serialized

    cards, evidence = _visible_projection(
        "image_product_search",
        raw_output,
        call_index=2,
    )
    assert cards[0].title == "商品候选 1"
    candidate = public["candidates"][0]
    assert (
        dict(cards[0].fields)["evidence_reference"] == candidate["evidence_reference"]
    )
    assert dict(cards[0].fields)["product_id"] == candidate["product_id"]
    assert "B06XBMZQ6F" not in evidence.model_dump_json()


def test_model_query_uses_aliases_and_runner_binds_asset_privately() -> None:
    original = {
        "asset_id": "asset.v2.private-evaluation-id",
        "image_path": "query_images/private.jpg",
        "text": "有同款吗？",
        "turns": [{"role": "user", "content": "有同款吗？"}],
    }

    projected = _model_query_projection(original)

    assert projected == {
        **original,
        "asset_id": "query_asset",
        "image_path": "query_image",
    }
    assert original["asset_id"] == "asset.v2.private-evaluation-id"
    arguments = {"asset_id": "query_asset"}
    assert _bind_model_tool_arguments(
        "visual_product_search",
        arguments,
        query_asset_id="asset.v2.private-evaluation-id",
        query_text="有同款吗？",
    ) == {"asset_id": "asset.v2.private-evaluation-id"}
    assert arguments == {"asset_id": "query_asset"}


def test_style_binding_injects_authoritative_asset_and_frozen_query() -> None:
    model_arguments = {"asset_id": "query_asset"}

    assert _bind_model_tool_arguments(
        "style_similar_search",
        model_arguments,
        query_asset_id="asset.v2.private-style-anchor",
        query_text="给这条裙子配一双平底鞋",
    ) == {
        "asset_id": "asset.v2.private-style-anchor",
        "query": "给这条裙子配一双平底鞋",
    }
    assert model_arguments == {"asset_id": "query_asset"}

    with pytest.raises(ToolCallError, match="opaque query_asset handle"):
        _bind_model_tool_arguments(
            "style_similar_search",
            {"asset_id": "query_asset", "query": "model-selected query"},
            query_asset_id="asset.v2.private-style-anchor",
            query_text="给这条裙子配一双平底鞋",
        )


def test_multi_product_model_dto_removes_rpc_class_and_crop_identities() -> None:
    public = _model_visible_tool_output(
        "multi_product_search",
        {
            "input_binding": {"asset_id": "asset.v2.private"},
            "objects": [
                {
                    "detection_id": "private-detection",
                    "label": "50_instant_noodles",
                    "label_zh": "方便面",
                    "bbox_xyxy": [1, 2, 3, 4],
                    "crop_sha256": "private-crop-digest",
                    "hits": [
                        {
                            "score": 1.0,
                            "product": {
                                "product_id": "rpc:50",
                                "title": "RPC catalog class 50_instant_noodles",
                                "category_l1": "instant_noodles",
                                "image_path": "rpc-category/50",
                                "source": "rpc",
                            },
                        }
                    ],
                }
            ],
        },
        call_index=1,
    )

    assert public == {
        "result_kind": "multi_product_candidates",
        "items": [
            {
                "item_ref": "item-001",
                "label": "方便面",
                "status": "matched",
                "candidate_ordinal": 1,
                "candidate": {
                    "evidence_reference": "tool-call-1-evidence-1",
                    "product_id": "tool-call-1-product-1",
                    "title": "商品候选 1",
                    "category": "instant_noodles",
                    "score": 1.0,
                },
            }
        ],
    }
    serialized = json.dumps(public, ensure_ascii=False)
    for hidden in (
        "asset.v2.private",
        "private-detection",
        "private-crop-digest",
        "rpc:50",
        "RPC catalog class 50_instant_noodles",
        "rpc-category/50",
    ):
        assert hidden not in serialized

    cards, evidence = _visible_projection(
        "multi_product_search",
        {
            "objects": [
                {
                    "label_zh": "方便面",
                    "hits": [
                        {
                            "score": 1.0,
                            "product": {
                                "product_id": "rpc:50",
                                "title": "RPC catalog class 50_instant_noodles",
                                "category_l1": "instant_noodles",
                                "source": "rpc",
                            },
                        }
                    ],
                }
            ]
        },
        call_index=1,
    )
    candidate = public["items"][0]["candidate"]
    assert (
        dict(cards[0].fields)["evidence_reference"] == candidate["evidence_reference"]
    )
    assert dict(cards[0].fields)["product_id"] == candidate["product_id"]
    assert evidence.cards == cards


def test_style_projection_exposes_facets_but_hides_source_identity() -> None:
    raw_output = {
        "style_submode": "same_category_alternative",
        "hits": [
            {
                "score": 0.8125,
                "style_submode": "same_category_alternative",
                "similarity_source": "fashioniq_relative_caption_graph",
                "facet_evidence": [
                    {
                        "facet": "relative_style_change",
                        "value": "brighter red with shorter sleeves",
                        "confidence": 0.8125,
                        "provenance": "fashioniq_relative_caption",
                        "source_record_id": "dress:private-anchor->private-target",
                    }
                ],
                "product": {
                    "product_id": "fashioniq:private-target",
                    "title": "FASHIONIQ item private-target",
                    "category_l1": "dress",
                    "image_path": "private/target.jpg",
                    "source": "fashioniq",
                },
            }
        ],
    }

    public = _model_visible_tool_output(
        "style_similar_search", raw_output, call_index=4
    )
    cards, evidence = _visible_projection(
        "style_similar_search", raw_output, call_index=4
    )

    assert public["result_kind"] == "style_candidates"
    assert public["support_status"] == "supported"
    assert public["style_submode"] == "same_category_alternative"
    candidate = public["candidates"][0]
    assert candidate["similarity_source"] == "fashioniq_relative_caption_graph"
    assert candidate["style_evidence"] == [
        {
            "evidence_reference": "tool-call-4-style-evidence-1-1",
            "facet": "relative_style_change",
            "value": "brighter red with shorter sleeves",
            "confidence": 0.8125,
            "provenance": "fashioniq_relative_caption",
        }
    ]
    assert len(cards) == 1
    assert dict(cards[0].fields)["style_evidence_1"] == (
        "tool-call-4-style-evidence-1-1"
    )
    serialized = json.dumps(public, ensure_ascii=False) + evidence.model_dump_json()
    for hidden in (
        "private-anchor",
        "private-target",
        "private/target.jpg",
    ):
        assert hidden not in serialized


def test_supported_coordination_projection_exposes_only_public_reviewed_evidence() -> (
    None
):
    raw_output = {
        "query_asset_id": "asset.v2.private-anchor",
        "query_image_sha256": "a" * 64,
        "style_submode": "cross_category_coordination",
        "hits": [
            {
                "score": 0.9,
                "style_submode": "cross_category_coordination",
                "similarity_source": "verified_coordination_graph",
                "facet_evidence": [
                    {
                        "facet": "palette",
                        "value": "black",
                        "confidence": 0.95,
                        "provenance": "portfolio_curated_coordination_rule",
                        "source_record_id": "private-edge-001",
                    },
                    {
                        "facet": "coordination_rule",
                        "value": "neutral_palette_rule",
                        "confidence": 0.9,
                        "provenance": "portfolio_curated_coordination_rule",
                        "source_record_id": "private-edge-001",
                    },
                ],
                "product": {
                    "product_id": "abo:B07PRIVATE",
                    "title": "Black low-heel ankle boot",
                    "category_l1": "footwear",
                    "image_path": "private/catalog/boot.jpg",
                    "source": "abo",
                },
            }
        ],
    }

    public = _model_visible_tool_output(
        "style_similar_search", raw_output, call_index=5
    )
    cards, evidence = _visible_projection(
        "style_similar_search", raw_output, call_index=5
    )

    assert public["support_status"] == "supported"
    assert public["style_submode"] == "cross_category_coordination"
    candidate = public["candidates"][0]
    assert candidate["title"] == "Black low-heel ankle boot"
    assert candidate["category"] == "footwear"
    assert "similarity_source" not in candidate
    assert "score" not in candidate
    assert [item["facet"] for item in candidate["style_evidence"]] == [
        "palette",
        "coordination_rule",
    ]
    assert len(cards) == 1
    assert dict(cards[0].fields)["product_id"] == candidate["product_id"]
    assert "similarity_source" not in dict(cards[0].fields)
    assert "相关度" not in dict(cards[0].fields)
    serialized = json.dumps(public, ensure_ascii=False) + evidence.model_dump_json()
    for hidden in (
        "asset.v2.private-anchor",
        "private-edge-001",
        "abo:B07PRIVATE",
        "private/catalog/boot.jpg",
        "a" * 64,
    ):
        assert hidden not in serialized


def test_unsupported_style_coordination_returns_no_fabricated_candidate() -> None:
    raw_output = {
        "style_submode": "cross_category_coordination",
        "unsupported_reason": "private internal reason and path",
        "hits": [],
    }

    public = _model_visible_tool_output(
        "style_similar_search", raw_output, call_index=1
    )
    cards, evidence = _visible_projection("style_similar_search", raw_output)

    assert public == {
        "result_kind": "style_candidates",
        "support_status": "unsupported",
        "candidates": [],
        "style_submode": "cross_category_coordination",
        "unsupported_reason": (
            "verified cross-category coordination evidence is unavailable"
        ),
    }
    assert cards == ()
    assert "不支持跨品类协调推荐" in evidence.visible_text
    assert "private internal reason" not in json.dumps(public, ensure_ascii=False)


def test_multi_mapping_deduplicates_candidates_and_marks_unresolved_items() -> None:
    shared_hit = {
        "score": 1.0,
        "product": {
            "product_id": "rpc:50",
            "title": "RPC catalog class 50_instant_noodles",
            "category_l1": "instant_noodles",
            "source": "rpc",
        },
    }
    raw_output = {
        "objects": [
            {"label_zh": "方便面", "hits": [shared_hit]},
            {"label_zh": "方便面", "hits": [shared_hit]},
            {"label_zh": "未知包装", "hits": []},
        ]
    }

    public = _model_visible_tool_output(
        "multi_product_search", raw_output, call_index=2
    )
    cards, evidence = _visible_projection(
        "multi_product_search", raw_output, call_index=2
    )

    assert [item["status"] for item in public["items"]] == [
        "matched",
        "matched",
        "unresolved",
    ]
    assert [item["candidate_ordinal"] for item in public["items"]] == [1, 1, None]
    assert public["items"][0]["candidate"] == public["items"][1]["candidate"]
    assert len(cards) == 1
    fields = dict(cards[0].fields)
    assert fields["item_refs"] == "item-001, item-002"
    assert fields["quantity"] == "2"
    assert "item-003 | 未知包装 | unresolved | candidate-none" in (
        evidence.visible_text
    )


def test_twelve_multi_product_handles_match_visible_card_surface_exactly() -> None:
    raw_output = {
        "objects": [
            {
                "label_zh": f"商品 {ordinal}",
                "hits": [
                    {
                        "score": 1.0 - ordinal / 100,
                        "product": {
                            "product_id": f"private:{ordinal}",
                            "title": f"公开标题 {ordinal}",
                            "category_l1": "catalog_product",
                        },
                    }
                ],
            }
            for ordinal in range(1, 13)
        ]
    }

    public = _model_visible_tool_output(
        "multi_product_search",
        raw_output,
        call_index=3,
    )
    cards, evidence = _visible_projection(
        "multi_product_search",
        raw_output,
        call_index=3,
    )

    model_handles = {
        (
            item["candidate"]["evidence_reference"],
            item["candidate"]["product_id"],
        )
        for item in public["items"]
        if item["candidate"] is not None
    }
    card_handles = {
        (
            dict(card.fields)["evidence_reference"],
            dict(card.fields)["product_id"],
        )
        for card in cards
    }
    assert len(public["items"]) == len(cards) == len(evidence.cards) == 12
    assert model_handles == card_handles


def test_ocr_projection_exposes_only_recognized_text() -> None:
    line = OCRLine(
        line_id=sha256_bytes(b"public-ocr-line"),
        text="Name: Ada / Phone: 123",
        polygon=((1.0, 1.0), (50.0, 1.0), (50.0, 8.0), (1.0, 8.0)),
        confidence=0.95,
    )
    raw_output = DocumentOCRResult(
        input_binding=OCRInputBinding(
            asset_id="asset.v2.private-ocr",
            image_sha256=sha256_bytes(b"private-ocr-image"),
            image_bytes=128,
            width=64,
            height=32,
            safety_decision="approved_no_pii",
            safety_approval_sha256=sha256_bytes(b"private-ocr-approval"),
        ),
        runtime_binding=ModelRuntimeBinding(
            artifact_kind="document_ocr",
            model_id="fixture-ocr",
            backend_name="fixture",
            backend_version="1.0",
            manifest_sha256=sha256_bytes(b"private-ocr-manifest"),
            artifacts=(),
        ),
        languages=("en",),
        full_text=line.text,
        lines=(line,),
        fields=(),
        truncated=False,
    )
    public = _model_visible_tool_output("document_ocr", raw_output, call_index=2)
    cards, evidence = _visible_projection(
        "document_ocr",
        raw_output,
        call_index=2,
    )

    assert cards == ()
    assert public == {
        "result_kind": "ocr_lines",
        "lines": [
            {
                "line_reference": "tool-call-2-line-1",
                "text": "Name: Ada / Phone: 123",
                "content_trust": "untrusted_document_text",
                "truncated": False,
                "fields": [],
            }
        ],
    }
    assert evidence.visible_text == ("[tool-call-2-line-1] Name: Ada / Phone: 123")
    assert "runtime_binding" not in evidence.model_dump_json()
    assert "asset.v2.private-ocr" not in evidence.model_dump_json()


def test_kb_projection_recursively_serializes_typed_hits() -> None:
    raw_output = [_KBHit(title="Potato salad", text="Boil, cool, and season.")]
    public = _model_visible_tool_output(
        "recipe_lookup",
        raw_output,
        call_index=4,
    )
    cards, evidence = _visible_projection(
        "recipe_lookup",
        raw_output,
        call_index=4,
    )

    assert cards == ()
    source = public["sources"][0]
    assert source["evidence_reference"] == "tool-call-4-source-1"
    assert evidence.visible_text == (
        "[tool-call-4-source-1] Potato salad：Boil, cool, and season."
    )
    assert source["evidence_reference"] in evidence.visible_text
