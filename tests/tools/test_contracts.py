import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from skillchain.config import DATA_DIR, RUNS_DIR
from skillchain.schemas import Product
from skillchain.tools import (
    Detection,
    ProductHit,
    ProductSearchTrace,
    RetrievalArtifactBinding,
    StyleFacetEvidence,
    StyleHit,
    validate_json_value,
)
from skillchain.tools.settings import (
    DASHSCOPE_EMBEDDING_ENDPOINT,
    DETECTOR_MANIFEST_PATH,
    DETECTOR_MODEL_PATH,
    EMBEDDING_CACHE_PATH,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    GOLD_RESULTS_PATH,
    IMAGE_BATCH_SIZE,
    KB_INDEX_DIR,
    KB_CATALOG_DIR,
    KB_K,
    KB_SOURCE_DIR,
    MMR_LAMBDA,
    OCR_MANIFEST_PATH,
    OCR_MODEL_DIR,
    PRODUCT_INDEX_DIR,
    PRODUCT_K,
    PRODUCT_QUERY_ARTIFACT_PATH,
    PRODUCTS_SOURCE_PATH,
    TEXT_BATCH_SIZE,
    TOOL_REGISTRY_MANIFEST_PATH,
    RETRIEVAL_RUNS_DIR,
)

PROVISIONAL_BINDING = RetrievalArtifactBinding(mode="provisional")


def test_phase2_settings_are_pinned():
    assert EMBEDDING_MODEL == "qwen3-vl-embedding"
    assert EMBEDDING_DIMENSION == 1024
    assert IMAGE_BATCH_SIZE == 5
    assert TEXT_BATCH_SIZE == 20
    assert PRODUCT_K == 10
    assert KB_K == 5
    assert MMR_LAMBDA == 0.5
    assert DASHSCOPE_EMBEDDING_ENDPOINT == (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )


def test_phase2_artifact_paths_are_pinned():
    assert PRODUCTS_SOURCE_PATH == DATA_DIR / "clean" / "products.parquet"
    assert PRODUCT_INDEX_DIR == DATA_DIR / "index" / "products"
    assert (
        PRODUCT_QUERY_ARTIFACT_PATH == DATA_DIR / "queries" / "split_assignment.jsonl"
    )
    assert KB_SOURCE_DIR == DATA_DIR / "kb"
    assert KB_INDEX_DIR == DATA_DIR / "index" / "kb"
    assert KB_CATALOG_DIR == DATA_DIR / "catalog" / "kb"
    assert EMBEDDING_CACHE_PATH == DATA_DIR / "index" / "embedding_cache.sqlite3"
    assert DETECTOR_MODEL_PATH == DATA_DIR / "models" / "yolo11n.pt"
    assert DETECTOR_MANIFEST_PATH == DATA_DIR / "models" / "detector-manifest.json"
    assert OCR_MODEL_DIR == DATA_DIR / "models" / "ocr"
    assert OCR_MANIFEST_PATH == DATA_DIR / "models" / "ocr-manifest.json"
    assert TOOL_REGISTRY_MANIFEST_PATH == DATA_DIR / "index" / "tool-registry.json"
    assert RETRIEVAL_RUNS_DIR == RUNS_DIR / "retrieval"
    assert GOLD_RESULTS_PATH == RUNS_DIR / "phase2" / "gold_results.json"
    assert all(
        isinstance(path, Path)
        for path in (
            PRODUCTS_SOURCE_PATH,
            PRODUCT_INDEX_DIR,
            PRODUCT_QUERY_ARTIFACT_PATH,
            KB_SOURCE_DIR,
            KB_INDEX_DIR,
            KB_CATALOG_DIR,
            EMBEDDING_CACHE_PATH,
            DETECTOR_MODEL_PATH,
            DETECTOR_MANIFEST_PATH,
            OCR_MODEL_DIR,
            OCR_MANIFEST_PATH,
            TOOL_REGISTRY_MANIFEST_PATH,
            RETRIEVAL_RUNS_DIR,
            GOLD_RESULTS_PATH,
        )
    )


def test_retrieval_binding_requires_all_verified_hashes_and_keeps_provisional_empty():
    verified = RetrievalArtifactBinding(
        mode="verified",
        index_integrity_sha256="1" * 64,
        eligibility_sha256="2" * 64,
        query_artifact_sha256="3" * 64,
        gallery_artifact_sha256="4" * 64,
        asset_catalog_sha256="5" * 64,
        products_parquet_sha256="6" * 64,
        leakage_policy_version="dataset-asset-components-v1",
    )

    assert verified.mode == "verified"
    with pytest.raises(ValidationError, match="完整 artifact"):
        RetrievalArtifactBinding(mode="verified")
    with pytest.raises(ValidationError, match="不得伪装"):
        RetrievalArtifactBinding(
            mode="provisional",
            eligibility_sha256="2" * 64,
        )


def test_product_hit_accepts_valid_product(product: Product):
    hit = ProductHit(
        rank=1,
        score=0.75,
        product=product,
        artifact_binding=PROVISIONAL_BINDING,
    )

    assert hit.rank == 1
    assert hit.score == 0.75
    assert hit.product is product


def test_product_trace_rejects_mixed_binding_and_noncontiguous_ranks(
    product: Product,
):
    verified = RetrievalArtifactBinding(
        mode="verified",
        index_integrity_sha256="1" * 64,
        eligibility_sha256="2" * 64,
        query_artifact_sha256="3" * 64,
        gallery_artifact_sha256="4" * 64,
        asset_catalog_sha256="5" * 64,
        products_parquet_sha256="6" * 64,
        leakage_policy_version="dataset-asset-components-v1",
    )
    common = {
        "tool_name": "image_product_search",
        "input_kind": "image",
        "query_input_sha256": "7" * 64,
        "query_image_sha256": "7" * 64,
        "query_asset_id": "asset-1",
        "query_vector_sha256": "8" * 64,
        "artifact_binding": verified,
    }
    with pytest.raises(ValidationError, match="inherit"):
        ProductSearchTrace(
            **common,
            hits=(
                ProductHit(
                    rank=1,
                    score=0.75,
                    product=product,
                    artifact_binding=PROVISIONAL_BINDING,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="contiguous"):
        ProductSearchTrace(
            **common,
            hits=(
                ProductHit(
                    rank=2,
                    score=0.75,
                    product=product,
                    artifact_binding=verified,
                ),
            ),
        )


@pytest.mark.parametrize("rank", [0, -1])
def test_product_hit_rejects_invalid_rank(product: Product, rank: int):
    with pytest.raises(ValidationError):
        ProductHit(
            rank=rank,
            score=0.75,
            product=product,
            artifact_binding=PROVISIONAL_BINDING,
        )


def test_product_hit_rejects_string_rank(product: Product):
    with pytest.raises(ValidationError):
        ProductHit(
            rank="1",
            score=0.75,
            product=product,
            artifact_binding=PROVISIONAL_BINDING,
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_product_hit_rejects_non_finite_score(product: Product, score: float):
    with pytest.raises(ValidationError):
        ProductHit(
            rank=1,
            score=score,
            product=product,
            artifact_binding=PROVISIONAL_BINDING,
        )


def test_style_hit_accepts_valid_scores_and_anchor(product: Product):
    hit = StyleHit(
        rank=1,
        score=0.75,
        product=product,
        mmr_score=0.5,
        anchor_category_l1="女装",
        artifact_binding=PROVISIONAL_BINDING,
    )

    assert hit.mmr_score == 0.5
    assert hit.anchor_category_l1 == "女装"


def test_style_hit_accepts_complete_inspectable_evidence(product: Product):
    evidence = StyleFacetEvidence(
        facet="relative_style_change",
        value="brighter red with shorter sleeves",
        confidence=0.8,
        provenance="fashioniq_relative_caption",
        source_record_id="dress:anchor->target",
    )
    hit = StyleHit(
        rank=1,
        score=0.8,
        product=product,
        mmr_score=0.55,
        anchor_category_l1=product.category_l1,
        artifact_binding=PROVISIONAL_BINDING,
        style_submode="same_category_alternative",
        similarity_source="fashioniq_relative_caption_graph",
        facet_evidence=(evidence,),
    )

    assert hit.facet_evidence == (evidence,)


def test_style_hit_accepts_curated_cross_category_evidence(product_factory):
    candidate = product_factory(category_l1="footwear", title="黑色低跟短靴")
    evidence = StyleFacetEvidence(
        facet="palette_coordination",
        value="neutral black candidate with the reviewed anchor palette",
        confidence=0.85,
        provenance="portfolio_curated_coordination_rule",
        source_record_id="coordination-edge-v1-001",
    )

    hit = StyleHit(
        rank=1,
        score=0.85,
        product=candidate,
        mmr_score=0.7,
        anchor_category_l1="dress",
        artifact_binding=PROVISIONAL_BINDING,
        style_submode="cross_category_coordination",
        similarity_source="verified_coordination_graph",
        facet_evidence=(evidence,),
    )

    assert hit.product.category_l1 == "footwear"
    assert hit.facet_evidence == (evidence,)


@pytest.mark.parametrize(
    ("candidate_category", "similarity_source", "provenance"),
    [
        (
            "dress",
            "verified_coordination_graph",
            "portfolio_curated_coordination_rule",
        ),
        (
            "footwear",
            "local_image_feature_cosine",
            "portfolio_curated_coordination_rule",
        ),
        (
            "footwear",
            "verified_coordination_graph",
            "local_image_feature",
        ),
    ],
)
def test_style_hit_rejects_mislabeled_cross_category_evidence(
    product_factory,
    candidate_category: str,
    similarity_source: str,
    provenance: str,
):
    candidate = product_factory(category_l1=candidate_category)
    evidence = StyleFacetEvidence(
        facet="palette_coordination",
        value="reviewed relation",
        confidence=0.8,
        provenance=provenance,
        source_record_id="coordination-edge-v1-001",
    )

    with pytest.raises(ValidationError):
        StyleHit(
            rank=1,
            score=0.8,
            product=candidate,
            mmr_score=0.6,
            anchor_category_l1="dress",
            artifact_binding=PROVISIONAL_BINDING,
            style_submode="cross_category_coordination",
            similarity_source=similarity_source,
            facet_evidence=(evidence,),
        )


def test_style_hit_rejects_partial_extended_evidence(product: Product):
    with pytest.raises(ValidationError):
        StyleHit(
            rank=1,
            score=0.8,
            product=product,
            mmr_score=0.55,
            anchor_category_l1="dress",
            artifact_binding=PROVISIONAL_BINDING,
            style_submode="same_category_alternative",
        )


@pytest.mark.parametrize("mmr_score", [math.nan, math.inf, -math.inf])
def test_style_hit_rejects_non_finite_mmr_score(product: Product, mmr_score: float):
    with pytest.raises(ValidationError):
        StyleHit(
            rank=1,
            score=0.75,
            product=product,
            mmr_score=mmr_score,
            anchor_category_l1="女装",
            artifact_binding=PROVISIONAL_BINDING,
        )


def test_style_hit_rejects_empty_anchor_category(product: Product):
    with pytest.raises(ValidationError):
        StyleHit(
            rank=1,
            score=0.75,
            product=product,
            mmr_score=0.5,
            anchor_category_l1="",
            artifact_binding=PROVISIONAL_BINDING,
        )


def test_detection_accepts_valid_box():
    detection = Detection(
        label="dress",
        label_zh="连衣裙",
        bbox=(1.0, 2.0, 3.0, 4.0),
        confidence=0.9,
    )

    assert detection.bbox == (1.0, 2.0, 3.0, 4.0)


@pytest.mark.parametrize("field", ["label", "label_zh"])
def test_detection_rejects_empty_labels(field: str):
    values = {
        "label": "dress",
        "label_zh": "连衣裙",
        "bbox": (1.0, 2.0, 3.0, 4.0),
        "confidence": 0.9,
    }
    values[field] = ""

    with pytest.raises(ValidationError):
        Detection.model_validate(values)


@pytest.mark.parametrize(
    "bbox",
    [
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0, 4.0, 5.0),
        (3.0, 2.0, 1.0, 4.0),
        (1.0, 4.0, 3.0, 2.0),
        (math.nan, 2.0, 3.0, 4.0),
        (1.0, math.inf, 3.0, 4.0),
        (1.0, 2.0, -math.inf, 4.0),
        (1.0, 2.0, 1.0, 4.0),
        (-1.0, 2.0, 3.0, 4.0),
    ],
)
def test_detection_rejects_invalid_bbox(bbox: tuple[float, ...]):
    with pytest.raises(ValidationError):
        Detection(
            label="dress",
            label_zh="连衣裙",
            bbox=bbox,
            confidence=0.9,
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_detection_rejects_invalid_confidence(confidence: float):
    with pytest.raises(ValidationError):
        Detection(
            label="dress",
            label_zh="连衣裙",
            bbox=(1.0, 2.0, 3.0, 4.0),
            confidence=confidence,
        )


def test_contract_models_reject_extra_fields(product: Product):
    valid_models_and_values = [
        (
            ProductHit,
            {
                "rank": 1,
                "score": 0.75,
                "product": product,
                "artifact_binding": PROVISIONAL_BINDING,
            },
        ),
        (
            StyleHit,
            {
                "rank": 1,
                "score": 0.75,
                "product": product,
                "mmr_score": 0.5,
                "anchor_category_l1": "女装",
                "artifact_binding": PROVISIONAL_BINDING,
            },
        ),
        (
            Detection,
            {
                "label": "dress",
                "label_zh": "连衣裙",
                "bbox": (1.0, 2.0, 3.0, 4.0),
                "confidence": 0.9,
            },
        ),
    ]

    for model, values in valid_models_and_values:
        with pytest.raises(ValidationError):
            model.model_validate({**values, "unexpected": True})


def test_validate_json_value_accepts_recursive_json_values():
    value = {
        "name": "连衣裙",
        "rank": 1,
        "score": 0.75,
        "available": True,
        "metadata": None,
        "items": ["blue", {"count": 2}],
    }

    assert validate_json_value(value) == value


def test_validate_json_value_allows_shared_non_cyclic_containers():
    shared = ["blue"]
    value = {"first": shared, "second": shared}

    assert validate_json_value(value) == value


def test_validate_json_value_rejects_self_referencing_list():
    value: list[object] = []
    value.append(value)

    with pytest.raises((TypeError, ValueError)):
        validate_json_value(value)


def test_validate_json_value_rejects_self_referencing_dict():
    value: dict[str, object] = {}
    value["self"] = value

    with pytest.raises((TypeError, ValueError)):
        validate_json_value(value)


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        {"score": math.nan},
        [1, math.inf],
        {1: "non-string key"},
        ("tuple",),
    ],
)
def test_validate_json_value_rejects_non_json_values(value: object):
    with pytest.raises((TypeError, ValueError)):
        validate_json_value(value)
