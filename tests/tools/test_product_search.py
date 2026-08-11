from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from skillchain.tools.contracts import ProductHit, StyleHit, validate_json_value
from skillchain.tools.product_index import ProductIndex
from skillchain.tools.product_search import (
    ProductSearchService,
    configure_product_search_service_factory,
    find_similar_styles,
    image_product_search,
    text_product_search,
)
from skillchain.tools.settings import EMBEDDING_DIMENSION


def _vector(*components: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    vector[: len(components)] = components
    return vector


class FakeBackend:
    def __init__(self, *, image_vector: np.ndarray, text_vector: np.ndarray) -> None:
        self.image_vector = image_vector
        self.text_vector = text_vector
        self.image_calls: list[list[Path]] = []
        self.text_calls: list[list[str]] = []

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        self.image_calls.append(list(paths))
        return np.vstack([self.image_vector for _ in paths]).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls.append(list(texts))
        return np.vstack([self.text_vector for _ in texts]).astype(np.float32)


class ReverseTieIndex:
    """Returns tied records in reverse order to expose FAISS-order dependence."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.calls: list[tuple[np.ndarray, int]] = []

    def search(self, query: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((query.copy(), limit))
        indices = np.arange(self.count - 1, -1, -1, dtype=np.int64)[:limit]
        return (
            np.zeros((len(query), len(indices)), dtype=np.float32),
            indices.reshape(1, -1),
        )


def _index(
    products, image_vectors: np.ndarray, text_vectors: np.ndarray
) -> ProductIndex:
    return ProductIndex(
        image_index=ReverseTieIndex(len(products)),
        text_index=ReverseTieIndex(len(products)),
        image_vectors=image_vectors.astype(np.float32),
        text_vectors=text_vectors.astype(np.float32),
        products=tuple(products),
        manifest={},
    )


def _service(
    products, image_vectors: np.ndarray, text_vectors: np.ndarray
) -> tuple[ProductSearchService, FakeBackend]:
    backend = FakeBackend(image_vector=_vector(1.0), text_vector=_vector(0.0, 1.0))
    return (
        ProductSearchService(
            _index(products, image_vectors, text_vectors),
            backend,
            allow_provisional_gallery=True,
        ),
        backend,
    )


def _write_png(path: Path) -> Path:
    Image.new("RGB", (2, 2), "red").save(path)
    return path


def test_image_search_uses_only_image_embeddings_and_breaks_ties_by_product_id(
    tmp_path: Path, product_factory
):
    products = [
        product_factory(product_id="zebra"),
        product_factory(product_id="alpha"),
    ]
    service, backend = _service(
        products,
        np.vstack([_vector(1.0), _vector(1.0)]),
        np.vstack([_vector(0.0, 1.0), _vector(0.0, 1.0)]),
    )
    image = _write_png(tmp_path / "query.png")

    result = service.image_product_search(image)

    assert [entry["product"]["product_id"] for entry in result] == ["alpha", "zebra"]
    assert [entry["score"] for entry in result] == [1.0, 1.0]
    assert backend.image_calls == [[image]]
    assert backend.text_calls == []
    assert service.index.image_index.calls[0][1] == len(products)
    assert {entry["artifact_binding"]["mode"] for entry in result} == {"provisional"}


def test_image_search_trace_keeps_binding_and_input_hashes_when_hits_are_empty(
    tmp_path: Path,
):
    service, _ = _service(
        [],
        np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32),
        np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32),
    )
    image = _write_png(tmp_path / "query.png")

    trace = service.trace_image_product_search(image)

    assert trace.artifact_binding.mode == "provisional"
    assert trace.hits == ()
    assert len(trace.query_input_sha256) == 64
    assert len(trace.query_vector_sha256) == 64


def test_image_search_rejects_non_unit_query_vectors(tmp_path: Path, product_factory):
    products = [product_factory()]
    service, backend = _service(
        products,
        np.vstack([_vector(1.0)]),
        np.vstack([_vector(0.0, 1.0)]),
    )
    backend.image_vector = _vector(2.0)

    with pytest.raises(ValueError, match="unit normalized"):
        service.image_product_search(_write_png(tmp_path / "query.png"))


def test_image_search_rejects_symlink_and_mutation_during_embedding(
    tmp_path: Path, product_factory
):
    products = [product_factory()]
    image = _write_png(tmp_path / "query.png")
    service, backend = _service(
        products,
        np.vstack([_vector(1.0)]),
        np.vstack([_vector(0.0, 1.0)]),
    )

    link = tmp_path / "link.png"
    if hasattr(os, "symlink"):
        try:
            link.symlink_to(image)
        except OSError:
            pass
        else:
            with pytest.raises(ValueError, match="non-symlink"):
                service.image_product_search(link)

    original_embed = backend.embed_images

    def mutating_embed(paths):
        vectors = original_embed(paths)
        Image.new("RGB", (2, 2), "blue").save(paths[0])
        return vectors

    backend.embed_images = mutating_embed
    with pytest.raises(ValueError, match="changed during embedding"):
        service.image_product_search(image)


def test_image_trace_prefers_the_already_validated_byte_snapshot(
    tmp_path: Path, product_factory
):
    products = [product_factory()]
    image = _write_png(tmp_path / "query.png")
    service, backend = _service(
        products,
        np.vstack([_vector(1.0)]),
        np.vstack([_vector(0.0, 1.0)]),
    )
    snapshots: list[bytes] = []

    def embed_image_bytes(images):
        snapshots.extend(images)
        return np.vstack([_vector(1.0) for _ in images])

    def reject_path_reopen(_paths):
        raise AssertionError("byte-capable backends must not reopen the path")

    backend.embed_image_bytes = embed_image_bytes
    backend.embed_images = reject_path_reopen
    trace = service.trace_image_product_search(image, asset_id="asset-1")

    assert snapshots == [image.read_bytes()]
    assert trace.query_asset_id == "asset-1"
    assert trace.query_input_sha256 == hashlib.sha256(snapshots[0]).hexdigest()


def test_search_service_rejects_unbound_index_without_explicit_provisional_flag(
    product_factory,
):
    products = [product_factory()]
    index = _index(
        products,
        np.vstack([_vector(1.0)]),
        np.vstack([_vector(0.0, 1.0)]),
    )

    with pytest.raises(ValueError, match="provisional|eligibility"):
        ProductSearchService(
            index,
            FakeBackend(image_vector=_vector(1.0), text_vector=_vector(0.0, 1.0)),
        )


def test_text_search_uses_only_text_embeddings_returns_a_stable_top_ten_and_json(
    product_factory,
):
    products = [product_factory(product_id=f"p-{index:02d}") for index in range(12)]
    service, backend = _service(
        products,
        np.vstack([_vector(1.0) for _ in products]),
        np.vstack([_vector(0.0, 1.0) for _ in products]),
    )

    first = service.text_product_search("dress")
    second = service.text_product_search("dress")

    assert [entry["product"]["product_id"] for entry in first] == [
        f"p-{index:02d}" for index in range(10)
    ]
    assert [entry["rank"] for entry in first] == list(range(1, 11))
    assert len(first) == 10
    assert backend.text_calls == [["dress"], ["dress"]]
    assert backend.image_calls == []
    assert service.index.text_index.calls[0][1] == len(products)
    assert validate_json_value(first) == first
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )
    assert [ProductHit.model_validate(item).model_dump_json() for item in first] == [
        ProductHit.model_validate(item).model_dump_json() for item in second
    ]


def test_search_scores_are_rounded_to_eight_decimal_places(
    tmp_path: Path, product_factory
):
    score = 0.123456789
    service, _ = _service(
        [product_factory()],
        np.vstack([_vector(score, (1.0 - score**2) ** 0.5)]),
        np.vstack([_vector(0.0, 1.0)]),
    )

    result = service.image_product_search(_write_png(tmp_path / "query.png"))

    assert result[0]["score"] == 0.12345679


def test_text_search_rejects_blank_queries_without_embedding(product_factory):
    service, backend = _service(
        [product_factory()], np.vstack([_vector(1.0)]), np.vstack([_vector(0.0, 1.0)])
    )

    with pytest.raises(ValueError, match="blank"):
        service.text_product_search(" \t\n")

    assert backend.text_calls == []


def test_image_search_rejects_missing_invalid_and_oversize_images_before_embedding(
    tmp_path: Path, product_factory
):
    service, backend = _service(
        [product_factory()], np.vstack([_vector(1.0)]), np.vstack([_vector(0.0, 1.0)])
    )
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    oversized = tmp_path / "large.png"
    oversized.write_bytes(b"0" * (10 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match="does not exist"):
        service.image_product_search(tmp_path / "missing.png")
    with pytest.raises(ValueError, match="cannot be decoded"):
        service.image_product_search(invalid)
    with pytest.raises(ValueError, match="10 MiB"):
        service.image_product_search(oversized)

    assert backend.image_calls == []


def test_module_functions_use_an_explicit_injected_service_factory(
    tmp_path: Path, product_factory
):
    products = [product_factory(source="mep3m")]
    service, backend = _service(
        products, np.vstack([_vector(1.0)]), np.vstack([_vector(0.0, 1.0)])
    )
    image = _write_png(tmp_path / "query.png")
    configure_product_search_service_factory(lambda: service)
    try:
        assert text_product_search("dress")[0]["product"]["product_id"] == "product-1"
        assert image_product_search(image)[0]["product"]["product_id"] == "product-1"
        assert find_similar_styles(image) == []
    finally:
        configure_product_search_service_factory(None)

    assert backend.text_calls == [["dress"]]
    assert backend.image_calls == [[image], [image]]


def test_style_search_uses_grounded_anchor_filters_category_and_prefers_diversity(
    tmp_path: Path, product_factory
):
    products = [
        product_factory(product_id="muge-top", source="muge", category_l1="unknown"),
        product_factory(product_id="anchor", source="fashioniq", category_l1="dress"),
        product_factory(product_id="a-similar", source="muge", category_l1="dress"),
        product_factory(product_id="b-redundant", source="muge", category_l1="dress"),
        product_factory(product_id="z-diverse", source="muge", category_l1="dress"),
        product_factory(
            product_id="other-category", source="muge", category_l1="shoes"
        ),
    ]
    similar = _vector(0.99, (1.0 - 0.99**2) ** 0.5)
    vectors = np.vstack(
        [
            _vector(1.0),
            _vector(0.995, (1.0 - 0.995**2) ** 0.5),
            similar,
            _vector(0.98, (1.0 - 0.98**2) ** 0.5),
            _vector(0.7, 0.0, (1.0 - 0.7**2) ** 0.5),
            _vector(0.999, 0.0, 0.0, (1.0 - 0.999**2) ** 0.5),
        ]
    )
    service, _ = _service(products, vectors, vectors)

    result = service.find_similar_styles(_write_png(tmp_path / "query.png"))

    assert [entry["product"]["product_id"] for entry in result[:2]] == [
        "a-similar",
        "z-diverse",
    ]
    query = _vector(1.0)
    assert result[0]["mmr_score"] == round(0.5 * float(np.dot(query, vectors[2])), 8)
    assert result[1]["mmr_score"] == round(
        0.5 * float(np.dot(query, vectors[4]))
        - 0.5 * float(np.dot(vectors[4], vectors[2])),
        8,
    )
    assert {entry["product"]["category_l1"] for entry in result} == {"dress"}
    assert "anchor" not in {entry["product"]["product_id"] for entry in result}
    assert {entry["anchor_category_l1"] for entry in result} == {"dress"}
    assert all(StyleHit.model_validate(entry).model_dump_json() for entry in result)


def test_style_search_skips_unknown_categories_and_errors_without_a_reliable_one(
    tmp_path: Path, product_factory
):
    products = [
        product_factory(product_id="muge-top", source="muge", category_l1="unknown"),
        product_factory(product_id="unknown", source="mep3m", category_l1="unknown"),
        product_factory(
            product_id="reliable", source="fashioniq", category_l1="fashion"
        ),
        product_factory(product_id="candidate", source="muge", category_l1="fashion"),
    ]
    vectors = np.vstack(
        [
            _vector(1.0),
            _vector(0.999, (1.0 - 0.999**2) ** 0.5),
            _vector(0.998, (1.0 - 0.998**2) ** 0.5),
            _vector(0.9, (1.0 - 0.9**2) ** 0.5),
        ]
    )
    service, _ = _service(products, vectors, vectors)
    image = _write_png(tmp_path / "query.png")

    result = service.find_similar_styles(image)

    assert [entry["product"]["product_id"] for entry in result] == ["candidate"]
    assert result[0]["anchor_category_l1"] == "fashion"

    unreliable, _ = _service(products[:2], vectors[:2], vectors[:2])
    with pytest.raises(ValueError, match="reliable anchor"):
        unreliable.find_similar_styles(image)


def test_style_search_breaks_mmr_ties_by_product_id_and_limits_to_ten(
    tmp_path: Path, product_factory
):
    products = [
        product_factory(product_id="anchor", source="mep3m", category_l1="dress")
    ]
    products.extend(
        product_factory(product_id=f"candidate-{index:02d}", category_l1="dress")
        for index in range(11, -1, -1)
    )
    candidate = _vector(0.8, 0.6)
    vectors = np.vstack([_vector(1.0), *[candidate for _ in range(12)]])
    service, _ = _service(products, vectors, vectors)

    result = service.find_similar_styles(_write_png(tmp_path / "query.png"))

    assert len(result) == 10
    assert [entry["product"]["product_id"] for entry in result] == [
        f"candidate-{index:02d}" for index in range(10)
    ]
    assert result[0]["mmr_score"] == round(
        0.5 * float(np.dot(_vector(1.0), candidate)), 8
    )
