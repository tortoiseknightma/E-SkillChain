"""Deterministic product search helpers over the audited dual-modal index."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np
from PIL import Image, UnidentifiedImageError

from skillchain.schemas import Product
from skillchain.tools.contracts import (
    ProductHit,
    ProductSearchTrace,
    RetrievalArtifactBinding,
    StyleHit,
    validate_json_value,
)
from skillchain.tools.embedding import (
    CachedEmbeddingBackend,
    DashScopeEmbeddingClient,
    EmbeddingBackend,
    EmbeddingCache,
    embedding_execution_location,
)
from skillchain.tools.product_index import ProductIndex
from skillchain.tools.settings import (
    EMBEDDING_CACHE_PATH,
    MMR_LAMBDA,
    PRODUCT_INDEX_DIR,
    PRODUCT_K,
    PRODUCT_QUERY_ARTIFACT_PATH,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    read_stable_regular_file,
    sha256_bytes,
)

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_STYLE_CANDIDATE_K = 100

type ServiceFactory = Callable[[], "ProductSearchService"]
type RankedProduct = tuple[int, Product, float]


class ProductSearchService:
    """Search one validated index with an explicitly supplied embedding backend."""

    def __init__(
        self,
        index: ProductIndex,
        backend: EmbeddingBackend,
        *,
        query_artifact: str | Path | None = None,
        allow_provisional_gallery: bool = False,
    ) -> None:
        self.index = index
        self.backend = backend
        self.artifact_binding: RetrievalArtifactBinding = index.retrieval_binding(
            query_artifact,
            allow_provisional_gallery=allow_provisional_gallery,
        )
        if self.artifact_binding.mode == "verified":
            runtime_model = getattr(backend, "model", None)
            expected_model = index.manifest.get("model")
            if runtime_model != expected_model:
                raise ValueError(
                    "verified product retrieval requires the index embedding model"
                )
            runtime_canary_text = getattr(backend, "canary_text", None)
            runtime_canary_vector = getattr(backend, "canary_vector", None)
            if (
                runtime_canary_text != index.canary_text
                or runtime_canary_vector is None
            ):
                raise ValueError(
                    "verified product retrieval requires the index canary baseline"
                )
            try:
                canary_matches = np.array_equal(
                    np.asarray(runtime_canary_vector, dtype=np.float32),
                    index.canary_vector,
                )
            except (TypeError, ValueError):
                canary_matches = False
            if not canary_matches:
                raise ValueError(
                    "verified product retrieval canary does not match the index"
                )

    @property
    def execution_location(self) -> str:
        return embedding_execution_location(self.backend)

    def image_product_search(self, image: str | Path) -> list[dict]:
        """Return the ten strongest image matches in a deterministic order."""
        trace = self.trace_image_product_search(image)
        return [_json_dict(hit.model_dump(mode="json")) for hit in trace.hits]

    def trace_image_product_search(
        self, image: str | Path, *, asset_id: str | None = None
    ) -> ProductSearchTrace:
        """Return image/vector hashes and a top-level binding, including on empty hits."""

        query, image_sha256 = self._embed_image_with_snapshot(image)
        matches = self._rank_products(query, "image")[:PRODUCT_K]
        vector_bytes = np.ascontiguousarray(query, dtype=np.float32).tobytes()
        return ProductSearchTrace(
            tool_name="image_product_search",
            input_kind="image",
            query_input_sha256=image_sha256,
            query_asset_id=asset_id or f"sha256:{image_sha256}",
            query_vector_sha256=sha256_bytes(vector_bytes),
            artifact_binding=self.artifact_binding,
            hits=self._product_hit_models(matches),
        )

    def text_product_search(self, query: str) -> list[dict]:
        """Return the ten strongest text matches in a deterministic order."""
        if not isinstance(query, str):
            raise TypeError("text query must be a string")
        if not query.strip():
            raise ValueError("text query must not be blank")
        trace = self.trace_text_product_search(query)
        return [_json_dict(hit.model_dump(mode="json")) for hit in trace.hits]

    def trace_text_product_search(self, query: str) -> ProductSearchTrace:
        """Return the exact text input, vector hash, binding, and ranked hits."""
        if not isinstance(query, str):
            raise TypeError("text query must be a string")
        if not query.strip():
            raise ValueError("text query must not be blank")
        vector = self._single_query_vector(self.backend.embed_texts([query]), "text")
        matches = self._rank_products(vector, "text")[:PRODUCT_K]
        return ProductSearchTrace(
            tool_name="text_product_search",
            input_kind="text",
            query_input_sha256=sha256_bytes(query.encode("utf-8")),
            query_text=query,
            query_vector_sha256=_vector_sha256(vector),
            artifact_binding=self.artifact_binding,
            hits=self._product_hit_models(matches),
        )

    def find_similar_styles(self, image: str | Path) -> list[dict]:
        """Diversify same-category products around a catalog-grounded anchor."""
        trace = self.trace_similar_styles(image)
        return [_json_dict(hit.model_dump(mode="json")) for hit in trace.hits]

    def trace_similar_styles(
        self, image: str | Path, *, asset_id: str | None = None
    ) -> ProductSearchTrace:
        """Return style ranks together with the immutable image/vector identity."""
        query, image_sha256 = self._embed_image_with_snapshot(image)
        image_matches = self._rank_products(query, "image")
        category_anchors = tuple(
            match
            for match in image_matches
            if _normalized_category(match[1]) is not None
        )
        anchor = next(
            (
                match
                for match in category_anchors
                if any(
                    other[1].product_id != match[1].product_id
                    and _normalized_category(other[1]) == _normalized_category(match[1])
                    for other in image_matches
                )
            ),
            category_anchors[0] if category_anchors else None,
        )
        if anchor is None:
            raise ValueError("image search did not yield a reliable anchor")

        anchor_row, anchor_product, _ = anchor
        anchor_category = _normalized_category(anchor_product)
        if anchor_category is None:
            raise AssertionError("validated anchor must have a category")

        candidates = [
            match
            for match in image_matches
            if match[0] != anchor_row
            and match[1].product_id != anchor_product.product_id
            and _normalized_category(match[1]) == anchor_category
        ][:_STYLE_CANDIDATE_K]
        selected_rows: list[int] = []
        selected: list[tuple[int, Product, float, float]] = []
        remaining = list(candidates)

        while remaining and len(selected) < PRODUCT_K:
            ranked_mmr: list[tuple[int, Product, float, float]] = []
            for row, product, relevance in remaining:
                candidate_vector = self.index.image_vectors[row]
                relevance = float(np.dot(query, candidate_vector))
                max_selected_similarity = max(
                    (
                        float(
                            np.dot(
                                candidate_vector, self.index.image_vectors[selected_row]
                            )
                        )
                        for selected_row in selected_rows
                    ),
                    default=0.0,
                )
                mmr_score = (
                    MMR_LAMBDA * relevance
                    - (1.0 - MMR_LAMBDA) * max_selected_similarity
                )
                ranked_mmr.append((row, product, relevance, mmr_score))

            ranked_mmr.sort(
                key=lambda item: (-_round_score(item[3]), item[1].product_id)
            )
            best = ranked_mmr[0]
            selected.append(best)
            selected_rows.append(best[0])
            remaining = [match for match in remaining if match[0] != best[0]]

        hits = self._style_hit_models(selected, anchor_category)
        return ProductSearchTrace(
            tool_name="style_similar_search",
            input_kind="image",
            query_input_sha256=image_sha256,
            query_asset_id=asset_id or f"sha256:{image_sha256}",
            query_vector_sha256=_vector_sha256(query),
            artifact_binding=self.artifact_binding,
            hits=hits,
        )

    @property
    def formal_runtime_binding_sha256(self) -> str:
        """Derive formal identity from the live index, backend, and cache policy."""
        if self.artifact_binding.mode != "verified":
            raise ValueError("formal runtime identity requires a verified gallery")
        self.index.require_formal_verified()
        backend_binding = getattr(self.backend, "runtime_binding_sha256", None)
        if not isinstance(backend_binding, str) or len(backend_binding) != 64:
            raise ValueError("formal runtime requires a bound embedding backend")
        if (
            self.index.manifest.get("embedding_runtime_binding_sha256")
            != backend_binding
        ):
            raise ValueError(
                "formal query runtime differs from the attested index build runtime"
            )
        if getattr(self.backend, "cache_policy", None) != "bypass":
            raise ValueError("formal retrieval must bypass mutable embedding caches")
        if getattr(self.backend, "formal_ready", False) is not True:
            raise ValueError("formal retrieval requires a freshly verified canary")
        try:
            import faiss

            image_faiss_sha256 = sha256_bytes(
                np.asarray(
                    faiss.serialize_index(self.index.image_index), dtype=np.uint8
                ).tobytes()
            )
            text_faiss_sha256 = sha256_bytes(
                np.asarray(
                    faiss.serialize_index(self.index.text_index), dtype=np.uint8
                ).tobytes()
            )
        except Exception as error:
            raise ValueError(
                "formal retrieval cannot snapshot live FAISS state"
            ) from error
        identity = {
            "artifact_binding": self.artifact_binding.model_dump(mode="json"),
            "backend_runtime_sha256": backend_binding,
            "canary_vector_sha256": self.index.manifest["canary"]["vector_sha256"],
            "dimension": self.index.manifest["dimension"],
            "index_embedding_execution_location": self.index.manifest[
                "embedding_execution_location"
            ],
            "index_integrity_sha256": self.index.manifest["integrity_sha256"],
            "image_vectors_sha256": _matrix_sha256(self.index.image_vectors),
            "image_faiss_sha256": image_faiss_sha256,
            "model": self.index.manifest["model"],
            "normalization": self.index.manifest["normalization"],
            "products_sha256": sha256_bytes(
                canonical_jsonl_bytes(
                    tuple(
                        product.model_dump(mode="json")
                        for product in self.index.products
                    )
                )
            ),
            "text_vectors_sha256": _matrix_sha256(self.index.text_vectors),
            "text_faiss_sha256": text_faiss_sha256,
            "query_embedding_execution_location": embedding_execution_location(
                self.backend
            ),
            "policy_version": "formal-product-runtime-v2",
        }
        return sha256_bytes(canonical_json_bytes(identity))

    def _embed_image(self, image: str | Path) -> np.ndarray:
        return self._embed_image_with_snapshot(image)[0]

    def _embed_image_with_snapshot(self, image: str | Path) -> tuple[np.ndarray, str]:
        path, before = _validated_image_snapshot(image)
        embed_bytes = getattr(self.backend, "embed_image_bytes", None)
        if callable(embed_bytes):
            embedded = embed_bytes([before])
        else:
            if self.artifact_binding.mode == "verified":
                raise ValueError(
                    "formal image retrieval requires immutable byte embedding"
                )
            embedded = self.backend.embed_images([path])
        vector = self._single_query_vector(embedded, "image")
        after = read_stable_regular_file(
            path,
            label="product query image",
            max_bytes=_MAX_IMAGE_BYTES,
        )
        if after != before:
            raise ValueError("product query image changed during embedding")
        return vector, sha256_bytes(before)

    def _single_query_vector(self, value: Any, modality: str) -> np.ndarray:
        try:
            vectors = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{modality} embedding response is not numeric") from error
        expected_dimension = self.index.image_vectors.shape[1]
        if vectors.shape != (1, expected_dimension):
            raise ValueError(
                f"{modality} embedding response must have shape (1, {expected_dimension})"
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError(
                f"{modality} embedding response contains non-finite values"
            )
        norm = float(np.linalg.norm(vectors[0]))
        if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError(f"{modality} embedding response must be unit normalized")
        return vectors[0]

    def _rank_products(self, query: np.ndarray, modality: str) -> list[RankedProduct]:
        if not self.index.products:
            return []
        if modality == "image":
            faiss_index = self.index.image_index
            vectors = self.index.image_vectors
        elif modality == "text":
            faiss_index = self.index.text_index
            vectors = self.index.text_vectors
        else:
            raise ValueError(f"unsupported product modality: {modality}")

        _, faiss_rows = faiss_index.search(
            query.reshape(1, -1), len(self.index.products)
        )
        rows = _valid_rows(faiss_rows, len(self.index.products))
        if len(rows) != len(self.index.products):
            raise ValueError("product index search did not return every product row")

        matches = [
            (row, self.index.products[row], float(np.dot(query, vectors[row])))
            for row in rows
        ]
        matches.sort(key=lambda item: (-_round_score(item[2]), item[1].product_id))
        return matches

    def _product_hits(self, matches: Sequence[RankedProduct]) -> list[dict]:
        return [
            _json_dict(hit.model_dump(mode="json"))
            for hit in self._product_hit_models(matches)
        ]

    def _product_hit_models(
        self, matches: Sequence[RankedProduct]
    ) -> tuple[ProductHit, ...]:
        return tuple(
            ProductHit(
                rank=rank,
                score=_round_score(score),
                product=product,
                artifact_binding=self.artifact_binding,
            )
            for rank, (_, product, score) in enumerate(matches, start=1)
        )

    def _style_hits(
        self, matches: Sequence[tuple[int, Product, float, float]], anchor_category: str
    ) -> list[dict]:
        return [
            _json_dict(hit.model_dump(mode="json"))
            for hit in self._style_hit_models(matches, anchor_category)
        ]

    def _style_hit_models(
        self,
        matches: Sequence[tuple[int, Product, float, float]],
        anchor_category: str,
    ) -> tuple[StyleHit, ...]:
        return tuple(
            StyleHit(
                rank=rank,
                score=_round_score(score),
                product=product,
                mmr_score=_round_score(mmr_score),
                anchor_category_l1=anchor_category,
                artifact_binding=self.artifact_binding,
            )
            for rank, (_, product, score, mmr_score) in enumerate(matches, start=1)
        )


def _validated_image_snapshot(image: str | Path) -> tuple[Path, bytes]:
    if not isinstance(image, (str, Path)):
        raise TypeError("image must be a path string or Path")
    path = Path(image)
    try:
        content = read_stable_regular_file(
            path,
            label="product query image",
            max_bytes=_MAX_IMAGE_BYTES,
        )
    except ValueError as error:
        if not path.exists():
            raise ValueError(f"image does not exist: {path}") from error
        if "exceeds" in str(error):
            raise ValueError("image must not exceed 10 MiB") from error
        raise ValueError("image must be a regular non-symlink file") from error
    try:
        with Image.open(BytesIO(content)) as decoded:
            decoded.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"image cannot be decoded: {path}") from error
    return path, content


def _valid_rows(rows: Any, product_count: int) -> list[int]:
    raw_rows = np.asarray(rows).reshape(-1)
    resolved: list[int] = []
    seen: set[int] = set()
    for raw_row in raw_rows:
        row = int(raw_row)
        if 0 <= row < product_count and row not in seen:
            resolved.append(row)
            seen.add(row)
    return resolved


def _normalized_category(product: Product) -> str | None:
    category = product.category_l1.strip()
    if not category or category.casefold() == "unknown":
        return None
    return category


def _round_score(score: float) -> float:
    return round(float(score), 8)


def _vector_sha256(vector: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(vector, dtype=np.float32).tobytes())


def _matrix_sha256(matrix: np.ndarray) -> str:
    value = np.asarray(matrix, dtype=np.float32)
    identity = {
        "bytes_sha256": sha256_bytes(np.ascontiguousarray(value).tobytes()),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    return sha256_bytes(canonical_json_bytes(identity))


def _json_dict(value: object) -> dict:
    validated = validate_json_value(value)
    if not isinstance(validated, dict):
        raise TypeError("tool results must be JSON objects")
    return cast(dict, validated)


_configured_factory: ServiceFactory | None = None
_service_singleton: ProductSearchService | None = None
_service_lock = Lock()


def configure_product_search_service_factory(factory: ServiceFactory | None) -> None:
    """Set a resettable service factory for explicit local/test dependency injection."""
    global _configured_factory, _service_singleton
    with _service_lock:
        _configured_factory = factory
        _service_singleton = None


def _default_service_factory() -> ProductSearchService:
    index = ProductIndex.load(PRODUCT_INDEX_DIR)
    backend = CachedEmbeddingBackend(
        DashScopeEmbeddingClient(),
        EmbeddingCache(EMBEDDING_CACHE_PATH),
        canary_text=index.canary_text,
        canary_vector=index.canary_vector,
    )
    return ProductSearchService(
        index,
        backend,
        query_artifact=PRODUCT_QUERY_ARTIFACT_PATH,
    )


def _get_service() -> ProductSearchService:
    global _service_singleton
    if _service_singleton is None:
        with _service_lock:
            if _service_singleton is None:
                factory = _configured_factory or _default_service_factory
                _service_singleton = factory()
    return _service_singleton


def image_product_search(image: str | Path) -> list[dict]:
    """Search products by a local image with the configured lazy singleton service."""
    return _get_service().image_product_search(image)


def text_product_search(query: str) -> list[dict]:
    """Search products by text with the configured lazy singleton service."""
    return _get_service().text_product_search(query)


def find_similar_styles(image: str | Path) -> list[dict]:
    """Return MMR-diversified products in the reliable anchor's category."""
    return _get_service().find_similar_styles(image)


__all__ = [
    "ProductSearchService",
    "configure_product_search_service_factory",
    "find_similar_styles",
    "image_product_search",
    "text_product_search",
]
