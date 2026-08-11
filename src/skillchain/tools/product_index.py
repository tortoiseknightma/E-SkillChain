"""Build and validate auditable dual-modal product indices."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import stat
import time
from typing import Any, Sequence

import faiss
import numpy as np
import pyarrow.parquet as pq

from skillchain.data import (
    discard_staging_directory,
    prepare_staging_directory,
    publish_staged_directory,
)
from skillchain.data.asset_catalog import AssetCatalog, GalleryAssetReference
from skillchain.data.gallery_eligibility import (
    ELIGIBILITY_POLICY_VERSION,
    VerifiedGalleryEligibility,
    load_gallery_eligibility_manifest,
    verify_gallery_eligibility,
)
from skillchain.schemas import Product
from skillchain.synthesis.store import (
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.tools.contracts import RetrievalArtifactBinding
from skillchain.tools.embedding import (
    EmbeddingBackend,
    embedding_execution_location,
    formal_embedding_runtime_binding,
)
from skillchain.tools.settings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    IMAGE_BATCH_SIZE,
    TEXT_BATCH_SIZE,
)

_CORE_ARTIFACT_NAMES = (
    "image.faiss",
    "text.faiss",
    "image_vectors.npy",
    "text_vectors.npy",
    "products.parquet",
)
_ELIGIBILITY_ARTIFACT = "query-gallery-eligibility.json"
_GALLERY_ARTIFACT = "gallery-assets.jsonl"
_VERIFIED_ARTIFACT_NAMES = _CORE_ARTIFACT_NAMES + (
    _ELIGIBILITY_ARTIFACT,
    _GALLERY_ARTIFACT,
)
_FORMAT_VERSION = 4
CANARY_TEXT = "skillchain product index canary v1"
_CANARY_TEXT_SHA256 = hashlib.sha256(CANARY_TEXT.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BuildProductIndexReport:
    """Stable progress fields emitted by the index build CLI."""

    products: int
    image_vectors: int
    text_vectors: int
    batches: dict[str, int]
    cache_hits: int
    api_calls: int
    cache_hits_by_modality: dict[str, int]
    api_calls_by_modality: dict[str, int]
    items_per_second: float
    elapsed_seconds: float
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "products": self.products,
            "image_vectors": self.image_vectors,
            "text_vectors": self.text_vectors,
            "batches": self.batches,
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "cache_hits_by_modality": self.cache_hits_by_modality,
            "api_calls_by_modality": self.api_calls_by_modality,
            "items_per_second": self.items_per_second,
            "elapsed_seconds": self.elapsed_seconds,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class ProductIndexCanary:
    """Validated canary baseline reusable without loading index artefacts."""

    text: str
    vector: np.ndarray


@dataclass(frozen=True)
class ProductIndex:
    """A validated, row-aligned image and text product index."""

    image_index: Any
    text_index: Any
    image_vectors: np.ndarray
    text_vectors: np.ndarray
    products: tuple[Product, ...]
    manifest: dict[str, Any]
    gallery_references: tuple[GalleryAssetReference, ...] | None = None
    root: Path | None = None
    external_manifest_sha256: str | None = None

    @property
    def canary_text(self) -> str:
        """Stable canary text that can configure later cache-backed consumers."""
        return CANARY_TEXT

    @property
    def canary_vector(self) -> np.ndarray:
        """Return a copy so callers cannot alter the validated manifest state."""
        return _canary_vector_from_manifest(self.manifest["canary"]).copy()

    @classmethod
    def load_canary(
        cls,
        output_dir: Path | str,
        *,
        allow_provisional_gallery: bool = False,
    ) -> ProductIndexCanary:
        """Read only a complete index manifest and validate its canary baseline."""
        manifest = _load_manifest(Path(output_dir) / "manifest.json")
        _validate_manifest(
            manifest,
            allow_provisional_gallery=allow_provisional_gallery,
        )
        if not manifest["complete"]:
            raise ValueError("manifest.json: index is incomplete")
        return ProductIndexCanary(
            text=CANARY_TEXT,
            vector=_canary_vector_from_manifest(manifest["canary"]).copy(),
        )

    @classmethod
    def load(
        cls,
        output_dir: Path | str,
        *,
        allow_incomplete: bool = False,
        allow_provisional_gallery: bool = False,
        expected_manifest_sha256: str | None = None,
    ) -> "ProductIndex":
        """Load only an index whose manifest and files pass all integrity checks."""
        output_dir = Path(output_dir)
        manifest_path = output_dir / "manifest.json"
        manifest = _load_manifest(manifest_path)
        manifest_file_sha256 = _sha256(manifest_path)
        if expected_manifest_sha256 is not None:
            if not _is_sha256(expected_manifest_sha256):
                raise ValueError("expected manifest sha256 is invalid")
            if manifest_file_sha256 != expected_manifest_sha256:
                raise ValueError("manifest.json: external digest mismatch")
        _validate_manifest(
            manifest,
            allow_provisional_gallery=allow_provisional_gallery,
        )
        if not manifest["complete"] and not allow_incomplete:
            raise ValueError("manifest.json: index is incomplete")

        artifact_names = _artifact_names_for_manifest(manifest)
        for filename in artifact_names:
            _verify_artifact(output_dir, filename, manifest["artifacts"][filename])

        expected_rows = manifest["product_count"]
        products = _load_products(output_dir / "products.parquet", expected_rows)
        image_vectors = _load_vectors(output_dir / "image_vectors.npy", expected_rows)
        text_vectors = _load_vectors(output_dir / "text_vectors.npy", expected_rows)
        image_index = _load_faiss_index(
            output_dir / "image.faiss", image_vectors, expected_rows
        )
        text_index = _load_faiss_index(
            output_dir / "text.faiss", text_vectors, expected_rows
        )
        gallery_references = _load_index_gallery_binding(
            output_dir,
            products,
            manifest,
        )
        for filename in artifact_names:
            _verify_artifact(output_dir, filename, manifest["artifacts"][filename])
        if _load_manifest(output_dir / "manifest.json") != manifest:
            raise ValueError("manifest.json: changed during index load")

        image_vectors.setflags(write=False)
        text_vectors.setflags(write=False)

        return cls(
            image_index=image_index,
            text_index=text_index,
            image_vectors=image_vectors,
            text_vectors=text_vectors,
            products=products,
            manifest=manifest,
            gallery_references=gallery_references,
            root=output_dir,
            external_manifest_sha256=expected_manifest_sha256,
        )

    def require_formal_verified(self) -> None:
        """Re-read an externally locked, attested index before formal use."""

        if self.root is None or self.external_manifest_sha256 is None:
            raise ValueError(
                "formal product index requires an external manifest digest"
            )
        if self.manifest.get("embedding_runtime_assurance") != "formal-attested":
            raise ValueError("formal product index requires an attested build runtime")
        reloaded = ProductIndex.load(
            self.root,
            expected_manifest_sha256=self.external_manifest_sha256,
        )
        if (
            reloaded.manifest != self.manifest
            or reloaded.products != self.products
            or reloaded.gallery_references != self.gallery_references
            or not np.array_equal(reloaded.image_vectors, self.image_vectors)
            or not np.array_equal(reloaded.text_vectors, self.text_vectors)
            or _faiss_sha256(reloaded.image_index) != _faiss_sha256(self.image_index)
            or _faiss_sha256(reloaded.text_index) != _faiss_sha256(self.text_index)
        ):
            raise ValueError("formal product index differs from its locked files")

    def retrieval_binding(
        self,
        query_artifact: Path | str | None,
        *,
        allow_provisional_gallery: bool = False,
    ) -> RetrievalArtifactBinding:
        """Bind a retrieval run to the exact query/index/gallery evidence."""

        gate = self.manifest.get("query_gallery_eligibility")
        if not isinstance(gate, dict) or gate.get("mode") == "provisional":
            if not allow_provisional_gallery:
                raise ValueError(
                    "product index uses a provisional gallery; explicit "
                    "allow_provisional_gallery=True is required"
                )
            return RetrievalArtifactBinding(mode="provisional")

        if query_artifact is None:
            raise ValueError("verified product retrieval requires a query artifact")
        try:
            query_sha256 = _sha256(Path(query_artifact))
        except OSError as error:
            raise ValueError(
                f"query artifact cannot be read: {Path(query_artifact)}"
            ) from error
        if query_sha256 != gate["query_artifact_sha256"]:
            raise ValueError("query artifact sha256 does not match the verified index")

        return RetrievalArtifactBinding(
            mode="verified",
            index_integrity_sha256=self.manifest["integrity_sha256"],
            eligibility_sha256=gate["eligibility_sha256"],
            query_artifact_sha256=query_sha256,
            gallery_artifact_sha256=gate["gallery_artifact_sha256"],
            asset_catalog_sha256=gate["asset_catalog_sha256"],
            products_parquet_sha256=self.manifest["artifacts"]["products.parquet"][
                "sha256"
            ],
            leakage_policy_version=gate["leakage_policy_version"],
        )


def build_product_index(
    products_path: Path | str,
    output_dir: Path | str,
    backend: EmbeddingBackend,
    limit: int | None = None,
    *,
    eligibility_manifest_path: Path | str | None = None,
    query_artifact_path: Path | str | None = None,
    gallery_artifact_path: Path | str | None = None,
    asset_catalog: AssetCatalog | None = None,
    allow_provisional_gallery: bool = False,
) -> BuildProductIndexReport:
    """Build, self-validate, then atomically publish a dual FAISS product index."""
    products_path = Path(products_path)
    output_dir = Path(output_dir)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    verified_eligibility = _resolve_build_eligibility(
        eligibility_manifest_path=eligibility_manifest_path,
        query_artifact_path=query_artifact_path,
        gallery_artifact_path=gallery_artifact_path,
        asset_catalog=asset_catalog,
        limit=limit,
        allow_provisional_gallery=allow_provisional_gallery,
    )

    started = time.perf_counter()
    source_bytes = _read_products_snapshot(products_path)
    source_sha256 = sha256_bytes(source_bytes)
    source_table = _read_products_table(source_bytes, products_path)
    source_rows = source_table.num_rows
    source_products = _validate_products(source_table, products_path)
    aligned_gallery = _validate_source_gallery_binding(
        source_products,
        verified_eligibility,
    )
    table = source_table if limit is None else source_table.slice(0, limit)
    products = source_products if limit is None else source_products[:limit]
    indexed_gallery = (
        None
        if aligned_gallery is None
        else aligned_gallery
        if limit is None
        else aligned_gallery[:limit]
    )
    execution_location = embedding_execution_location(backend)
    embedding_runtime_binding_sha256 = formal_embedding_runtime_binding(backend)
    embedding_runtime_assurance = (
        "formal-attested"
        if embedding_runtime_binding_sha256 is not None
        else "diagnostic-unattested"
    )
    _enforce_gallery_embedding_permissions(
        # Unknown/custom backends are conservatively treated as remote.  Their
        # self-declared ``local`` string is not a permission trust root for a
        # verified gallery.  An explicitly provisional limited smoke has no
        # formal or permission claim and retains diagnostic backend support.
        execution_location
        if embedding_runtime_binding_sha256 is not None or verified_eligibility is None
        else "remote",
        indexed_gallery,
        verified_eligibility,
    )
    image_paths = _image_paths(products, products_path)
    _verify_build_snapshots(
        products_path=products_path,
        products_bytes=source_bytes,
        eligibility_manifest_path=eligibility_manifest_path,
        query_artifact_path=query_artifact_path,
        gallery_artifact_path=gallery_artifact_path,
        verified=verified_eligibility,
    )
    cache_hits_before = _backend_counter(backend, "cache_hits")
    api_calls_before = _backend_counter(backend, "api_calls")
    cache_hits_by_modality_before = _backend_modality_counters(
        backend, "cache_hits_by_modality"
    )
    api_calls_by_modality_before = _backend_modality_counters(
        backend, "api_calls_by_modality"
    )
    canary_vector = _record_canary(backend)

    image_vectors, image_batches = _embed_in_batches(
        backend.embed_images, image_paths, IMAGE_BATCH_SIZE, "image"
    )
    text_vectors, text_batches = _embed_in_batches(
        backend.embed_texts,
        [_product_text(product) for product in products],
        TEXT_BATCH_SIZE,
        "text",
    )
    _verify_build_snapshots(
        products_path=products_path,
        products_bytes=source_bytes,
        eligibility_manifest_path=eligibility_manifest_path,
        query_artifact_path=query_artifact_path,
        gallery_artifact_path=gallery_artifact_path,
        verified=verified_eligibility,
    )

    staging = prepare_staging_directory(output_dir)
    try:
        pq.write_table(table, staging / "products.parquet", compression="zstd")
        np.save(staging / "image_vectors.npy", image_vectors, allow_pickle=False)
        np.save(staging / "text_vectors.npy", text_vectors, allow_pickle=False)
        _write_faiss_index(staging / "image.faiss", image_vectors)
        _write_faiss_index(staging / "text.faiss", text_vectors)

        if verified_eligibility is not None:
            (staging / _ELIGIBILITY_ARTIFACT).write_bytes(
                verified_eligibility.manifest_bytes
            )
            (staging / _GALLERY_ARTIFACT).write_bytes(
                verified_eligibility.gallery_bytes
            )

        manifest = _build_manifest(
            staging=staging,
            source_rows=source_rows,
            product_count=len(products),
            products=products,
            complete=limit is None,
            model=_backend_model_name(backend),
            embedding_execution_location=execution_location,
            embedding_runtime_assurance=embedding_runtime_assurance,
            embedding_runtime_binding_sha256=embedding_runtime_binding_sha256,
            canary_vector=canary_vector,
            source_sha256=source_sha256,
            verified_eligibility=verified_eligibility,
            indexed_gallery=indexed_gallery,
        )
        _write_manifest(staging / "manifest.json", manifest)

        # Reload every persisted artefact before it can replace a known-good index.
        ProductIndex.load(
            staging,
            allow_incomplete=True,
            allow_provisional_gallery=verified_eligibility is None,
        )
        publish_staged_directory(staging, output_dir)
    except Exception:
        discard_staging_directory(staging)
        raise

    elapsed_seconds = time.perf_counter() - started
    total_vectors = len(products) * 2
    cache_hits_by_modality = _modality_counter_delta(
        cache_hits_by_modality_before,
        _backend_modality_counters(backend, "cache_hits_by_modality"),
    )
    api_calls_by_modality = _modality_counter_delta(
        api_calls_by_modality_before,
        _backend_modality_counters(backend, "api_calls_by_modality"),
    )
    cache_hits = _counter_delta(
        cache_hits_before, _backend_counter(backend, "cache_hits")
    )
    api_calls = _counter_delta(api_calls_before, _backend_counter(backend, "api_calls"))
    return BuildProductIndexReport(
        products=len(products),
        image_vectors=len(products),
        text_vectors=len(products),
        batches={"image": image_batches, "text": text_batches},
        cache_hits=sum(cache_hits_by_modality.values())
        if cache_hits_by_modality
        else cache_hits,
        api_calls=sum(api_calls_by_modality.values())
        if api_calls_by_modality
        else api_calls,
        cache_hits_by_modality=cache_hits_by_modality,
        api_calls_by_modality=api_calls_by_modality,
        items_per_second=(total_vectors / elapsed_seconds if elapsed_seconds else 0.0),
        elapsed_seconds=elapsed_seconds,
        complete=limit is None,
    )


def _resolve_build_eligibility(
    *,
    eligibility_manifest_path: Path | str | None,
    query_artifact_path: Path | str | None,
    gallery_artifact_path: Path | str | None,
    asset_catalog: AssetCatalog | None,
    limit: int | None,
    allow_provisional_gallery: bool,
) -> VerifiedGalleryEligibility | None:
    inputs = (
        eligibility_manifest_path,
        query_artifact_path,
        gallery_artifact_path,
        asset_catalog,
    )
    provided = tuple(value is not None for value in inputs)
    if any(provided) and not all(provided):
        raise ValueError(
            "verified product index requires eligibility manifest, actual queries, "
            "gallery artifact, and verified asset catalog together"
        )
    if allow_provisional_gallery:
        if any(provided):
            raise ValueError(
                "allow_provisional_gallery cannot be combined with verified gallery inputs"
            )
        if limit is None:
            raise ValueError(
                "provisional gallery is allowed only for a limited smoke index"
            )
        return None
    if not all(provided):
        raise ValueError(
            "formal product index requires a verified query/gallery eligibility bundle"
        )

    assert eligibility_manifest_path is not None
    assert query_artifact_path is not None
    assert gallery_artifact_path is not None
    assert asset_catalog is not None
    asset_catalog.require_verified_files()
    return verify_gallery_eligibility(
        eligibility_manifest_path,
        query_artifact_path,
        gallery_artifact_path,
        asset_catalog,
    )


def _validate_source_gallery_binding(
    products: tuple[Product, ...],
    verified: VerifiedGalleryEligibility | None,
) -> tuple[GalleryAssetReference, ...] | None:
    if verified is None:
        return None
    references = verified.gallery_references
    assets = verified.gallery_assets
    if len(products) != len(references):
        raise ValueError(
            "products.parquet/gallery: full source and verified gallery row counts differ"
        )
    if len(assets) != len(references):
        raise ValueError("verified gallery asset/reference counts differ")

    reference_by_path: dict[str, GalleryAssetReference] = {}
    asset_by_id = {asset.asset_id: asset for asset in assets}
    for reference in references:
        asset = asset_by_id.get(reference.asset_id)
        if asset is None:
            raise ValueError(
                f"verified gallery asset is missing from catalog: {reference.asset_id}"
            )
        if reference.image_path != asset.local_path:
            raise ValueError(
                "gallery image_path must use the catalog's canonical local_path: "
                f"{reference.asset_id}"
            )
        if reference.image_path in reference_by_path:
            raise ValueError(
                f"verified gallery contains duplicate image_path: {reference.image_path}"
            )
        reference_by_path[reference.image_path] = reference

    aligned: list[GalleryAssetReference] = []
    for product in products:
        reference = reference_by_path.get(product.image_path)
        if reference is None:
            raise ValueError(
                "products.parquet/gallery: product image is absent from verified gallery: "
                f"{product.product_id} ({product.image_path})"
            )
        asset = asset_by_id[reference.asset_id]
        if product.source != asset.source_dataset:
            raise ValueError(
                "products.parquet/gallery: product source does not match catalog asset: "
                f"{product.product_id}"
            )
        if asset.product_id is None:
            raise ValueError(
                "products.parquet/gallery: formal gallery asset lacks product_id: "
                f"{asset.asset_id}"
            )
        if product.product_id != asset.product_id:
            raise ValueError(
                "products.parquet/gallery: product_id does not match catalog asset: "
                f"{product.product_id} != {asset.product_id}"
            )
        aligned.append(reference)

    if {reference.asset_id for reference in aligned} != {
        reference.asset_id for reference in references
    }:
        raise ValueError(
            "products.parquet/gallery: source and verified gallery are not a bijection"
        )
    return tuple(aligned)


def _enforce_gallery_embedding_permissions(
    execution_location: str,
    indexed_gallery: tuple[GalleryAssetReference, ...] | None,
    verified: VerifiedGalleryEligibility | None,
) -> None:
    """Reject remote image upload unless every exact gallery asset permits it."""

    if execution_location == "local":
        return
    if execution_location != "remote":
        raise ValueError("unsupported embedding execution location")
    if verified is None or indexed_gallery is None:
        raise ValueError(
            "remote image embedding requires a verified gallery and asset permissions"
        )
    assets = {asset.asset_id: asset for asset in verified.gallery_assets}
    denied = sorted(
        reference.asset_id
        for reference in indexed_gallery
        if assets[reference.asset_id].cloud_upload_allowed is not True
    )
    if denied:
        raise ValueError(
            "remote image embedding is not licensed for gallery assets: "
            + ", ".join(denied[:5])
        )


def _verify_build_snapshots(
    *,
    products_path: Path,
    products_bytes: bytes,
    eligibility_manifest_path: Path | str | None,
    query_artifact_path: Path | str | None,
    gallery_artifact_path: Path | str | None,
    verified: VerifiedGalleryEligibility | None,
) -> None:
    try:
        current_products = products_path.read_bytes()
    except OSError as error:
        raise ValueError(
            "products.parquet: became unreadable during index build"
        ) from error
    if current_products != products_bytes:
        raise ValueError("products.parquet: changed during index build")
    if verified is None:
        return
    assert eligibility_manifest_path is not None
    assert query_artifact_path is not None
    assert gallery_artifact_path is not None
    snapshots = (
        (
            Path(eligibility_manifest_path),
            verified.manifest_bytes,
            "eligibility manifest",
        ),
        (Path(query_artifact_path), verified.query_bytes, "query artifact"),
        (Path(gallery_artifact_path), verified.gallery_bytes, "gallery artifact"),
    )
    for path, expected, label in snapshots:
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise ValueError(f"{label} became unreadable during index build") from error
        if actual != expected:
            raise ValueError(f"{label} changed during index build")
    verified.catalog.verify_asset_ids(
        [query.asset_id for query in verified.queries]
        + [reference.asset_id for reference in verified.gallery_references]
    )


def _read_products_snapshot(products_path: Path) -> bytes:
    if products_path.is_symlink():
        raise ValueError("products.parquet: symlink sources are not allowed")
    try:
        metadata = products_path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("products.parquet: source must be a regular file")
        return products_path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(f"products.parquet: unable to read {products_path}") from error
    except OSError as error:
        raise ValueError(f"products.parquet: unable to read {products_path}") from error


def _read_products_table(content: bytes, products_path: Path):
    try:
        return pq.read_table(io.BytesIO(content))
    except Exception as error:
        raise ValueError(f"products.parquet: unable to read {products_path}") from error


def _validate_products(table: Any, products_path: Path) -> tuple[Product, ...]:
    try:
        products = tuple(Product.model_validate(row) for row in table.to_pylist())
    except Exception as error:
        raise ValueError(
            f"products.parquet: invalid Product record in {products_path}"
        ) from error
    if not products:
        raise ValueError("products.parquet: contains no products")
    product_ids = [product.product_id for product in products]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("products.parquet: duplicate product_id")
    return products


def _image_paths(products: Sequence[Product], products_path: Path) -> list[Path]:
    resolved: list[Path] = []
    for product in products:
        image_path = Path(product.image_path)
        if not image_path.is_absolute():
            image_path = products_path.parent / image_path
        if not image_path.is_file():
            raise ValueError(
                f"products.parquet: missing image for product {product.product_id}: {image_path}"
            )
        resolved.append(image_path)
    return resolved


def _product_text(product: Product) -> str:
    """Use the mandatory title as the stable product text representation."""
    return product.title


def _record_canary(backend: EmbeddingBackend) -> np.ndarray:
    """Embed the reproducible canary once and retain its audited baseline."""
    establish_canary = getattr(backend, "establish_canary", None)
    if callable(establish_canary) and getattr(backend, "canary_vector", None) is None:
        response = establish_canary(CANARY_TEXT)
    else:
        response = backend.embed_texts([CANARY_TEXT])
    return _validate_backend_vectors(response, 1, "canary")[0]


def _embed_in_batches(
    embed: Any, values: Sequence[Any], batch_size: int, modality: str
) -> tuple[np.ndarray, int]:
    vectors: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        response = embed(batch)
        vectors.append(_validate_backend_vectors(response, len(batch), modality))
    if not vectors:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), 0
    return np.vstack(vectors).astype(np.float32, copy=False), len(vectors)


def _validate_backend_vectors(
    response: Any, expected_rows: int, modality: str
) -> np.ndarray:
    try:
        vectors = np.asarray(response)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{modality} backend response is not a matrix") from error
    if vectors.dtype != np.float32:
        raise ValueError(f"{modality} backend response must use float32 vectors")
    if vectors.shape != (expected_rows, EMBEDDING_DIMENSION):
        raise ValueError(
            f"{modality} backend response must have shape "
            f"({expected_rows}, {EMBEDDING_DIMENSION})"
        )
    _validate_vector_matrix(vectors, f"{modality} backend response")
    return vectors


def _validate_vector_matrix(vectors: np.ndarray, filename: str) -> None:
    if vectors.dtype != np.float32:
        raise ValueError(f"{filename}: vectors must be float32")
    if vectors.ndim != 2 or vectors.shape[1:] != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"{filename}: vectors must have dimension {EMBEDDING_DIMENSION}"
        )
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{filename}: vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{filename}: vectors must be unit normalized")


def _write_faiss_index(path: Path, vectors: np.ndarray) -> None:
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    if len(vectors):
        index.add(vectors)
    faiss.write_index(index, str(path))


def _build_manifest(
    *,
    staging: Path,
    source_rows: int,
    product_count: int,
    products: tuple[Product, ...],
    complete: bool,
    model: str,
    embedding_execution_location: str,
    embedding_runtime_assurance: str,
    embedding_runtime_binding_sha256: str | None,
    canary_vector: np.ndarray,
    source_sha256: str,
    verified_eligibility: VerifiedGalleryEligibility | None,
    indexed_gallery: tuple[GalleryAssetReference, ...] | None,
) -> dict[str, Any]:
    eligibility = _index_eligibility_manifest(
        verified_eligibility,
        products,
        indexed_gallery,
    )
    artifact_names = (
        _VERIFIED_ARTIFACT_NAMES
        if verified_eligibility is not None
        else _CORE_ARTIFACT_NAMES
    )
    manifest = {
        "format_version": _FORMAT_VERSION,
        "model": model,
        "embedding_execution_location": embedding_execution_location,
        "embedding_runtime_assurance": embedding_runtime_assurance,
        "embedding_runtime_binding_sha256": embedding_runtime_binding_sha256,
        "dimension": EMBEDDING_DIMENSION,
        "normalization": "l2",
        "complete": complete,
        "source": {
            "parquet_sha256": source_sha256,
            "row_count": source_rows,
        },
        "query_gallery_eligibility": eligibility,
        "product_count": product_count,
        "vector_counts": {"image": product_count, "text": product_count},
        "faiss": {
            "type": "IndexFlatIP",
            "dimension": EMBEDDING_DIMENSION,
            "image_ntotal": product_count,
            "text_ntotal": product_count,
        },
        "batch_parameters": {
            "image_batch_size": IMAGE_BATCH_SIZE,
            "text_batch_size": TEXT_BATCH_SIZE,
        },
        "canary": _canary_manifest(canary_vector),
        "artifacts": {
            filename: _artifact_descriptor(staging / filename)
            for filename in artifact_names
        },
    }
    manifest["integrity_sha256"] = _manifest_digest(manifest)
    return manifest


def _index_eligibility_manifest(
    verified: VerifiedGalleryEligibility | None,
    products: tuple[Product, ...],
    indexed_gallery: tuple[GalleryAssetReference, ...] | None,
) -> dict[str, Any]:
    if verified is None:
        if indexed_gallery is not None:
            raise ValueError("provisional index cannot contain verified gallery rows")
        return {
            "mode": "provisional",
            "reason": "explicit-limited-smoke",
        }
    if indexed_gallery is None:
        raise ValueError("verified index requires row-aligned gallery references")
    eligibility = verified.manifest
    return {
        "mode": "verified",
        "eligibility_schema_version": eligibility.schema_version,
        "eligibility_policy_version": eligibility.eligibility_policy_version,
        "eligibility_sha256": eligibility.eligibility_sha256,
        "asset_catalog_sha256": eligibility.catalog_sha256,
        "catalog_policy_version": eligibility.catalog_policy_version,
        "leakage_policy_version": eligibility.leakage_policy_version,
        "query_artifact_sha256": eligibility.query_artifact_sha256,
        "gallery_artifact_sha256": eligibility.gallery_artifact_sha256,
        "gallery_count": len(verified.gallery_references),
        "indexed_row_binding_sha256": _row_binding_sha256(products, indexed_gallery),
    }


def _canary_manifest(vector: np.ndarray) -> dict[str, Any]:
    _validate_vector_matrix(vector.reshape(1, -1), "canary backend response")
    return {
        "text_sha256": _CANARY_TEXT_SHA256,
        "dtype": "float32",
        "dimension": EMBEDDING_DIMENSION,
        "normalization": "l2",
        "vector_sha256": _vector_sha256(vector),
        "vector": vector.tolist(),
    }


def _backend_model_name(backend: Any) -> str:
    # The public backend protocol does not require a model property.  The pinned
    # setting is therefore the reproducibility default and is replaced by a
    # backend declaration only when one is explicitly available to the builder.
    model = getattr(backend, "model", EMBEDDING_MODEL)
    return model if isinstance(model, str) and model else EMBEDDING_MODEL


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise ValueError("manifest.json: symlink is not allowed")
        content = path.read_bytes()
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except FileNotFoundError as error:
        raise ValueError("manifest.json: missing") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest.json: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("manifest.json: root must be an object")
    if content != (_canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError("manifest.json: must be canonical JSON")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"manifest.json: duplicate key {key}")
        value[key] = item
    return value


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    allow_provisional_gallery: bool,
) -> None:
    required = {
        "format_version",
        "model",
        "embedding_execution_location",
        "embedding_runtime_assurance",
        "embedding_runtime_binding_sha256",
        "dimension",
        "normalization",
        "complete",
        "source",
        "query_gallery_eligibility",
        "product_count",
        "vector_counts",
        "faiss",
        "batch_parameters",
        "canary",
        "artifacts",
        "integrity_sha256",
    }
    if set(manifest) != required:
        raise ValueError("manifest.json: root field set is invalid")
    integrity_sha256 = manifest["integrity_sha256"]
    if not _is_sha256(integrity_sha256) or integrity_sha256 != _manifest_digest(
        manifest
    ):
        raise ValueError("manifest.json: integrity sha256 mismatch")
    if manifest["format_version"] != _FORMAT_VERSION:
        raise ValueError("manifest.json: unsupported format version")
    if not isinstance(manifest["model"], str) or not manifest["model"]:
        raise ValueError("manifest.json: model must be a non-empty string")
    if manifest["embedding_execution_location"] not in {"local", "remote"}:
        raise ValueError("manifest.json: embedding execution location is invalid")
    assurance = manifest["embedding_runtime_assurance"]
    runtime_binding = manifest["embedding_runtime_binding_sha256"]
    if assurance == "formal-attested":
        if not _is_sha256(runtime_binding):
            raise ValueError("manifest.json: embedding runtime binding is invalid")
    elif assurance == "diagnostic-unattested":
        if runtime_binding is not None:
            raise ValueError(
                "manifest.json: diagnostic embedding runtime must not self-attest"
            )
    else:
        raise ValueError("manifest.json: embedding runtime assurance is invalid")
    if manifest["dimension"] != EMBEDDING_DIMENSION:
        raise ValueError("manifest.json: unexpected dimension")
    if manifest["normalization"] != "l2":
        raise ValueError("manifest.json: unexpected normalization")
    if not isinstance(manifest["complete"], bool):
        raise ValueError("manifest.json: complete must be boolean")
    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != {"parquet_sha256", "row_count"}:
        raise ValueError("manifest.json: source metadata is invalid")
    source_rows = _require_count(source, "row_count")
    if not _is_sha256(source.get("parquet_sha256")):
        raise ValueError("manifest.json: source parquet hash is invalid")
    product_count = _require_count(manifest, "product_count")
    if product_count > source_rows:
        raise ValueError("manifest.json: product count exceeds source rows")
    if manifest["complete"] and product_count != source_rows:
        raise ValueError("manifest.json: complete index must cover all source rows")
    gate = _validate_index_eligibility(
        manifest["query_gallery_eligibility"],
        complete=manifest["complete"],
        source_rows=source_rows,
        allow_provisional_gallery=allow_provisional_gallery,
    )
    if manifest["vector_counts"] != {"image": product_count, "text": product_count}:
        raise ValueError("manifest.json: vector counts are invalid")
    expected_faiss = {
        "type": "IndexFlatIP",
        "dimension": EMBEDDING_DIMENSION,
        "image_ntotal": product_count,
        "text_ntotal": product_count,
    }
    if manifest["faiss"] != expected_faiss:
        raise ValueError("manifest.json: FAISS metadata is invalid")
    expected_batches = {
        "image_batch_size": IMAGE_BATCH_SIZE,
        "text_batch_size": TEXT_BATCH_SIZE,
    }
    if manifest["batch_parameters"] != expected_batches:
        raise ValueError("manifest.json: batch parameters are invalid")
    _canary_vector_from_manifest(manifest["canary"])
    artifacts = manifest["artifacts"]
    artifact_names = (
        _VERIFIED_ARTIFACT_NAMES if gate["mode"] == "verified" else _CORE_ARTIFACT_NAMES
    )
    if not isinstance(artifacts, dict) or set(artifacts) != set(artifact_names):
        raise ValueError("manifest.json: artifact set is invalid")
    for filename in artifact_names:
        descriptor = artifacts[filename]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError(
                f"manifest.json: artifact metadata for {filename} is invalid"
            )
        if descriptor.get("path") != filename:
            raise ValueError(f"manifest.json: artifact path for {filename} is invalid")
        if not isinstance(descriptor.get("bytes"), int) or descriptor["bytes"] < 0:
            raise ValueError(
                f"manifest.json: artifact bytes for {filename} are invalid"
            )
        if not _is_sha256(descriptor.get("sha256")):
            raise ValueError(f"manifest.json: artifact hash for {filename} is invalid")


def _validate_index_eligibility(
    value: Any,
    *,
    complete: bool,
    source_rows: int,
    allow_provisional_gallery: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("manifest.json: query/gallery eligibility is invalid")
    mode = value.get("mode")
    if mode == "provisional":
        if set(value) != {"mode", "reason"} or value.get("reason") != (
            "explicit-limited-smoke"
        ):
            raise ValueError("manifest.json: provisional gallery metadata is invalid")
        if complete:
            raise ValueError(
                "manifest.json: complete index cannot use provisional gallery"
            )
        if not allow_provisional_gallery:
            raise ValueError(
                "manifest.json: provisional gallery requires explicit opt-in"
            )
        return value
    if mode != "verified":
        raise ValueError("manifest.json: query/gallery eligibility mode is invalid")
    required = {
        "mode",
        "eligibility_schema_version",
        "eligibility_policy_version",
        "eligibility_sha256",
        "asset_catalog_sha256",
        "catalog_policy_version",
        "leakage_policy_version",
        "query_artifact_sha256",
        "gallery_artifact_sha256",
        "gallery_count",
        "indexed_row_binding_sha256",
    }
    if set(value) != required:
        raise ValueError("manifest.json: verified gallery field set is invalid")
    if value["eligibility_schema_version"] != 1:
        raise ValueError("manifest.json: eligibility schema version is invalid")
    if value["eligibility_policy_version"] != ELIGIBILITY_POLICY_VERSION:
        raise ValueError("manifest.json: eligibility policy version is invalid")
    for field in (
        "eligibility_sha256",
        "asset_catalog_sha256",
        "query_artifact_sha256",
        "gallery_artifact_sha256",
        "indexed_row_binding_sha256",
    ):
        if not _is_sha256(value[field]):
            raise ValueError(f"manifest.json: {field} is invalid")
    for field in ("catalog_policy_version", "leakage_policy_version"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"manifest.json: {field} is invalid")
    gallery_count = _require_count(value, "gallery_count")
    if gallery_count != source_rows:
        raise ValueError("manifest.json: verified gallery must cover the full source")
    return value


def _artifact_names_for_manifest(manifest: dict[str, Any]) -> tuple[str, ...]:
    return (
        _VERIFIED_ARTIFACT_NAMES
        if manifest["query_gallery_eligibility"]["mode"] == "verified"
        else _CORE_ARTIFACT_NAMES
    )


def _require_count(mapping: Any, key: str) -> int:
    if (
        not isinstance(mapping, dict)
        or isinstance(mapping.get(key), bool)
        or not isinstance(mapping.get(key), int)
    ):
        raise ValueError(f"manifest.json: {key} must be an integer")
    if mapping[key] < 0:
        raise ValueError(f"manifest.json: {key} must not be negative")
    return mapping[key]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canary_vector_from_manifest(canary: Any) -> np.ndarray:
    if not isinstance(canary, dict):
        raise ValueError("manifest.json: canary metadata is invalid")
    if canary.get("text_sha256") != _CANARY_TEXT_SHA256:
        raise ValueError("manifest.json: canary text hash is invalid")
    if canary.get("dtype") != "float32":
        raise ValueError("manifest.json: canary dtype is invalid")
    if canary.get("dimension") != EMBEDDING_DIMENSION:
        raise ValueError("manifest.json: canary dimension is invalid")
    if canary.get("normalization") != "l2":
        raise ValueError("manifest.json: canary normalization is invalid")
    if not isinstance(canary.get("vector"), list):
        raise ValueError("manifest.json: canary vector is invalid")
    try:
        vector = np.asarray(canary["vector"], dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("manifest.json: canary vector is invalid") from error
    if vector.shape != (EMBEDDING_DIMENSION,):
        raise ValueError("manifest.json: canary vector dimension is invalid")
    _validate_vector_matrix(vector.reshape(1, -1), "manifest.json: canary vector")
    if canary.get("vector_sha256") != _vector_sha256(vector):
        raise ValueError("manifest.json: canary vector hash is invalid")
    return vector


def _verify_artifact(
    output_dir: Path, filename: str, descriptor: dict[str, Any]
) -> None:
    path = output_dir / filename
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{filename}: missing") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{filename}: must be a regular non-symlink file")
    if metadata.st_size != descriptor["bytes"]:
        raise ValueError(f"{filename}: byte size mismatch")
    if _sha256(path) != descriptor["sha256"]:
        raise ValueError(f"{filename}: sha256 mismatch")


def _load_index_gallery_binding(
    output_dir: Path,
    products: tuple[Product, ...],
    manifest: dict[str, Any],
) -> tuple[GalleryAssetReference, ...] | None:
    gate = manifest["query_gallery_eligibility"]
    if gate["mode"] == "provisional":
        return None

    try:
        eligibility = load_gallery_eligibility_manifest(
            output_dir / _ELIGIBILITY_ARTIFACT
        )
    except ValueError as error:
        raise ValueError(f"{_ELIGIBILITY_ARTIFACT}: invalid manifest") from error

    expected_gate = {
        "eligibility_schema_version": eligibility.schema_version,
        "eligibility_policy_version": eligibility.eligibility_policy_version,
        "eligibility_sha256": eligibility.eligibility_sha256,
        "asset_catalog_sha256": eligibility.catalog_sha256,
        "catalog_policy_version": eligibility.catalog_policy_version,
        "leakage_policy_version": eligibility.leakage_policy_version,
        "query_artifact_sha256": eligibility.query_artifact_sha256,
        "gallery_artifact_sha256": eligibility.gallery_artifact_sha256,
    }
    for field, expected in expected_gate.items():
        if gate[field] != expected:
            raise ValueError(
                f"manifest.json: {field} does not match bundled eligibility"
            )

    gallery_bytes = (output_dir / _GALLERY_ARTIFACT).read_bytes()
    if sha256_bytes(gallery_bytes) != eligibility.gallery_artifact_sha256:
        raise ValueError(f"{_GALLERY_ARTIFACT}: eligibility hash mismatch")
    references = _parse_bundled_gallery(gallery_bytes)
    if len(references) != gate["gallery_count"]:
        raise ValueError(f"{_GALLERY_ARTIFACT}: row count mismatch")
    if eligibility.report.gallery_count != len(references):
        raise ValueError(f"{_ELIGIBILITY_ARTIFACT}: gallery count mismatch")
    if eligibility.report.query_count <= 0:
        raise ValueError(f"{_ELIGIBILITY_ARTIFACT}: query count must be positive")

    reference_by_path = {reference.image_path: reference for reference in references}
    if len(reference_by_path) != len(references):
        raise ValueError(f"{_GALLERY_ARTIFACT}: duplicate image_path")
    aligned: list[GalleryAssetReference] = []
    for product in products:
        reference = reference_by_path.get(product.image_path)
        if reference is None:
            raise ValueError(
                f"{_GALLERY_ARTIFACT}: missing indexed product {product.product_id}"
            )
        aligned.append(reference)
    if manifest["complete"] and len(aligned) != len(references):
        raise ValueError(f"{_GALLERY_ARTIFACT}: complete index is not a bijection")
    aligned_references = tuple(aligned)
    if gate["indexed_row_binding_sha256"] != _row_binding_sha256(
        products, aligned_references
    ):
        raise ValueError("manifest.json: indexed gallery row binding mismatch")
    return aligned_references


def _parse_bundled_gallery(content: bytes) -> tuple[GalleryAssetReference, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{_GALLERY_ARTIFACT}: must be UTF-8") from error
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{_GALLERY_ARTIFACT}: must contain at least one row")
    references: list[GalleryAssetReference] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
            references.append(GalleryAssetReference.model_validate(row))
        except Exception as error:
            raise ValueError(
                f"{_GALLERY_ARTIFACT}: invalid row {line_number}"
            ) from error
    result = tuple(references)
    if content != canonical_jsonl_bytes(result):
        raise ValueError(f"{_GALLERY_ARTIFACT}: must be canonical JSONL")
    asset_ids = [reference.asset_id for reference in result]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError(f"{_GALLERY_ARTIFACT}: duplicate asset_id")
    return result


def _row_binding_sha256(
    products: Sequence[Product],
    references: Sequence[GalleryAssetReference],
) -> str:
    if len(products) != len(references):
        raise ValueError("product/gallery row binding counts differ")
    rows = (
        {
            "row_index": index,
            "product_id": product.product_id,
            "source": product.source,
            "asset_id": reference.asset_id,
            "image_path": reference.image_path,
        }
        for index, (product, reference) in enumerate(
            zip(products, references, strict=True)
        )
    )
    return sha256_bytes(canonical_jsonl_bytes(rows))


def _load_products(path: Path, expected_rows: int) -> tuple[Product, ...]:
    try:
        table = pq.read_table(path)
    except Exception as error:
        raise ValueError("products.parquet: unreadable") from error
    products = _validate_products(table, path)
    if len(products) != expected_rows:
        raise ValueError("products.parquet: row count mismatch")
    return products


def _load_vectors(path: Path, expected_rows: int) -> np.ndarray:
    try:
        vectors = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"{path.name}: unreadable") from error
    if vectors.shape != (expected_rows, EMBEDDING_DIMENSION):
        raise ValueError(f"{path.name}: vector shape mismatch")
    _validate_vector_matrix(vectors, path.name)
    return vectors


def _load_faiss_index(path: Path, vectors: np.ndarray, expected_rows: int):
    try:
        index = faiss.read_index(str(path))
    except Exception as error:
        raise ValueError(f"{path.name}: unreadable") from error
    if type(index).__name__ != "IndexFlatIP":
        raise ValueError(f"{path.name}: expected IndexFlatIP")
    if index.d != EMBEDDING_DIMENSION or index.ntotal != expected_rows:
        raise ValueError(f"{path.name}: dimension or total mismatch")
    if expected_rows:
        stored = np.asarray(index.reconstruct_n(0, expected_rows), dtype=np.float32)
        if stored.shape != vectors.shape or not np.array_equal(stored, vectors):
            raise ValueError(f"{path.name}: vector row order mismatch")
    return index


def _manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("integrity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _faiss_sha256(index: Any) -> str:
    try:
        content = np.asarray(faiss.serialize_index(index), dtype=np.uint8).tobytes()
    except Exception as error:
        raise ValueError("FAISS index cannot be serialized for verification") from error
    return sha256_bytes(content)


def _vector_sha256(vector: np.ndarray) -> str:
    return hashlib.sha256(vector.astype(np.float32, copy=False).tobytes()).hexdigest()


def _backend_counter(backend: Any, attribute: str) -> int | None:
    value = getattr(backend, attribute, None)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _counter_delta(before: int | None, after: int | None) -> int:
    if before is None or after is None or after < before:
        return 0
    return after - before


def _backend_modality_counters(backend: Any, attribute: str) -> dict[str, int] | None:
    value = getattr(backend, attribute, None)
    if not isinstance(value, dict) or set(value) != {"image", "text"}:
        return None
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in value.values()
    ):
        return None
    return {"image": value["image"], "text": value["text"]}


def _modality_counter_delta(
    before: dict[str, int] | None, after: dict[str, int] | None
) -> dict[str, int]:
    if before is None or after is None:
        return {"image": 0, "text": 0}
    return {
        modality: max(0, after[modality] - before[modality])
        for modality in ("image", "text")
    }


__all__ = [
    "CANARY_TEXT",
    "BuildProductIndexReport",
    "ProductIndex",
    "ProductIndexCanary",
    "build_product_index",
]
