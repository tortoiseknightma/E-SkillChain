"""Build or inspect the auditable product index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

from skillchain.data.asset_catalog import load_asset_catalog
from skillchain.data.kb_catalog import build_kb_catalog, load_kb_catalog
from skillchain.tools.embedding import (
    CachedEmbeddingBackend,
    DashScopeEmbeddingClient,
    EmbeddingBackend,
    EmbeddingCache,
    FormalEmbeddingBackend,
    OpenCLIPEmbeddingBackend,
)
from skillchain.tools.product_index import (
    CANARY_TEXT,
    ProductIndex,
    build_product_index,
)
from skillchain.tools.kb_index import build_kb_bundle, load_kb_bundle
from skillchain.tools.model_artifacts import (
    load_model_artifact_manifest,
    publish_model_artifact_manifest,
)
from skillchain.tools.registry import (
    build_mvp_registry_spec,
    load_registry_manifest,
    publish_registry_manifest,
)
from skillchain.tools.settings import (
    EMBEDDING_CACHE_PATH,
    KB_CATALOG_DIR,
    KB_INDEX_DIR,
    KB_SOURCE_DIR,
    PRODUCT_INDEX_DIR,
    PRODUCTS_SOURCE_PATH,
    TOOL_REGISTRY_MANIFEST_PATH,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    products = commands.add_parser(
        "products", help="build the product image and text indices"
    )
    products.add_argument("--source", type=Path, default=PRODUCTS_SOURCE_PATH)
    products.add_argument("--output", type=Path)
    products.add_argument("--limit", type=_positive_limit)
    products.add_argument("--eligibility-manifest", type=Path)
    products.add_argument("--queries", type=Path)
    products.add_argument("--gallery-assets", type=Path)
    products.add_argument("--catalog", type=Path)
    products.add_argument("--asset-root", type=Path)
    products.add_argument("--allow-provisional-gallery", action="store_true")
    products.add_argument(
        "--embedding-provider",
        choices=("dashscope", "open_clip"),
        default="dashscope",
    )
    products.add_argument("--embedding-manifest", type=Path)
    products.add_argument("--embedding-manifest-sha256")

    kb_catalog = commands.add_parser(
        "kb-catalog", help="build a strict two-source KB catalog"
    )
    kb_catalog.add_argument(
        "--encyclopedia",
        type=Path,
        default=KB_SOURCE_DIR / "encyclopedia.jsonl",
    )
    kb_catalog.add_argument(
        "--recipes",
        type=Path,
        default=KB_SOURCE_DIR / "recipes.jsonl",
    )
    kb_catalog.add_argument(
        "--encyclopedia-sha256",
        help="externally pinned SHA-256 of the complete encyclopedia JSONL",
    )
    kb_catalog.add_argument(
        "--recipes-sha256",
        help="externally pinned SHA-256 of the complete recipes JSONL",
    )
    kb_catalog.add_argument("--output", type=Path, default=KB_CATALOG_DIR)
    kb_catalog.add_argument("--limit-per-kind", type=_positive_limit)
    kb_catalog.add_argument("--allow-provisional", action="store_true")

    kb = commands.add_parser("kb", help="build the atomic encyclopedia/recipe bundle")
    kb.add_argument("--catalog", type=Path, default=KB_CATALOG_DIR)
    kb.add_argument("--catalog-sha256", help="externally pinned catalog SHA-256")
    kb.add_argument("--output", type=Path, default=KB_INDEX_DIR)
    kb.add_argument("--limit-per-kind", type=_positive_limit)
    kb.add_argument("--allow-provisional", action="store_true")

    kb_status = commands.add_parser("kb-status", help="verify an existing KB bundle")
    kb_status.add_argument("--catalog", type=Path, default=KB_CATALOG_DIR)
    kb_status.add_argument("--catalog-sha256", help="externally pinned catalog SHA-256")
    kb_status.add_argument("--output", type=Path, default=KB_INDEX_DIR)
    kb_status.add_argument("--bundle-sha256", help="externally pinned bundle SHA-256")
    kb_status.add_argument("--allow-provisional", action="store_true")

    model_manifest = commands.add_parser(
        "model-manifest", help="publish an embedding, detector, or OCR model lock"
    )
    model_manifest.add_argument(
        "--kind",
        choices=("multimodal_embedding", "object_detector", "document_ocr"),
        required=True,
    )
    model_manifest.add_argument("--output", type=Path, required=True)
    model_manifest.add_argument("--model-id", required=True)
    model_manifest.add_argument("--backend-name", required=True)
    model_manifest.add_argument("--backend-version", required=True)
    model_manifest.add_argument(
        "--artifact",
        action="append",
        type=_artifact_pair,
        required=True,
        metavar="ROLE=PATH",
    )

    model_status = commands.add_parser(
        "model-status", help="verify a model lock and all referenced files"
    )
    model_status.add_argument("--manifest", type=Path, required=True)
    model_status.add_argument("--manifest-sha256", required=True)
    model_status.add_argument(
        "--kind",
        choices=("multimodal_embedding", "object_detector", "document_ocr"),
    )

    registry = commands.add_parser(
        "registry", help="publish the canonical seven-tool ToolSpec registry"
    )
    registry.add_argument("--output", type=Path, default=TOOL_REGISTRY_MANIFEST_PATH)

    registry_status = commands.add_parser(
        "registry-status", help="verify a canonical ToolSpec registry"
    )
    registry_status.add_argument(
        "--output", type=Path, default=TOOL_REGISTRY_MANIFEST_PATH
    )

    status = commands.add_parser(
        "status", help="validate and report an existing product index"
    )
    status.add_argument("--output", type=Path, default=PRODUCT_INDEX_DIR)
    status.add_argument("--allow-incomplete", action="store_true")
    status.add_argument("--allow-provisional-gallery", action="store_true")
    return parser.parse_args(argv)


def _positive_limit(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be greater than zero")
    return parsed


def _artifact_pair(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or not role or not raw_path:
        raise argparse.ArgumentTypeError("artifact must use ROLE=PATH")
    return role, Path(raw_path)


def _default_backend(output_dir: Path) -> EmbeddingBackend:
    canary_arguments = {}
    try:
        existing_canary = ProductIndex.load_canary(output_dir)
    except ValueError:
        pass
    else:
        canary_arguments = {
            "canary_text": existing_canary.text,
            "canary_vector": existing_canary.vector,
        }
    return CachedEmbeddingBackend(
        DashScopeEmbeddingClient(),
        EmbeddingCache(EMBEDDING_CACHE_PATH),
        **canary_arguments,
    )


def _formal_backend(arguments: argparse.Namespace) -> EmbeddingBackend:
    if arguments.embedding_provider == "dashscope":
        if (
            arguments.embedding_manifest is not None
            or arguments.embedding_manifest_sha256 is not None
        ):
            raise ValueError(
                "DashScope embedding cannot use a local embedding manifest"
            )
        upstream: EmbeddingBackend = DashScopeEmbeddingClient()
    else:
        if (
            arguments.embedding_manifest is None
            or arguments.embedding_manifest_sha256 is None
        ):
            raise ValueError(
                "OpenCLIP embedding requires --embedding-manifest and "
                "--embedding-manifest-sha256"
            )
        artifact = load_model_artifact_manifest(
            arguments.embedding_manifest,
            expected_kind="multimodal_embedding",
            expected_manifest_sha256=arguments.embedding_manifest_sha256,
            verify_files=True,
        )
        upstream = OpenCLIPEmbeddingBackend(artifact)
    canary = upstream.embed_texts([CANARY_TEXT])[0]
    backend = FormalEmbeddingBackend(
        upstream,
        canary_text=CANARY_TEXT,
        canary_vector=canary,
    )
    backend.verify_canary(force=True)
    return backend


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: Callable[[], EmbeddingBackend] | None = None,
) -> int:
    arguments = _arguments(argv)
    try:
        if arguments.command == "products":
            verified_inputs = (
                arguments.eligibility_manifest,
                arguments.queries,
                arguments.gallery_assets,
                arguments.catalog,
                arguments.asset_root,
            )
            provided = tuple(value is not None for value in verified_inputs)
            if any(provided) and not all(provided):
                raise ValueError(
                    "verified products build requires --eligibility-manifest, --queries, "
                    "--gallery-assets, --catalog, and --asset-root together"
                )
            if arguments.allow_provisional_gallery:
                if any(provided):
                    raise ValueError(
                        "--allow-provisional-gallery cannot be combined with verified inputs"
                    )
                if arguments.limit is None:
                    raise ValueError(
                        "--allow-provisional-gallery is limited to --limit smoke builds"
                    )
                catalog = None
            else:
                if not all(provided):
                    raise ValueError(
                        "formal products build requires a verified query/gallery bundle"
                    )
                catalog = load_asset_catalog(
                    arguments.catalog,
                    arguments.asset_root,
                    verify_files=True,
                )
            output = arguments.output or (
                PRODUCT_INDEX_DIR.with_name(f"{PRODUCT_INDEX_DIR.name}-smoke")
                if arguments.limit is not None
                else PRODUCT_INDEX_DIR
            )
            if backend_factory is not None:
                backend = backend_factory()
            elif all(provided) and arguments.limit is None:
                backend = _formal_backend(arguments)
            else:
                backend = _default_backend(output)
            report = build_product_index(
                arguments.source,
                output,
                backend,
                limit=arguments.limit,
                eligibility_manifest_path=arguments.eligibility_manifest,
                query_artifact_path=arguments.queries,
                gallery_artifact_path=arguments.gallery_assets,
                asset_catalog=catalog,
                allow_provisional_gallery=arguments.allow_provisional_gallery,
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
            return 0

        if arguments.command == "kb-catalog":
            catalog = build_kb_catalog(
                arguments.encyclopedia,
                arguments.recipes,
                arguments.output,
                expected_encyclopedia_sha256=arguments.encyclopedia_sha256,
                expected_recipe_sha256=arguments.recipes_sha256,
                limit_per_kind=arguments.limit_per_kind,
                allow_provisional=arguments.allow_provisional,
            )
            print(
                json.dumps(
                    catalog.manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if arguments.command == "kb":
            bundle = build_kb_bundle(
                arguments.catalog,
                arguments.output,
                expected_catalog_sha256=arguments.catalog_sha256,
                limit_per_kind=arguments.limit_per_kind,
                allow_provisional=arguments.allow_provisional,
            )
            print(
                json.dumps(
                    bundle.manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if arguments.command == "kb-status":
            catalog = load_kb_catalog(
                arguments.catalog,
                allow_provisional=arguments.allow_provisional,
                expected_catalog_sha256=arguments.catalog_sha256,
            )
            bundle = load_kb_bundle(
                arguments.output,
                allow_provisional=arguments.allow_provisional,
                catalog=catalog,
                expected_bundle_sha256=arguments.bundle_sha256,
            )
            print(
                json.dumps(
                    {
                        "bundle_sha256": bundle.manifest.bundle_sha256,
                        "catalog_sha256": bundle.manifest.catalog_sha256,
                        "complete": bundle.manifest.complete,
                        "encyclopedia_entries": len(bundle.encyclopedia.entries),
                        "mode": bundle.manifest.mode,
                        "recipe_entries": len(bundle.recipe.entries),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if arguments.command == "model-manifest":
            artifacts = dict(arguments.artifact)
            if len(artifacts) != len(arguments.artifact):
                raise ValueError("model artifact roles must be unique")
            verified = publish_model_artifact_manifest(
                arguments.output,
                artifact_kind=arguments.kind,
                model_id=arguments.model_id,
                backend_name=arguments.backend_name,
                backend_version=arguments.backend_version,
                artifacts=artifacts,
            )
            print(
                json.dumps(
                    verified.manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if arguments.command == "model-status":
            verified = load_model_artifact_manifest(
                arguments.manifest,
                expected_kind=arguments.kind,
                expected_manifest_sha256=arguments.manifest_sha256,
                verify_files=True,
            )
            print(
                json.dumps(
                    verified.manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if arguments.command == "registry":
            registry = build_mvp_registry_spec()
            publish_registry_manifest(registry, arguments.output)
            print(
                json.dumps(
                    registry.manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if arguments.command == "registry-status":
            manifest = load_registry_manifest(arguments.output)
            print(
                json.dumps(
                    manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        index = ProductIndex.load(
            arguments.output,
            allow_incomplete=arguments.allow_incomplete,
            allow_provisional_gallery=arguments.allow_provisional_gallery,
        )
        eligibility = index.manifest["query_gallery_eligibility"]
        print(
            json.dumps(
                {
                    "complete": index.manifest["complete"],
                    "dimension": index.manifest["dimension"],
                    "products": index.manifest["product_count"],
                    "eligibility_mode": eligibility["mode"],
                    "eligibility_sha256": eligibility.get("eligibility_sha256"),
                    "query_artifact_sha256": eligibility.get("query_artifact_sha256"),
                    "gallery_artifact_sha256": eligibility.get(
                        "gallery_artifact_sha256"
                    ),
                    "asset_catalog_sha256": eligibility.get("asset_catalog_sha256"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(f"build-index: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
