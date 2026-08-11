"""Run one formal product retrieval tool over an authoritative Query artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

from skillchain.data.asset_catalog import load_asset_catalog
from skillchain.tools.embedding import (
    DashScopeEmbeddingClient,
    EmbeddingBackend,
    FormalEmbeddingBackend,
)
from skillchain.tools.product_index import ProductIndex
from skillchain.tools.product_search import ProductSearchService
from skillchain.tools.registry import (
    MVPToolServices,
    ToolCallError,
    build_mvp_registry,
    load_registry_manifest,
)
from skillchain.tools.retrieval_run import (
    RegistryRetrievalExecutor,
    create_registry_retrieval_run,
)
from skillchain.tools.settings import (
    PRODUCT_INDEX_DIR,
    PRODUCT_QUERY_ARTIFACT_PATH,
    TOOL_REGISTRY_MANIFEST_PATH,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        choices=(
            "image_product_search",
            "text_product_search",
            "style_similar_search",
        ),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--queries", type=Path, default=PRODUCT_QUERY_ARTIFACT_PATH)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--product-index", type=Path, default=PRODUCT_INDEX_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry-manifest",
        type=Path,
        default=TOOL_REGISTRY_MANIFEST_PATH,
    )
    return parser.parse_args(argv)


class _UnavailableService:
    def __getattr__(self, _name: str):
        def unavailable(*_args, **_kwargs):
            raise ToolCallError(
                "runtime_unconfigured",
                "tool runtime is not configured for this retrieval run",
            )

        return unavailable


def _default_backend(
    index: ProductIndex,
) -> FormalEmbeddingBackend:
    return FormalEmbeddingBackend(
        DashScopeEmbeddingClient(),
        canary_text=index.canary_text,
        canary_vector=index.canary_vector,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: Callable[[ProductIndex], EmbeddingBackend] | None = None,
) -> int:
    arguments = _arguments(argv)
    try:
        catalog = load_asset_catalog(
            arguments.catalog,
            arguments.asset_root,
            verify_files=True,
        )
        index = ProductIndex.load(arguments.product_index)
        backend = (
            backend_factory(index)
            if backend_factory is not None
            else _default_backend(index)
        )
        verify_canary = getattr(backend, "verify_canary", None)
        if not callable(verify_canary):
            raise ValueError(
                "formal retrieval backend must support an uncached canary check"
            )
        verify_canary(force=True)
        product_service = ProductSearchService(
            index,
            backend,
            query_artifact=arguments.queries,
        )
        unavailable = _UnavailableService()
        registry = build_mvp_registry(
            MVPToolServices(
                product_search=product_service,
                kb_lookup=unavailable,
                object_detection=unavailable,
                document_ocr=unavailable,
                safety_approval_for=lambda _query_id, _asset_id: None,
            ),
            _allow_unconfigured=True,
        )
        load_registry_manifest(
            arguments.registry_manifest,
            expected_registry=registry,
        )
        executor = RegistryRetrievalExecutor(
            registry=registry,
            tool_name=arguments.tool,
            product_search_service=product_service,
        )
        verified = create_registry_retrieval_run(
            arguments.queries,
            arguments.output,
            catalog,
            product_service.artifact_binding,
            executor,
            run_id=arguments.run_id,
        )
        print(
            json.dumps(
                verified.manifest.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(f"run-retrieval: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
