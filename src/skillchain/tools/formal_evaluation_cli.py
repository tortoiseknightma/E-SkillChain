"""Dedicated CLI for formal tool-run creation, verification, and evaluation.

The CLI has no diagnostic/formal mode switch.  Every subcommand uses the
formal trust boundary and requires external digests.  A runtime factory is an
explicit ``module:callable`` returning :class:`FormalRuntimeContext`; the
resulting live registry identities are still checked against the supplied
external digests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Sequence

from skillchain.data.asset_catalog import AssetCatalog
from skillchain.evaluation.assistant_runs import load_verified_phase4_inputs
from skillchain.tools.formal_evaluation import (
    create_formal_tool_run,
    evaluate_formal_tool_run,
    load_formal_gold_assignments,
    load_verified_formal_tool_run,
)
from skillchain.tools.document_safety import DocumentSafetyCatalog
from skillchain.tools.registry import ToolRegistry
from skillchain.tools.serialization import canonical_json_bytes


@dataclass(frozen=True)
class FormalRuntimeContext:
    catalog: AssetCatalog
    registry: ToolRegistry
    document_safety_catalog: DocumentSafetyCatalog | None = None
    multi_product_executor: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, AssetCatalog):
            raise TypeError("formal runtime catalog must be an AssetCatalog")
        if not isinstance(self.registry, ToolRegistry):
            raise TypeError("formal runtime registry must be a ToolRegistry")
        if self.document_safety_catalog is not None and not isinstance(
            self.document_safety_catalog, DocumentSafetyCatalog
        ):
            raise TypeError(
                "formal runtime document_safety_catalog must be a DocumentSafetyCatalog"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Externally locked formal tool-run and evaluation workflow"
    )
    parser.add_argument(
        "--runtime-factory",
        help="module:callable returning FormalRuntimeContext",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create-run")
    _add_upstream_arguments(create)
    _add_runtime_digest_arguments(create)
    create.add_argument("--destination", type=Path, required=True)
    create.add_argument("--run-id", required=True)

    verify = commands.add_parser("verify-run")
    _add_upstream_arguments(verify)
    _add_runtime_digest_arguments(verify)
    _add_run_arguments(verify)

    evaluate = commands.add_parser("evaluate-run")
    _add_upstream_arguments(evaluate)
    _add_runtime_digest_arguments(evaluate)
    _add_run_arguments(evaluate)
    evaluate.add_argument("--destination", type=Path, required=True)
    evaluate.add_argument("--evaluation-id", required=True)
    return parser


def _add_upstream_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--expected-assignment-sha256", required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--expected-gold-sha256", required=True)
    parser.add_argument("--gold-review-ledger", type=Path, required=True)
    parser.add_argument("--expected-gold-review-ledger-sha256", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--expected-query-sha256", required=True)
    parser.add_argument("--split-assignment", type=Path, required=True)
    parser.add_argument("--expected-split-assignment-sha256", required=True)
    parser.add_argument("--expected-asset-catalog-sha256", required=True)
    parser.add_argument("--expected-document-safety-catalog-sha256")
    parser.add_argument("--expected-document-safety-review-ledger-sha256")
    parser.add_argument("--assistant-queries", type=Path, required=True)
    parser.add_argument("--expected-assistant-query-sha256", required=True)
    parser.add_argument("--assistant-split-manifest", type=Path, required=True)
    parser.add_argument("--expected-assistant-split-manifest-sha256", required=True)
    parser.add_argument("--assistant-rubric", type=Path, required=True)
    parser.add_argument("--expected-assistant-rubric-sha256", required=True)
    parser.add_argument("--assistant-selection", type=Path, required=True)
    parser.add_argument("--expected-assistant-selection-sha256", required=True)


def _add_runtime_digest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-runtime-sha256", required=True)
    parser.add_argument("--expected-multi-product-runtime-sha256")


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-run-bundle-sha256", required=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_context: FormalRuntimeContext | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    context = runtime_context or _load_runtime_context(args.runtime_factory)
    assistant_inputs = load_verified_phase4_inputs(
        query_path=args.assistant_queries,
        expected_query_sha256=args.expected_assistant_query_sha256,
        split_manifest_path=args.assistant_split_manifest,
        expected_split_manifest_sha256=(args.expected_assistant_split_manifest_sha256),
        rubric_path=args.assistant_rubric,
        expected_rubric_file_sha256=args.expected_assistant_rubric_sha256,
        selection_path=args.assistant_selection,
        expected_selection_file_sha256=args.expected_assistant_selection_sha256,
    )
    assignments = load_formal_gold_assignments(
        args.assignment,
        expected_assignment_sha256=args.expected_assignment_sha256,
        gold_path=args.gold,
        expected_gold_sha256=args.expected_gold_sha256,
        gold_review_ledger_path=args.gold_review_ledger,
        expected_gold_review_ledger_sha256=(args.expected_gold_review_ledger_sha256),
        query_path=args.queries,
        expected_query_sha256=args.expected_query_sha256,
        split_assignment_path=args.split_assignment,
        expected_split_assignment_sha256=args.expected_split_assignment_sha256,
        catalog=context.catalog,
        expected_asset_catalog_sha256=args.expected_asset_catalog_sha256,
        document_safety_catalog=context.document_safety_catalog,
        expected_document_safety_catalog_sha256=(
            args.expected_document_safety_catalog_sha256
        ),
        expected_document_safety_review_ledger_sha256=(
            args.expected_document_safety_review_ledger_sha256
        ),
        assistant_inputs=assistant_inputs,
    )
    common = {
        "expected_registry_sha256": args.expected_registry_sha256,
        "expected_registry_runtime_sha256": args.expected_registry_runtime_sha256,
        "multi_product_executor": context.multi_product_executor,
        "expected_multi_product_runtime_sha256": (
            args.expected_multi_product_runtime_sha256
        ),
    }
    if args.command == "create-run":
        result = create_formal_tool_run(
            assignments,
            context.registry,
            args.destination,
            run_id=args.run_id,
            **common,
        )
        payload = {
            "kind": "created-formal-tool-run",
            "path": str(result.root),
            "run_bundle_sha256": result.run_bundle_sha256,
        }
    else:
        run = load_verified_formal_tool_run(
            args.run_dir,
            assignments,
            context.registry,
            expected_run_bundle_sha256=args.expected_run_bundle_sha256,
            **common,
        )
        if args.command == "verify-run":
            payload = {
                "kind": "verified-formal-tool-run",
                "path": str(run.root),
                "query_count": run.manifest.query_count,
                "run_bundle_sha256": run.manifest.run_bundle_sha256,
            }
        else:
            evaluation = evaluate_formal_tool_run(
                run,
                args.destination,
                evaluation_id=args.evaluation_id,
            )
            payload = {
                "kind": "verified-formal-tool-evaluation",
                "path": str(evaluation.root),
                "case_count": evaluation.manifest.case_count,
                "manifest_sha256": evaluation.manifest.manifest_sha256,
                "manifest_file_sha256": evaluation.manifest_file_sha256,
            }
    sys.stdout.buffer.write(canonical_json_bytes(payload))
    return 0


def _load_runtime_context(factory_path: str | None) -> FormalRuntimeContext:
    if factory_path is None:
        raise ValueError(
            "formal CLI requires --runtime-factory module:callable or an injected context"
        )
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("runtime factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError("runtime factory target must be callable")
    context = factory()
    if not isinstance(context, FormalRuntimeContext):
        raise TypeError("runtime factory must return FormalRuntimeContext")
    return context


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())


__all__ = ["FormalRuntimeContext", "build_parser", "main"]
