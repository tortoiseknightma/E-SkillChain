"""Publish or verify a Portfolio remote-processing catalog overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.portfolio_remote_processing import (  # noqa: E402
    PortfolioRemoteProcessingError,
    load_portfolio_remote_processing_receipt,
    publish_remote_authorized_portfolio_catalog,
    verify_portfolio_remote_processing_runtime,
    verify_remote_authorized_query_bindings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    publish = subcommands.add_parser(
        "publish",
        help="create a permission-only AssetCatalog overlay and receipt",
    )
    publish.add_argument("--authorization", type=Path, required=True)
    publish.add_argument("--authorization-sha256", required=True)
    publish.add_argument("--selection-manifest", type=Path, required=True)
    publish.add_argument("--base-catalog", type=Path, required=True)
    publish.add_argument("--asset-root", type=Path, required=True)
    publish.add_argument("--output-catalog", type=Path, required=True)
    publish.add_argument("--receipt-output", type=Path, required=True)

    verify = subcommands.add_parser(
        "verify-receipt",
        help="strictly reload a published receipt",
    )
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--receipt-sha256", required=True)

    bindings = subcommands.add_parser(
        "verify-query-bindings",
        help="verify every accepted query against the cloud-enabled catalog",
    )
    bindings.add_argument("--catalog", type=Path, required=True)
    bindings.add_argument("--asset-root", type=Path, required=True)
    bindings.add_argument("--queries", type=Path, required=True)
    bindings.add_argument("--query-sha256")

    runtime = subcommands.add_parser(
        "verify-runtime",
        help="freshly verify owner authority and every runtime artifact binding",
    )
    runtime.add_argument("--authorization", type=Path, required=True)
    runtime.add_argument("--authorization-sha256", required=True)
    runtime.add_argument("--receipt", type=Path, required=True)
    runtime.add_argument("--receipt-sha256", required=True)
    runtime.add_argument("--selection-manifest", type=Path, required=True)
    runtime.add_argument("--dataset-assets", type=Path, required=True)
    runtime.add_argument("--base-catalog", type=Path, required=True)
    runtime.add_argument("--output-catalog", type=Path, required=True)
    runtime.add_argument("--asset-root", type=Path, required=True)
    runtime.add_argument("--plan", type=Path, required=True)
    runtime.add_argument("--queries", type=Path, required=True)
    runtime.add_argument(
        "--processor",
        choices=(
            "aifast-gemini-feedback",
            "aifast-gemini-judge",
            "dashscope-kimi-feedback",
            "dashscope-kimi-judge",
            "dashscope-qwen-assistant",
        ),
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "publish":
            result = publish_remote_authorized_portfolio_catalog(
                authorization_file=arguments.authorization,
                expected_authorization_sha256=arguments.authorization_sha256,
                selection_manifest=arguments.selection_manifest,
                base_catalog=arguments.base_catalog,
                asset_root=arguments.asset_root,
                output_catalog=arguments.output_catalog,
                receipt_output=arguments.receipt_output,
            )
            payload = {
                "asset_count": result.asset_count,
                "authorization_file_sha256": (result.authorization_file_sha256),
                "output_catalog": str(result.output_catalog),
                "output_catalog_sha256": result.output_catalog_sha256,
                "receipt": str(result.receipt_path),
                "receipt_sha256": result.receipt_sha256,
                "status": "published",
                "track": "portfolio",
            }
        elif arguments.command == "verify-receipt":
            receipt = load_portfolio_remote_processing_receipt(
                arguments.receipt,
                expected_sha256=arguments.receipt_sha256,
            )
            payload = {
                "asset_count": receipt.asset_count,
                "authorization_id": receipt.authorization_id,
                "output_catalog_sha256": receipt.output_catalog_sha256,
                "receipt_sha256": receipt.receipt_sha256,
                "status": "verified",
                "track": "portfolio",
            }
        elif arguments.command == "verify-query-bindings":
            bindings = verify_remote_authorized_query_bindings(
                catalog_dir=arguments.catalog,
                asset_root=arguments.asset_root,
                queries_path=arguments.queries,
                expected_query_artifact_sha256=arguments.query_sha256,
            )
            payload = {
                "catalog_sha256": bindings.catalog_sha256,
                "query_artifact_sha256": bindings.query_artifact_sha256,
                "query_count": bindings.query_count,
                "status": "verified",
                "track": "portfolio",
                "unique_asset_count": bindings.unique_asset_count,
            }
        else:
            runtime = verify_portfolio_remote_processing_runtime(
                authorization_file=arguments.authorization,
                expected_authorization_sha256=arguments.authorization_sha256,
                receipt_file=arguments.receipt,
                expected_receipt_file_sha256=arguments.receipt_sha256,
                selection_manifest=arguments.selection_manifest,
                dataset_assets=arguments.dataset_assets,
                base_catalog=arguments.base_catalog,
                output_catalog=arguments.output_catalog,
                asset_root=arguments.asset_root,
                plan_file=arguments.plan,
                queries_file=arguments.queries,
                processor=arguments.processor,
            )
            payload = {
                "asset_count": len(runtime.catalog.assets),
                "authorization_id": runtime.authorization.authorization_id,
                "catalog_sha256": runtime.catalog.catalog_sha256,
                "dataset_assets_sha256": runtime.dataset_assets_sha256,
                "plan_sha256": runtime.plan_sha256,
                "processor": runtime.processor,
                "query_artifact_sha256": runtime.query_artifact_sha256,
                "receipt_sha256": runtime.receipt.receipt_sha256,
                "scope": runtime.authorization.scope,
                "status": "runtime-verified",
                "track": "portfolio",
            }
    except (PortfolioRemoteProcessingError, FileExistsError) as error:
        print(
            f"authorize-portfolio-remote-processing: {error}",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
