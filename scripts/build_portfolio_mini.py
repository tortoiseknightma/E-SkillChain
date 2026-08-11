"""Build the non-formal 184-image Portfolio mini pool or its assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.portfolio_mini import (  # noqa: E402
    assemble_portfolio_mini,
    build_portfolio_mini_capability_assignments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    assemble = subcommands.add_parser(
        "assemble",
        help="select/copy the 184 existing query images",
    )
    assemble.add_argument("--abo-audit-receipt", type=Path, required=True)
    assemble.add_argument("--abo-original-root", type=Path, required=True)
    assemble.add_argument("--fashioniq-adapter-root", type=Path, required=True)
    assemble.add_argument("--fashioniq-manifest-sha256", required=True)
    assemble.add_argument("--fashioniq-asset-root", type=Path, required=True)
    assemble.add_argument("--rpc-adapter-root", type=Path, required=True)
    assemble.add_argument("--rpc-manifest-sha256", required=True)
    assemble.add_argument("--inaturalist-manifest", type=Path, required=True)
    assemble.add_argument("--inaturalist-asset-root", type=Path, required=True)
    assemble.add_argument("--commons-document-manifest", type=Path, required=True)
    assemble.add_argument("--commons-document-asset-root", type=Path, required=True)
    assemble.add_argument("--isia-food-manifest", type=Path, required=True)
    assemble.add_argument("--isia-food-asset-root", type=Path, required=True)
    assemble.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "clean" / "query_images",
    )

    assign = subcommands.add_parser(
        "assign",
        help="bind the selected images to a verified AssetCatalog",
    )
    assign.add_argument("--selection-manifest", type=Path, required=True)
    assign.add_argument("--asset-catalog", type=Path, required=True)
    assign.add_argument("--asset-root", type=Path, required=True)
    assign.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "assemble":
        result = assemble_portfolio_mini(
            abo_audit_receipt=arguments.abo_audit_receipt,
            abo_original_root=arguments.abo_original_root,
            fashioniq_adapter_root=arguments.fashioniq_adapter_root,
            fashioniq_manifest_sha256=arguments.fashioniq_manifest_sha256,
            fashioniq_asset_root=arguments.fashioniq_asset_root,
            rpc_adapter_root=arguments.rpc_adapter_root,
            rpc_manifest_sha256=arguments.rpc_manifest_sha256,
            inaturalist_manifest=arguments.inaturalist_manifest,
            inaturalist_asset_root=arguments.inaturalist_asset_root,
            commons_document_manifest=arguments.commons_document_manifest,
            commons_document_asset_root=arguments.commons_document_asset_root,
            isia_food_manifest=arguments.isia_food_manifest,
            isia_food_asset_root=arguments.isia_food_asset_root,
            output_root=arguments.output_root,
        )
        payload = {
            "asset_count": result.asset_count,
            "dataset_assets_path": str(result.dataset_assets_path),
            "dataset_assets_sha256": result.dataset_assets_sha256,
            "output_root": str(result.output_root),
            "selection_manifest_path": str(result.selection_manifest_path),
            "selection_manifest_sha256": result.selection_manifest_sha256,
            "status": "published",
            "track": "portfolio",
        }
    else:
        result = build_portfolio_mini_capability_assignments(
            selection_manifest=arguments.selection_manifest,
            asset_catalog_dir=arguments.asset_catalog,
            asset_root=arguments.asset_root,
            output_path=arguments.output,
        )
        payload = {
            "assignment_count": result.assignment_count,
            "assignment_sha256": result.assignment_sha256,
            "output_path": str(result.output_path),
            "selection_manifest_sha256": result.selection_manifest_sha256,
            "status": "published",
            "track": "portfolio",
        }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
