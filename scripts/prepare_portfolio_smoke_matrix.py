"""Create a verified Portfolio 1-query x 5-config checklist without model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.portfolio_checklist import (  # noqa: E402
    PortfolioChecklistError,
    build_portfolio_five_config_run_checklist,
    create_portfolio_five_config_run_checklist,
)
from skillchain.evaluation.portfolio_inputs import (  # noqa: E402
    PortfolioInputError,
    PortfolioRemoteProcessingFiles,
    load_verified_portfolio_dev_mini_inputs,
)


DEFAULT_MATRIX_RUN_ID = "portfolio-dev-mini-smoke-dm-019-v1"

PLAN_SHA256 = "5e3b0d67545c3d9a311df6c133c427564e4064174158eddf603b3b075e9fa6be"
PLAN_MANIFEST_FILE_SHA256 = (
    "59f63c873f49bc2610df9547ddd5933d9019d6c3eb6dd19b4874f6a89a3f953a"
)
ACCEPTED_LEDGER_SHA256 = (
    "79835fa73bfde60a005ea16480c1efc04635b3fee64f8a01bae893fe5cd4e69a"
)
QUERY_ARTIFACT_SHA256 = (
    "6f8eda4fe663733708d6e797c954e58f098d3938857c6f2b143e24e202437f03"
)
CAPABILITY_ASSIGNMENTS_SHA256 = (
    "b579c616251b1734cd0cf9da4dbc53f9e90dec3c06c42aaac39b60738df2ca2f"
)
SEED_SET_SHA256 = (
    "767a10caa954890088efe85a3b5ba50628e0ab772bafdf06b6b43dc7159c3250"
)
BASE_CATALOG_SHA256 = (
    "df17da7dd7ad7e1077c2e6d084a5e1e79516021979915c9881314b7815c28c96"
)
RUNTIME_CATALOG_SHA256 = (
    "d7f44371a03af0e54e6671012bb3ff54ba6181c0dbff3ca8d219da1a362284a8"
)
AUTHORIZATION_FILE_SHA256 = (
    "b31c622eb2e9bd97495746092f7214a04548af8e0ff7438c90eeb83627bea7b4"
)
RECEIPT_FILE_SHA256 = (
    "ffa198f253e2f47e83037f2c5ef6867a8c1c1323a72b9bcdfbe0ded3463eb858"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Create-only path for the canonical checklist JSON.",
    )
    parser.add_argument(
        "--matrix-run-id",
        default=DEFAULT_MATRIX_RUN_ID,
        help="Stable run namespace; no execution is authorized by this ID.",
    )
    parser.add_argument(
        "--query-id",
        help=(
            "Optional exact accepted query ID. Without it, the frozen local "
            "selection policy deterministically chooses dm-019."
        ),
    )
    return parser


def _remote_files() -> PortfolioRemoteProcessingFiles:
    permission_root = (
        REPOSITORY_ROOT
        / "specs"
        / "data_sources"
        / "c2"
        / "portfolio-mini-remote-processing-v1"
    )
    clean_root = REPOSITORY_ROOT / "data" / "clean"
    return PortfolioRemoteProcessingFiles(
        authorization_file=permission_root / "owner-authorization-v2.json",
        expected_authorization_file_sha256=AUTHORIZATION_FILE_SHA256,
        receipt_file=permission_root / "catalog-v2-receipt-v2.json",
        expected_receipt_file_sha256=RECEIPT_FILE_SHA256,
        selection_manifest=clean_root / "query_images" / "selection-manifest.json",
        dataset_assets=clean_root / "query_images" / "dataset-assets.jsonl",
        base_catalog_dir=clean_root / "portfolio-mini-asset-catalog-v1",
        output_catalog_dir=clean_root / "portfolio-mini-asset-catalog-v2",
        asset_root=clean_root,
    )


def load_current_portfolio_inputs():
    """Load the repository's externally committed Portfolio snapshot."""

    return load_verified_portfolio_dev_mini_inputs(
        queries_root=REPOSITORY_ROOT / "data" / "queries",
        plan_path=REPOSITORY_ROOT / "data" / "queries" / "plans" / "dev_mini.json",
        capability_assignments_path=(
            REPOSITORY_ROOT
            / "data"
            / "clean"
            / "portfolio-mini-capability-assignments-v1.jsonl"
        ),
        expected_plan_sha256=PLAN_SHA256,
        expected_plan_manifest_file_sha256=PLAN_MANIFEST_FILE_SHA256,
        expected_accepted_ledger_sha256=ACCEPTED_LEDGER_SHA256,
        expected_query_artifact_sha256=QUERY_ARTIFACT_SHA256,
        expected_capability_assignments_sha256=CAPABILITY_ASSIGNMENTS_SHA256,
        expected_seed_set_sha256=SEED_SET_SHA256,
        expected_base_catalog_sha256=BASE_CATALOG_SHA256,
        expected_output_catalog_sha256=RUNTIME_CATALOG_SHA256,
        remote_files=_remote_files(),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        inputs = load_current_portfolio_inputs()
        checklist = build_portfolio_five_config_run_checklist(
            inputs,
            matrix_run_id=arguments.matrix_run_id,
            query_id=arguments.query_id,
        )
        created = create_portfolio_five_config_run_checklist(
            arguments.output,
            checklist,
            inputs=inputs,
        )
    except (
        FileExistsError,
        OSError,
        PortfolioChecklistError,
        PortfolioInputError,
        TypeError,
        ValueError,
    ) as error:
        print(f"prepare-portfolio-smoke-matrix: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "checklist_path": str(created.path),
                "config_count": created.checklist.planned_config_count,
                "execution_authorized": (
                    created.checklist.execution_authorized
                ),
                "file_sha256": created.file_sha256,
                "model_calls_performed": (
                    created.checklist.model_calls_performed
                ),
                "query_id": created.checklist.selected_query.query_id,
                "status": created.checklist.status,
                "track": created.checklist.track,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
