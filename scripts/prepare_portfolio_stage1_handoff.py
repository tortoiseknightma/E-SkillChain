"""Create a Portfolio dev_mini Stage 1 input handoff without model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.prepare_portfolio_smoke_matrix import (  # noqa: E402
    load_current_portfolio_inputs,
)
from skillchain.evaluation.portfolio_checklist import (  # noqa: E402
    PortfolioChecklistError,
    load_verified_portfolio_five_config_run_checklist,
)
from skillchain.evaluation.portfolio_inputs import (  # noqa: E402
    PortfolioInputError,
)
from skillchain.evaluation.portfolio_stage1 import (  # noqa: E402
    PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS,
    PortfolioStage1Error,
    create_portfolio_s1_handoff,
)


DEFAULT_CHECKLIST = (
    REPOSITORY_ROOT
    / "runs"
    / "portfolio"
    / "portfolio-dev-mini-smoke-dm-019-v1"
    / "run-checklist.json"
)
DEFAULT_CHECKLIST_SHA256 = (
    "83c70a00e20c4878028eda923f0d5c8779681680afd4c475d58941892b76c623"
)
DEFAULT_AUTHORING_INPUT = (
    REPOSITORY_ROOT
    / "specs"
    / "authoring"
    / "authoring-packet-codex-high-v5.json"
)
DEFAULT_AUTHORING_INPUT_SHA256 = (
    "b389da568575e7b1502923db0a4667e70954233924623856b4f96d6a4fdf9cf1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Create-only directory for the three-file Portfolio S1 handoff.",
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=DEFAULT_CHECKLIST,
        help="Externally committed Portfolio 1x5 zero-call checklist.",
    )
    parser.add_argument(
        "--checklist-sha256",
        default=DEFAULT_CHECKLIST_SHA256,
        help="Expected file SHA-256 for --checklist.",
    )
    parser.add_argument(
        "--authoring-input",
        type=Path,
        default=DEFAULT_AUTHORING_INPUT,
        help="Common Codex v5 authoring input shared by LLMStatic and S1.",
    )
    parser.add_argument(
        "--authoring-input-sha256",
        default=DEFAULT_AUTHORING_INPUT_SHA256,
        help="Expected file SHA-256 for --authoring-input.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        inputs = load_current_portfolio_inputs()
        checklist = load_verified_portfolio_five_config_run_checklist(
            arguments.checklist,
            expected_file_sha256=arguments.checklist_sha256,
            inputs=inputs,
        )
        created = create_portfolio_s1_handoff(
            inputs=inputs,
            checklist=checklist,
            authoring_input_path=arguments.authoring_input,
            expected_authoring_input_file_sha256=(
                arguments.authoring_input_sha256
            ),
            output_dir=arguments.output_dir,
        )
    except (
        FileExistsError,
        OSError,
        PortfolioChecklistError,
        PortfolioInputError,
        PortfolioStage1Error,
        TypeError,
        ValueError,
    ) as error:
        print(f"prepare-portfolio-stage1-handoff: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "bank_generated": False,
                "creator_invoked": False,
                "execution_authorized": False,
                "formal_eligible": False,
                "manifest_file_sha256": created.manifest_file_sha256,
                "model_calls_performed": 0,
                "output_dir": str(created.output_dir),
                "purpose": "input_handoff_smoke",
                "selected_query_count": created.selected_query_count,
                "selected_query_ids": list(PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS),
                "status": "prepared_not_invoked",
                "track": "portfolio",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
