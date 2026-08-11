"""Create-only preparation CLI for a paper-aligned Stage 1 Creator packet.

The command verifies inputs and writes preparation artifacts.  It never invokes
an author model and never produces a Skill Bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.stage1 import (
    build_query_selection,
    load_query_selection,
    prepare_stage1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoring-input", type=Path, required=True)
    parser.add_argument(
        "--expected-authoring-input-sha256",
        required=True,
        help=(
            "Externally supplied SHA-256 of the byte-identical common "
            "CodexAuthoringInput."
        ),
    )
    parser.add_argument("--queries-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        help="Explicit opt_pool query ID; repeat for every selected query.",
    )
    selection.add_argument(
        "--selection-file",
        type=Path,
        help="Canonical QuerySelection JSON with unique lexically ordered IDs.",
    )
    parser.add_argument(
        "--expected-selection-file-sha256",
        help="Required external SHA-256 when --selection-file is used.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.selection_file is not None:
        if args.expected_selection_file_sha256 is None:
            parser.error("--selection-file requires --expected-selection-file-sha256")
        selection = load_query_selection(
            args.selection_file,
            expected_file_sha256=args.expected_selection_file_sha256,
        )
    else:
        if args.expected_selection_file_sha256 is not None:
            parser.error("--expected-selection-file-sha256 requires --selection-file")
        selection = build_query_selection(args.query_ids or ())

    prepared = prepare_stage1(
        authoring_input_path=args.authoring_input,
        expected_authoring_input_file_sha256=(args.expected_authoring_input_sha256),
        queries_root=args.queries_root,
        selected_query_ids=selection.query_ids,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "prepared_not_invoked",
                "model_invoked": False,
                "bank_generated": False,
                "selected_query_count": prepared.selected_query_count,
                "trajectory_bundle": str(prepared.trajectory_bundle_path),
                "trajectory_bundle_file_sha256": (
                    prepared.trajectory_bundle_file_sha256
                ),
                "s1_creator_packet": str(prepared.s1_creator_packet_path),
                "s1_creator_packet_file_sha256": (
                    prepared.s1_creator_packet_file_sha256
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
