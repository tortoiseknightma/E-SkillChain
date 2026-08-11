"""Promote and verify the bounded Recipe1M+ image selection without extracting it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.recipe1m_plus_selection import (  # noqa: E402
    RecipeSelectionError,
    promote_completed_images,
    verify_selected_images,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--selection-root", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "promote":
            report = promote_completed_images(args.selection_root)
            print(
                f"promoted={report.promoted} invalid_partials={report.invalid} "
                f"active_partials={report.active_partials}"
            )
            return
        report = verify_selected_images(args.inventory)
    except RecipeSelectionError as error:
        parser.error(str(error))
    print(
        f"expected={report.expected} verified={report.verified} "
        f"missing={report.missing} invalid={report.invalid}"
    )
    raise SystemExit(0 if report.missing == 0 and report.invalid == 0 else 1)


if __name__ == "__main__":
    main()
