"""Assemble a completed JD Pan Products-10K split archive without moving parts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.products10k_parts import (  # noqa: E402
    ProductPartsError,
    assemble_split_archive,
    expected_md5,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--md5-manifest", type=Path, required=True)
    parser.add_argument("--archive-name", required=True)
    args = parser.parse_args()
    try:
        size = assemble_split_archive(
            parts_directory=args.parts_dir,
            stem=args.stem,
            destination=args.destination,
            expected_md5_hex=expected_md5(args.md5_manifest, args.archive_name),
        )
    except ProductPartsError as error:
        parser.error(str(error))
    print(f"assembled={args.destination} bytes={size}")


if __name__ == "__main__":
    main()
