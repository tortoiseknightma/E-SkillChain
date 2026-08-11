"""Acquire exact official originals for the selected ABO review pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.data.abo_original_review import prepare_abo_original_review_packet
from skillchain.data.abo_original_review import (
    prepare_abo_original_replenishment_packet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-packet-root", type=Path, required=True)
    parser.add_argument("--compact-manifest-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-workers", type=int, default=8)
    parser.add_argument("--previous-packet-root", type=Path)
    parser.add_argument("--previous-packet-manifest-sha256")
    arguments = parser.parse_args()
    previous_values = (
        arguments.previous_packet_root,
        arguments.previous_packet_manifest_sha256,
    )
    if any(value is not None for value in previous_values) and not all(
        value is not None for value in previous_values
    ):
        parser.error(
            "--previous-packet-root and "
            "--previous-packet-manifest-sha256 must be supplied together"
        )
    if arguments.previous_packet_root is None:
        result = prepare_abo_original_review_packet(
            compact_packet_root=arguments.compact_packet_root,
            expected_compact_manifest_sha256=arguments.compact_manifest_sha256,
            raw_root=arguments.raw_root,
            output_dir=arguments.output_dir,
            maximum_workers=arguments.maximum_workers,
        )
    else:
        result = prepare_abo_original_replenishment_packet(
            compact_packet_root=arguments.compact_packet_root,
            expected_compact_manifest_sha256=arguments.compact_manifest_sha256,
            raw_root=arguments.raw_root,
            previous_packet_root=arguments.previous_packet_root,
            expected_previous_packet_manifest_sha256=(
                arguments.previous_packet_manifest_sha256
            ),
            output_dir=arguments.output_dir,
            maximum_workers=arguments.maximum_workers,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
