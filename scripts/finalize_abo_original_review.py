"""Validate exported ABO decisions and publish a canonical human ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.data.abo_original_review import (
    finalize_abo_original_review_decisions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--packet-manifest-sha256", required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = finalize_abo_original_review_decisions(
        packet_root=arguments.packet_root,
        expected_packet_manifest_sha256=arguments.packet_manifest_sha256,
        decisions_path=arguments.decisions,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
