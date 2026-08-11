"""Carry exact prior ABO human decisions into an expanded review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.data.abo_review_replenishment import (
    prepare_abo_review_decision_carry_forward,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-packet-root", type=Path, required=True)
    parser.add_argument("--target-packet-manifest-sha256", required=True)
    parser.add_argument("--source-packet-root", type=Path, required=True)
    parser.add_argument("--source-packet-manifest-sha256", required=True)
    parser.add_argument("--source-human-review-root", type=Path, required=True)
    parser.add_argument("--source-human-review-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = prepare_abo_review_decision_carry_forward(
        target_packet_root=arguments.target_packet_root,
        expected_target_packet_manifest_sha256=(
            arguments.target_packet_manifest_sha256
        ),
        source_packet_root=arguments.source_packet_root,
        expected_source_packet_manifest_sha256=(
            arguments.source_packet_manifest_sha256
        ),
        source_human_review_root=arguments.source_human_review_root,
        expected_source_human_review_manifest_sha256=(
            arguments.source_human_review_manifest_sha256
        ),
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
