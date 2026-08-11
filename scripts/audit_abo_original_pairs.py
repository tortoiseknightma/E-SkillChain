"""Audit human-approved ABO original pairs for local exact/near-duplicate leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.data.abo_pair_audit import audit_abo_original_pairs
from skillchain.data.asset_catalog import NearDuplicatePolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--packet-manifest-sha256", required=True)
    parser.add_argument("--human-review-root", type=Path, required=True)
    parser.add_argument("--human-review-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-phash-hamming-distance", type=int, default=4)
    arguments = parser.parse_args()
    result = audit_abo_original_pairs(
        packet_root=arguments.packet_root,
        expected_packet_manifest_sha256=arguments.packet_manifest_sha256,
        human_review_root=arguments.human_review_root,
        expected_human_review_manifest_sha256=(arguments.human_review_manifest_sha256),
        output_path=arguments.output,
        near_duplicate_policy=NearDuplicatePolicy(
            max_phash_hamming_distance=arguments.max_phash_hamming_distance,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
