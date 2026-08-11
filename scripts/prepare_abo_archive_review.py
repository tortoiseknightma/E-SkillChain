"""Prepare deterministic ABO compact-archive candidates for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.data.abo_archive_review import prepare_abo_archive_review_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--source-lock-sha256", required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    parser.add_argument("--source-review-policy", type=Path, required=True)
    parser.add_argument("--source-review-policy-sha256", required=True)
    parser.add_argument("--source-portfolio-sha256", required=True)
    parser.add_argument("--source-review-ledger", type=Path, required=True)
    parser.add_argument("--source-review-ledger-sha256", required=True)
    parser.add_argument("--maximum-pair-candidates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = prepare_abo_archive_review_packet(
        required_source_lock_path=arguments.source_lock,
        expected_required_source_lock_sha256=arguments.source_lock_sha256,
        raw_root=arguments.raw_root,
        license_evidence_path=arguments.license_evidence,
        source_review_policy_path=arguments.source_review_policy,
        expected_source_review_policy_sha256=(
            arguments.source_review_policy_sha256
        ),
        expected_source_review_portfolio_sha256=(
            arguments.source_portfolio_sha256
        ),
        source_review_ledger_path=arguments.source_review_ledger,
        expected_source_review_ledger_sha256=(
            arguments.source_review_ledger_sha256
        ),
        output_dir=arguments.output,
        maximum_pair_candidates=arguments.maximum_pair_candidates,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
