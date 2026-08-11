"""Build an exact-lock-bound FashionIQ image adapter bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillchain.data.fashioniq import build_fashioniq_image_adapter
from skillchain.data.source_review import (
    load_source_review_policy,
    load_verified_source_review_ledger,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--source-lock-sha256", required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    parser.add_argument("--license-evidence-sha256", required=True)
    parser.add_argument("--source-review-policy", type=Path, required=True)
    parser.add_argument("--source-review-policy-sha256", required=True)
    parser.add_argument("--source-portfolio-sha256", required=True)
    parser.add_argument("--source-review-ledger", type=Path, required=True)
    parser.add_argument("--source-review-ledger-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    policy = load_source_review_policy(
        arguments.source_review_policy,
        expected_policy_file_sha256=arguments.source_review_policy_sha256,
        expected_portfolio_sha256=arguments.source_portfolio_sha256,
    )
    ledger = load_verified_source_review_ledger(
        arguments.source_review_ledger,
        policy,
        policy_file_sha256=arguments.source_review_policy_sha256,
        expected_ledger_file_sha256=arguments.source_review_ledger_sha256,
    )
    result = build_fashioniq_image_adapter(
        raw_root=arguments.raw_root,
        source_lock_path=arguments.source_lock,
        expected_source_lock_sha256=arguments.source_lock_sha256,
        license_evidence_path=arguments.license_evidence,
        expected_license_evidence_sha256=arguments.license_evidence_sha256,
        source_review_ledger=ledger,
        output_dir=arguments.output,
    )
    print(
        json.dumps(
            {
                "accepted_assets": result.manifest.accepted_assets,
                "excluded_locked": result.manifest.excluded_locked,
                "manifest_path": str(result.output_dir / "manifest.json"),
                "manifest_sha256": result.manifest_sha256,
                "output_dir": str(result.output_dir),
                "source_lock_sha256": result.manifest.source_lock_sha256,
                "source_review_ledger_sha256": (
                    result.manifest.source_review_ledger_sha256
                ),
                "status": "published",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
