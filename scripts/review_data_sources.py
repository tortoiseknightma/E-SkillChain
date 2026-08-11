"""Verify externally pinned dataset permission and PII review decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from skillchain.data.source_review import (
    load_source_review_ledger,
    load_source_review_policy,
    load_verified_source_review_ledger,
)
from skillchain.tools.serialization import sha256_bytes


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("policy-status", "status", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--policy-sha256", required=True)
        command.add_argument("--portfolio", type=Path, required=True)
    for name in ("status", "verify"):
        ledger_command = commands.choices[name]
        ledger_command.add_argument("--ledger", type=Path, required=True)
        ledger_command.add_argument("--ledger-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        portfolio_sha256 = sha256_bytes(arguments.portfolio.read_bytes())
        policy = load_source_review_policy(
            arguments.policy,
            expected_policy_file_sha256=arguments.policy_sha256,
            expected_portfolio_sha256=portfolio_sha256,
        )
        if arguments.command == "policy-status":
            payload = {
                "policy_id": policy.policy_id,
                "policy_sha256": arguments.policy_sha256,
                "portfolio_sha256": portfolio_sha256,
                "required_sources": [
                    item.source_id for item in policy.requirements if item.required
                ],
                "status": "policy-verified",
            }
        elif arguments.command == "status":
            ledger = load_source_review_ledger(
                arguments.ledger,
                policy,
                policy_file_sha256=arguments.policy_sha256,
                expected_ledger_file_sha256=arguments.ledger_sha256,
            )
            payload = {
                "approved_sources": sorted(ledger.approved),
                "blockers": list(ledger.blockers),
                "decisions": {
                    record.source_id: record.decision for record in ledger.records
                },
                "ledger_sha256": ledger.ledger_file_sha256,
                "policy_id": policy.policy_id,
                "policy_sha256": ledger.policy_file_sha256,
                "status": "approved" if not ledger.blockers else "incomplete",
            }
        else:
            ledger = load_verified_source_review_ledger(
                arguments.ledger,
                policy,
                policy_file_sha256=arguments.policy_sha256,
                expected_ledger_file_sha256=arguments.ledger_sha256,
            )
            payload = {
                "approved_sources": sorted(ledger.approved),
                "ledger_sha256": ledger.ledger_file_sha256,
                "policy_id": policy.policy_id,
                "policy_sha256": ledger.policy_file_sha256,
                "status": "approved",
            }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"review-data-sources: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
