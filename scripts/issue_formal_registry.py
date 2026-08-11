"""Create or verify the external lock for the authority-issued tool registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from skillchain.tools.production_registry import (
    load_authority_issued_formal_registry,
    publish_formal_registry_runtime_candidate,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    candidate = commands.add_parser(
        "candidate",
        help="run all preflights and create an unapproved runtime-lock candidate",
    )
    _add_artifact_arguments(candidate)
    candidate.add_argument("--destination", type=Path, required=True)

    verify = commands.add_parser(
        "verify",
        help="rebuild and verify an externally pinned runtime lock",
    )
    _add_artifact_arguments(verify)
    verify.add_argument("--runtime-lock", type=Path, required=True)
    verify.add_argument("--expected-runtime-lock-sha256", required=True)
    return parser


def _add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--expected-artifact-lock-sha256", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "candidate":
            lock = publish_formal_registry_runtime_candidate(
                args.artifact_lock,
                args.artifact_root,
                args.destination,
                expected_artifact_lock_file_sha256=(
                    args.expected_artifact_lock_sha256
                ),
            )
            content = args.destination.read_bytes()
            payload = {
                "artifact_lock_file_sha256": lock.artifact_lock_file_sha256,
                "formal_eligible": False,
                "kind": "formal-registry-runtime-lock-candidate",
                "path": str(args.destination),
                "registry_runtime_sha256": lock.registry_runtime_sha256,
                "registry_sha256": lock.registry_sha256,
                "runtime_lock_file_sha256": sha256_bytes(content),
            }
        else:
            issued = load_authority_issued_formal_registry(
                args.artifact_lock,
                args.runtime_lock,
                args.artifact_root,
                expected_artifact_lock_file_sha256=(
                    args.expected_artifact_lock_sha256
                ),
                expected_runtime_lock_file_sha256=(
                    args.expected_runtime_lock_sha256
                ),
            )
            payload = {
                "artifact_lock_file_sha256": issued.artifact_lock_file_sha256,
                "authority_issued": True,
                "formal_eligible": True,
                "kind": "verified-authority-issued-formal-registry",
                "registry_runtime_sha256": (
                    issued.snapshot.registry_runtime_sha256
                ),
                "registry_sha256": issued.snapshot.registry_sha256,
                "runtime_lock_file_sha256": issued.runtime_lock_file_sha256,
                "tool_count": len(issued.snapshot.tool_runtime_bindings),
            }
        sys.stdout.buffer.write(canonical_json_bytes(payload))
        return 0
    except Exception as error:
        print(f"issue-formal-registry: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
