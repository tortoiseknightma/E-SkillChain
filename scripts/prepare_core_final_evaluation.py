#!/usr/bin/env python
"""Compile a zero-call Portfolio core-final evaluation framework artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from skillchain.evaluation.core_final_config import (
    CoreFinalFrameworkError,
    bind_framework_artifact,
    build_core_final_evaluation_framework,
    create_core_final_framework_artifact,
    load_core_final_framework_artifact,
    resolve_framework_slot,
)


_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _binding(value: str) -> tuple[str, Path, str]:
    try:
        name, file_and_hash = value.split("=", 1)
        path_text, digest = file_and_hash.rsplit("@", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected NAME=PATH@SHA256"
        ) from exc
    if not name or not path_text or not _HASH_PATTERN.fullmatch(digest):
        raise argparse.ArgumentTypeError("expected NAME=PATH@SHA256")
    return name, Path(path_text), digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a content-addressed core-final evaluation framework. "
            "This command performs zero model calls and cannot start a core run."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base",
        type=Path,
        help="Existing canonical framework artifact to extend; default is all-pending.",
    )
    parser.add_argument(
        "--artifact-root",
        "--repository-root",
        dest="artifact_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=(
            "Root below which bound artifacts must reside. "
            "--repository-root remains as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--artifact-root-id",
        default="repository",
        help="Portable identifier stored beside paths bound from --artifact-root.",
    )
    parser.add_argument(
        "--bind",
        type=_binding,
        action="append",
        default=[],
        metavar="ROLE=PATH@SHA256",
        help="Verify and bind one required dataset/runtime/evidence artifact.",
    )
    parser.add_argument(
        "--resolve",
        type=_binding,
        action="append",
        default=[],
        metavar="SLOT=PATH@SHA256",
        help=(
            "Resolve one mini-dependent slot with a decision artifact. "
            "Its required evidence roles must already be bound."
        ),
    )
    parser.add_argument(
        "--require-ready-to-freeze",
        action="store_true",
        help="Fail without writing unless no required binding or refinement is pending.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        framework = (
            load_core_final_framework_artifact(args.base)
            if args.base is not None
            else build_core_final_evaluation_framework()
        )
        for role, path, digest in args.bind:
            framework = bind_framework_artifact(
                framework,
                role=role,
                path=path,
                expected_sha256=digest,
                repository_root=args.artifact_root,
                artifact_root_id=args.artifact_root_id,
            )
        for slot_id, path, digest in args.resolve:
            framework = resolve_framework_slot(
                framework,
                slot_id=slot_id,
                decision_artifact_path=path,
                expected_sha256=digest,
                repository_root=args.artifact_root,
                artifact_root_id=args.artifact_root_id,
            )
        if args.require_ready_to_freeze and framework.blockers:
            print(
                f"not ready to freeze: {len(framework.blockers)} blocker(s)",
                file=sys.stderr,
            )
            return 3
        output = create_core_final_framework_artifact(framework, args.output)
    except (CoreFinalFrameworkError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        f"created {output} status={framework.status} "
        f"blockers={len(framework.blockers)} model_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
