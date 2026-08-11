"""Publish or verify a document-safety catalog from a pinned review ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from skillchain.data.asset_catalog import load_asset_catalog
from skillchain.tools.document_safety import (
    DocumentSafetyRecord,
    load_document_safety_catalog,
    publish_document_safety_catalog,
)
from skillchain.tools.serialization import (
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)

_ERROR_MESSAGE = "build-document-safety: failed"
_MAX_LEDGER_BYTES = 64 * 1024 * 1024


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    publish = commands.add_parser(
        "build",
        help="publish a create-only catalog from a canonical review ledger",
    )
    publish.add_argument("--approvals", type=Path, required=True)
    _add_common_arguments(publish)

    status = commands.add_parser(
        "status",
        aliases=("verify",),
        help="verify an existing safety catalog against its external ledger hash",
    )
    _add_common_arguments(status)
    return parser.parse_args(argv)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--review-ledger-sha256", required=True)
    parser.add_argument("--asset-catalog", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _load_review_ledger(
    path: Path,
    expected_sha256: str,
) -> tuple[DocumentSafetyRecord, ...]:
    content = read_stable_regular_file(
        path,
        label="document safety review ledger",
        max_bytes=_MAX_LEDGER_BYTES,
    )
    if sha256_bytes(content) != expected_sha256:
        raise ValueError("review ledger hash mismatch")
    rows = parse_canonical_jsonl(content, label="document safety review ledger")
    return tuple(DocumentSafetyRecord.model_validate(row, strict=True) for row in rows)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        catalog = load_asset_catalog(
            arguments.asset_catalog,
            arguments.asset_root,
            verify_files=True,
        )
        if arguments.command == "build":
            approvals = _load_review_ledger(
                arguments.approvals,
                arguments.review_ledger_sha256,
            )
            verified = publish_document_safety_catalog(
                approvals,
                arguments.output,
                catalog,
                expected_review_ledger_sha256=arguments.review_ledger_sha256,
            )
        else:
            verified = load_document_safety_catalog(
                arguments.output,
                catalog,
                expected_review_ledger_sha256=arguments.review_ledger_sha256,
            )
        print(
            json.dumps(
                verified.manifest.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception:
        print(_ERROR_MESSAGE, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
