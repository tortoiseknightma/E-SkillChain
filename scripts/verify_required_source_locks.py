"""Independently re-hash RAW scopes committed by required-source locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.source_lock import (  # noqa: E402
    load_and_verify_required_source_lock,
)
from skillchain.tools.serialization import (  # noqa: E402
    parse_canonical_json,
    sha256_bytes,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--lock-root",
        type=Path,
        default=REPOSITORY_ROOT / "specs/data_sources/c2/source-locks",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Verify only this source ID; repeat for multiple sources.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    manifest_path = arguments.lock_root / "required-source-lock-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = parse_canonical_json(
        manifest_bytes, label="required source lock manifest"
    )
    rows = manifest["sources"]
    known = {row["source_id"] for row in rows}
    requested = set(arguments.source)
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError("unknown required source: " + ", ".join(unknown))
    selected = [
        row for row in rows if not requested or row["source_id"] in requested
    ]
    results = []
    for row in selected:
        verified = load_and_verify_required_source_lock(
            arguments.lock_root / row["lock_path"],
            arguments.raw_root,
            expected_lock_file_sha256=row["source_lock_sha256"],
        )
        results.append(
            {
                "file_count": sum(
                    scope.file_count for scope in verified.lock.artifact_scopes
                ),
                "source_id": verified.lock.source_id,
                "source_lock_sha256": verified.lock_file_sha256,
                "status": "verified",
                "total_bytes": sum(
                    scope.total_bytes for scope in verified.lock.artifact_scopes
                ),
            }
        )
    print(
        json.dumps(
            {
                "lock_manifest_sha256": sha256_bytes(manifest_bytes),
                "results": results,
                "source_count": len(results),
                "status": "verified",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
