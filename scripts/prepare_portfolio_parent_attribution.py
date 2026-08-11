"""Build a current-parent-bound S2/S3 attribution packet from a real smoke run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pydantic import ValidationError  # noqa: E402

from skillchain.evaluation.portfolio_attribution import (  # noqa: E402
    PortfolioAttributionError,
    build_portfolio_parent_attribution,
)
from skillchain.static_authoring import StaticBankArtifact  # noqa: E402
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 64 * 1024 * 1024


def _external_file(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> bytes:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise PortfolioAttributionError(f"{label} expected SHA-256 is invalid")
    try:
        content = read_stable_regular_file(
            path,
            label=label,
            max_bytes=_MAX_BYTES,
        )
    except (ArtifactFormatError, OSError) as error:
        raise PortfolioAttributionError(f"{label} cannot be read") from error
    if sha256_bytes(content) != expected_sha256:
        raise PortfolioAttributionError(f"{label} file digest mismatch")
    return content


def _load_parent_bank(path: Path, expected_sha256: str) -> StaticBankArtifact:
    content = _external_file(
        path,
        expected_sha256,
        label="current parent Bank",
    )
    try:
        raw = parse_canonical_json(content, label="current parent Bank")
        if not isinstance(raw, dict):
            raise ArtifactFormatError("current parent Bank must be an object")
        bank = StaticBankArtifact.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise PortfolioAttributionError(
            "current parent Bank is not canonical and valid"
        ) from error
    if bank.canonical_bytes() != content:
        raise PortfolioAttributionError("current parent Bank bytes are not canonical")
    return bank


def _query_ids_from_artifact(
    path: Path,
    expected_sha256: str,
) -> tuple[str, ...]:
    content = _external_file(
        path,
        expected_sha256,
        label="optimization query-ID artifact",
    )
    try:
        raw = parse_canonical_json(content, label="optimization query-ID artifact")
    except ArtifactFormatError as error:
        raise PortfolioAttributionError(
            "optimization query-ID artifact is not canonical"
        ) from error

    values: object
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict):
        if "optimization_query_ids" in raw:
            values = raw["optimization_query_ids"]
        elif "query_ids" in raw:
            values = raw["query_ids"]
        elif isinstance(raw.get("manifest"), dict):
            values = raw["manifest"].get("query_ids")
        else:
            values = None
    else:
        values = None
    if not isinstance(values, list) or any(
        not isinstance(item, str) for item in values
    ):
        raise PortfolioAttributionError(
            "optimization query-ID artifact lacks one string query-ID list"
        )
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("s2_route_optimizer", "s3_body_refiner"),
        required=True,
    )
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--parent-bank", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-bank-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--optimization-query-ids-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-optimization-query-ids-file-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output = arguments.output.absolute()
    if os.path.lexists(output):
        print(
            "prepare-portfolio-parent-attribution: output already exists",
            file=sys.stderr,
        )
        return 2
    try:
        parent_bank = _load_parent_bank(
            arguments.parent_bank.absolute(),
            arguments.expected_parent_bank_file_sha256,
        )
        optimization_query_ids = _query_ids_from_artifact(
            arguments.optimization_query_ids_file.absolute(),
            arguments.expected_optimization_query_ids_file_sha256,
        )
        packet = build_portfolio_parent_attribution(
            stage=arguments.stage,
            smoke_root=arguments.smoke_root,
            parent_bank=parent_bank,
            optimization_query_ids=optimization_query_ids,
        )
        content = packet.canonical_bytes()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_create_file(output, content)
    except (PortfolioAttributionError, OSError) as error:
        print(f"prepare-portfolio-parent-attribution: {error}", file=sys.stderr)
        return 2

    print(
        canonical_json_bytes(
            {
                "output": output.as_posix(),
                "output_file_sha256": sha256_bytes(content),
                "packet_sha256": packet.packet_sha256,
                "source_bank_sha256": packet.source_bank_sha256,
                "source_config": packet.source_config,
                "stage": packet.stage,
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
