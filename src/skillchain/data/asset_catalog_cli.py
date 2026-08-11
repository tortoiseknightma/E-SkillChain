"""Command-line build and verification for immutable asset catalogs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from skillchain.data.asset_catalog import (
    AssetCatalog,
    DatasetAssetDraft,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.data.gallery_eligibility import (
    GalleryEligibilityViolation,
    build_gallery_eligibility_manifest,
    verify_gallery_eligibility,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
)


class DraftJsonlError(ValueError):
    """The adapter-owned draft inventory is not strict, complete JSONL."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DraftJsonlError(f"JSON object contains duplicate key: {key}")
        value[key] = item
    return value


def load_drafts(path: str | Path) -> tuple[DatasetAssetDraft, ...]:
    """Load strict UTF-8 JSONL and reject blanks, duplicate keys and rows."""

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DraftJsonlError("drafts must be UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise DraftJsonlError("drafts JSONL must contain at least one row")

    drafts: list[DatasetAssetDraft] = []
    seen_rows: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DraftJsonlError(f"drafts line {line_number} is blank")
        try:
            raw = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
        except (json.JSONDecodeError, DraftJsonlError) as exc:
            raise DraftJsonlError(
                f"drafts line {line_number} is invalid JSON: {exc}"
            ) from exc
        try:
            draft = DatasetAssetDraft.model_validate(raw)
        except ValidationError as exc:
            raise DraftJsonlError(
                f"drafts line {line_number} violates DatasetAssetDraft: {exc}"
            ) from exc
        signature = json.dumps(
            draft.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen_rows:
            raise DraftJsonlError(
                f"drafts line {line_number} duplicates an earlier row"
            )
        seen_rows.add(signature)
        drafts.append(draft)
    return tuple(drafts)


def _summary(command: str, catalog: AssetCatalog) -> dict[str, Any]:
    manifest = catalog.manifest
    return {
        "asset_count": manifest.asset_count,
        "catalog_path": str(catalog.root.resolve()),
        "catalog_policy_version": manifest.catalog_policy_version,
        "catalog_sha256": manifest.catalog_sha256,
        "command": command,
        "component_count": manifest.component_count,
        "coverage_roots": list(manifest.coverage_roots),
        "leakage_policy_version": manifest.leakage_policy_version,
        "near_duplicate_cluster_count": manifest.near_duplicate_cluster_count,
        "near_duplicate_policy": manifest.near_duplicate_policy.model_dump(mode="json"),
        "status": "ok",
        "verified_files": True,
    }


def _emit(payload: dict[str, Any], *, stream) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=stream,
    )


def _run_build(args: argparse.Namespace) -> dict[str, Any]:
    if not args.coverage_root:
        raise ValueError("formal catalog build requires at least one --coverage-root")
    drafts = load_drafts(args.drafts)
    asset_root = Path(args.asset_root)
    assets = tuple(inventory_dataset_asset(draft, asset_root) for draft in drafts)
    policy = NearDuplicatePolicy(max_phash_hamming_distance=args.max_phash_distance)
    publish_asset_catalog(
        assets,
        args.output,
        asset_root,
        policy,
        coverage_roots=args.coverage_root,
    )
    catalog = load_asset_catalog(args.output, asset_root, verify_files=True)
    return _summary("build", catalog)


def _run_verify(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_asset_catalog(
        args.catalog,
        args.asset_root,
        verify_files=True,
    )
    return _summary("verify", catalog)


def _run_audit_gallery(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_asset_catalog(
        args.catalog,
        args.asset_root,
        verify_files=True,
    )
    manifest = build_gallery_eligibility_manifest(
        args.queries,
        args.gallery_assets,
        catalog,
    )
    atomic_create_file(args.output, canonical_json_bytes(manifest))
    verified = verify_gallery_eligibility(
        args.output,
        args.queries,
        args.gallery_assets,
        catalog,
    )
    payload = verified.manifest.model_dump(mode="json")
    payload["output"] = str(Path(args.output).resolve())
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-asset-catalog",
        description="Build or verify an immutable DatasetAsset catalog.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="inventory drafts and publish a catalog"
    )
    build.add_argument("--drafts", required=True, help="strict DatasetAssetDraft JSONL")
    build.add_argument("--asset-root", required=True, help="root for draft local_path")
    build.add_argument("--output", required=True, help="create-only catalog directory")
    build.add_argument(
        "--coverage-root",
        action="append",
        default=[],
        help="catalog-relative image tree that must be completely inventoried; repeatable",
    )
    build.add_argument(
        "--max-phash-distance",
        type=int,
        default=4,
        help="maximum 64-bit pHash Hamming distance (0..16)",
    )
    build.set_defaults(handler=_run_build)

    verify = subparsers.add_parser("verify", help="strictly verify a frozen catalog")
    verify.add_argument("--catalog", required=True, help="catalog directory")
    verify.add_argument(
        "--asset-root", required=True, help="root for catalog local_path"
    )
    verify.set_defaults(handler=_run_verify)

    audit = subparsers.add_parser(
        "audit-gallery",
        help="fail closed on query/gallery duplication or invalid retrieval eligibility",
    )
    audit.add_argument("--catalog", required=True, help="catalog directory")
    audit.add_argument(
        "--asset-root", required=True, help="root for catalog local_path"
    )
    audit.add_argument(
        "--queries",
        required=True,
        help="canonical schema-v2 Query JSONL",
    )
    audit.add_argument(
        "--gallery-assets",
        required=True,
        help='canonical JSONL rows shaped as {"asset_id":"...","image_path":"..."}',
    )
    audit.add_argument(
        "--output",
        required=True,
        help="create-only eligibility manifest path",
    )
    audit.set_defaults(handler=_run_audit_gallery)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
    except GalleryEligibilityViolation as exc:
        _emit(exc.payload, stream=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        _emit(
            {
                "command": args.command,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 2
    _emit(payload, stream=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
