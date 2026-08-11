"""Build immutable opt800 folds and the discovery-only Creator v2 index.

The command is deliberately offline.  It projects only ``opt_pool`` metadata
from the caller-bound materialised query artifact, verifies the immutable r2
pre-generation publication, and publishes one create-only directory.  It never
uses query text, validation/test rows, a model, or a network service.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from dataclasses import dataclass
from typing import Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pydantic import ValidationError  # noqa: E402

from skillchain.synthesis.models import CorpusPlan  # noqa: E402
from skillchain.synthesis.portfolio_core_authoring import (  # noqa: E402
    PromptRecipe,
    PromptRecipeManifest,
    RealismAssignment,
    RealismAssignmentsManifest,
    RealismSidecar,
    ReuseAssignment,
    canonical_final_split_sidecar_bytes,
    canonical_prompt_recipe_bytes,
    canonical_realism_assignments_bytes,
    canonical_reuse_sidecar_bytes,
)
from skillchain.synthesis.portfolio_core_publication import (  # noqa: E402
    PreGenerationManifest,
)
from skillchain.synthesis.portfolio_core_selection import (  # noqa: E402
    CoreR2CreatorSelection,
    CreatorSelectionAudit,
    CreatorSelectionEntry,
    CreatorSelectionManifest,
    CreatorSelectionPolicy,
    canonical_creator_selection_index_bytes,
)
from skillchain.synthesis.portfolio_opt_folds import (  # noqa: E402
    OPT_QUERY_COUNT,
    OptFoldError,
    OptFoldQuery,
    PublishedR2SelectionContext,
    build_opt_creator_selection_v2_from_publication,
    build_opt_fold_plan,
    canonical_opt_fold_mapping_bytes,
    canonical_opt_projection_bytes,
    creator_selection_v2_index_bytes,
    validate_opt_creator_selection_v2_from_publication,
    validate_opt_fold_plan,
    validate_published_r2_selection_context,
)
from skillchain.synthesis.splitting import R2SplitConstraints  # noqa: E402
from skillchain.synthesis.store import (  # noqa: E402
    atomic_create_file,
    atomic_publish_new_directory,
    canonical_json_bytes,
    new_staging_directory,
    sha256_bytes,
)
from skillchain.tools.serialization import (  # noqa: E402
    parse_canonical_jsonl,
    read_stable_regular_file,
)


_MAX_QUERY_BYTES = 64 * 1024 * 1024
_MAX_PUBLICATION_FILE_BYTES = 64 * 1024 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
_PUBLICATION_PATHS = {
    "audit/audit.json",
    "audit/index.jsonl",
    "audit/manifest.json",
    "authoring/prompt-recipes-manifest.json",
    "authoring/prompt-recipes.jsonl",
    "authoring/realism-manifest.json",
    "authoring/realism.jsonl",
    "plan/core.json",
    "sidecars/final-splits.jsonl",
    "sidecars/reuse.jsonl",
    "sidecars/split-constraints.json",
    "sidecars/val-interactions.jsonl",
    "stage1/audit.json",
    "stage1/index.jsonl",
    "stage1/manifest.json",
}
_SELECTION_REQUIRED_PATHS = _PUBLICATION_PATHS - {
    "audit/audit.json",
    "audit/index.jsonl",
    "audit/manifest.json",
    "sidecars/val-interactions.jsonl",
}


@dataclass(frozen=True)
class LoadedPublishedSelection:
    context: PublishedR2SelectionContext
    legacy_selection: CoreR2CreatorSelection
    publication_manifest: PreGenerationManifest
    publication_manifest_file_sha256: str


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _read_exact(path: Path, expected_sha256: str, label: str, *, limit: int) -> bytes:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA-256")
    content = read_stable_regular_file(path, label=label, max_bytes=limit)
    if sha256_bytes(content) != expected:
        raise ValueError(f"{label} does not match the caller-supplied SHA-256")
    return content


def load_opt_projection(
    path: Path, *, expected_file_sha256: str
) -> tuple[OptFoldQuery, ...]:
    """Read a bound Core query artifact but project metadata for opt_pool only."""

    content = _read_exact(
        path,
        expected_file_sha256,
        "materialised Core query artifact",
        limit=_MAX_QUERY_BYTES,
    )
    rows = parse_canonical_jsonl(content, label="materialised Core query artifact")
    projected: list[OptFoldQuery] = []
    for line_number, value in enumerate(rows, start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"Core query row {line_number} must be an object")
        # Do not validate, retain, or project text/turns for any non-opt row.
        if value.get("split") != "opt_pool":
            continue
        try:
            projected.append(
                OptFoldQuery(
                    query_id=value.get("query_id"),
                    leakage_group_id=value.get("leakage_group_id"),
                    boundary_group_id=value.get("boundary_group_id"),
                    template_family=value.get("template_family"),
                    generator_batch_id=value.get("generator_batch_id"),
                    canonical_capability=value.get("canonical_capability"),
                    is_boundary=value.get("is_boundary"),
                )
            )
        except ValidationError as error:
            raise ValueError(
                f"opt_pool metadata row {line_number} is invalid"
            ) from error
    if len(projected) != OPT_QUERY_COUNT:
        raise ValueError(
            f"materialised Core artifact must contain exactly {OPT_QUERY_COUNT} "
            "opt_pool rows"
        )
    ordered = tuple(sorted(projected, key=lambda item: item.query_id))
    if len({item.query_id for item in ordered}) != len(ordered):
        raise ValueError("opt_pool projection contains duplicate query IDs")
    return ordered


def _plan_opt_projection(
    context: PublishedR2SelectionContext,
) -> tuple[OptFoldQuery, ...]:
    projected = tuple(
        OptFoldQuery(
            query_id=row.plan_id,
            leakage_group_id=row.leakage_group_id,
            boundary_group_id=row.boundary_group_id,
            template_family=row.template_family,
            generator_batch_id=row.generator_batch_id,
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
        )
        for row in context.plan.queries
        if context.final_split_by_plan_id[row.plan_id] == "opt_pool"
    )
    return tuple(sorted(projected, key=lambda item: item.query_id))


def _load_published_files(
    root: Path, *, expected_manifest_file_sha256: str
) -> tuple[PreGenerationManifest, dict[str, bytes], str]:
    root = Path(root)
    metadata = root.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ValueError("r2 publication root must be a non-reparse real directory")
    manifest_bytes = _read_exact(
        root / "pre-generation-manifest.json",
        expected_manifest_file_sha256,
        "r2 pre-generation root manifest",
        limit=1 * 1024 * 1024,
    )
    try:
        manifest = PreGenerationManifest.model_validate_json(
            manifest_bytes, strict=True
        )
    except ValidationError as error:
        raise ValueError("r2 pre-generation root manifest is invalid") from error
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("r2 pre-generation root manifest is not canonical JSON")
    entries = {item.relative_path: item for item in manifest.files}
    if set(entries) != _PUBLICATION_PATHS:
        raise ValueError("r2 publication manifest path inventory drifted")
    files: dict[str, bytes] = {}
    for relative_path in sorted(_SELECTION_REQUIRED_PATHS):
        entry = entries[relative_path]
        if entry.bytes > _MAX_PUBLICATION_FILE_BYTES:
            raise ValueError(f"published file exceeds offline limit: {relative_path}")
        content = read_stable_regular_file(
            root / Path(relative_path),
            label=f"published {relative_path}",
            max_bytes=max(entry.bytes, 1),
        )
        if len(content) != entry.bytes or sha256_bytes(content) != entry.sha256:
            raise ValueError(f"published file binding mismatch: {relative_path}")
        files[relative_path] = content
    return manifest, files, sha256_bytes(manifest_bytes)


def _load_final_splits(content: bytes) -> dict[str, str]:
    rows = parse_canonical_jsonl(content, label="published final splits")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"plan_id", "final_split"}:
            raise ValueError("published final-split row is invalid")
        plan_id = row.get("plan_id")
        split = row.get("final_split")
        if not isinstance(plan_id, str) or not isinstance(split, str) or plan_id in result:
            raise ValueError("published final-split row is duplicated or malformed")
        result[plan_id] = split
    if canonical_final_split_sidecar_bytes(result) != content:
        raise ValueError("published final-split sidecar is not canonical")
    return result


def _load_reuse(content: bytes) -> dict[str, ReuseAssignment]:
    rows = parse_canonical_jsonl(content, label="published reuse sidecar")
    try:
        assignments = tuple(
            ReuseAssignment.model_validate(row, strict=True) for row in rows
        )
    except ValidationError as error:
        raise ValueError("published reuse sidecar is invalid") from error
    result = {item.plan_id: item for item in assignments}
    if len(result) != len(assignments):
        raise ValueError("published reuse sidecar contains duplicate plan IDs")
    if canonical_reuse_sidecar_bytes(result) != content:
        raise ValueError("published reuse sidecar is not canonical")
    return result


def _load_realism(files: Mapping[str, bytes]) -> RealismSidecar:
    try:
        assignment_rows = parse_canonical_jsonl(
            files["authoring/realism.jsonl"], label="published realism assignments"
        )
        assignments = tuple(
            RealismAssignment.model_validate(row, strict=True)
            for row in assignment_rows
        )
        manifest = RealismAssignmentsManifest.model_validate_json(
            files["authoring/realism-manifest.json"], strict=True
        )
        recipe_rows = parse_canonical_jsonl(
            files["authoring/prompt-recipes.jsonl"],
            label="published prompt recipes",
        )
        recipes = tuple(
            PromptRecipe.model_validate_json(canonical_json_bytes(row), strict=True)
            for row in recipe_rows
        )
        recipe_manifest = PromptRecipeManifest.model_validate_json(
            files["authoring/prompt-recipes-manifest.json"], strict=True
        )
    except ValidationError as error:
        raise ValueError("published realism or recipe payload is invalid") from error
    if canonical_realism_assignments_bytes(assignments) != files[
        "authoring/realism.jsonl"
    ]:
        raise ValueError("published realism assignments are not canonical")
    if canonical_json_bytes(manifest) != files["authoring/realism-manifest.json"]:
        raise ValueError("published realism manifest is not canonical")
    if canonical_prompt_recipe_bytes(recipes) != files[
        "authoring/prompt-recipes.jsonl"
    ]:
        raise ValueError("published prompt recipes are not canonical")
    if canonical_json_bytes(recipe_manifest) != files[
        "authoring/prompt-recipes-manifest.json"
    ]:
        raise ValueError("published prompt recipe manifest is not canonical")
    if (
        manifest.assignments_sha256
        != sha256_bytes(files["authoring/realism.jsonl"])
        or recipe_manifest.recipes_sha256
        != sha256_bytes(files["authoring/prompt-recipes.jsonl"])
        or manifest.prompt_recipe_manifest_sha256
        != sha256_bytes(files["authoring/prompt-recipes-manifest.json"])
    ):
        raise ValueError("published realism/recipe digest binding drifted")
    return RealismSidecar(
        assignments=assignments,
        manifest=manifest,
        recipes=recipes,
        recipe_manifest=recipe_manifest,
    )


def _load_legacy_selection(
    files: Mapping[str, bytes], context: PublishedR2SelectionContext
) -> CoreR2CreatorSelection:
    try:
        rows = parse_canonical_jsonl(
            files["stage1/index.jsonl"], label="published legacy Creator index"
        )
        entries = tuple(
            CreatorSelectionEntry.model_validate(row, strict=True) for row in rows
        )
        manifest = CreatorSelectionManifest.model_validate_json(
            files["stage1/manifest.json"], strict=True
        )
        audit = CreatorSelectionAudit.model_validate_json(
            files["stage1/audit.json"], strict=True
        )
    except ValidationError as error:
        raise ValueError("published legacy Creator selection is invalid") from error
    policy = CreatorSelectionPolicy()
    if canonical_creator_selection_index_bytes(entries) != files[
        "stage1/index.jsonl"
    ]:
        raise ValueError("published legacy Creator index is not canonical")
    if canonical_json_bytes(manifest) != files["stage1/manifest.json"]:
        raise ValueError("published legacy Creator manifest is not canonical")
    if canonical_json_bytes(audit) != files["stage1/audit.json"]:
        raise ValueError("published legacy Creator audit is not canonical")
    realism = context.realism_sidecar
    expected = {
        "policy_sha256": sha256_bytes(canonical_json_bytes(policy)),
        "plan_sha256": context.trusted_plan_sha256,
        "asset_catalog_sha256": context.plan.asset_catalog_sha256,
        "capability_assignments_sha256": (
            context.plan.capability_assignments_sha256
        ),
        "realism_manifest_sha256": sha256_bytes(
            canonical_json_bytes(realism.manifest)
        ),
        "realism_assignments_sha256": realism.manifest.assignments_sha256,
        "prompt_recipe_manifest_sha256": (
            realism.manifest.prompt_recipe_manifest_sha256
        ),
        "final_split_sidecar_sha256": (
            realism.manifest.final_split_sidecar_sha256
        ),
        "reuse_sidecar_sha256": realism.manifest.reuse_sidecar_sha256,
        "selection_bytes_sha256": sha256_bytes(files["stage1/index.jsonl"]),
    }
    if any(getattr(manifest, field_name) != value for field_name, value in expected.items()):
        raise ValueError("published legacy Creator manifest binding drifted")
    if tuple(entry.selection_rank for entry in entries) != tuple(range(1, 241)):
        raise ValueError("published legacy Creator ranks must be exactly 1..240")
    if len({entry.plan_id for entry in entries}) != len(entries):
        raise ValueError("published legacy Creator index contains duplicate IDs")
    return CoreR2CreatorSelection(
        entries=entries,
        manifest=manifest,
        audit=audit,
        policy=policy,
    )


def load_published_r2_selection_context(
    root: Path, *, expected_manifest_file_sha256: str
) -> LoadedPublishedSelection:
    """Load frozen selection evidence without consulting current assignments."""

    manifest, files, manifest_file_sha256 = _load_published_files(
        root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    try:
        plan = CorpusPlan.model_validate_json(files["plan/core.json"], strict=True)
        constraints = R2SplitConstraints.model_validate_json(
            files["sidecars/split-constraints.json"], strict=True
        )
    except ValidationError as error:
        raise ValueError("published r2 plan or split constraints are invalid") from error
    if canonical_json_bytes(plan) != files["plan/core.json"]:
        raise ValueError("published r2 plan is not canonical JSON")
    if canonical_json_bytes(constraints) != files["sidecars/split-constraints.json"]:
        raise ValueError("published split constraints are not canonical JSON")
    plan_sha256 = sha256_bytes(files["plan/core.json"])
    if manifest.r2_plan_sha256 != plan_sha256:
        raise ValueError("root manifest r2 plan binding drifted")
    final_splits = _load_final_splits(files["sidecars/final-splits.jsonl"])
    reuse = _load_reuse(files["sidecars/reuse.jsonl"])
    realism = _load_realism(files)
    context = PublishedR2SelectionContext(
        plan=plan,
        final_split_by_plan_id=final_splits,
        reuse_by_plan_id=reuse,
        split_constraints=constraints,
        realism_sidecar=realism,
        trusted_plan_sha256=plan_sha256,
    )
    validate_published_r2_selection_context(context)
    legacy = _load_legacy_selection(files, context)
    return LoadedPublishedSelection(
        context=context,
        legacy_selection=legacy,
        publication_manifest=manifest,
        publication_manifest_file_sha256=manifest_file_sha256,
    )


def _bundle_files(arguments: argparse.Namespace) -> dict[str, bytes]:
    source_sha256 = _require_sha256(
        arguments.expected_queries_sha256,
        "expected materialised query artifact SHA-256",
    )
    projection = load_opt_projection(
        arguments.queries,
        expected_file_sha256=source_sha256,
    )
    published = load_published_r2_selection_context(
        arguments.r2_publication_root,
        expected_manifest_file_sha256=(
            arguments.expected_r2_publication_manifest_sha256
        ),
    )
    context = published.context
    legacy_ids = tuple(entry.plan_id for entry in published.legacy_selection.entries)
    legacy_sha256 = published.legacy_selection.manifest.selection_bytes_sha256
    plan_projection = _plan_opt_projection(context)
    if canonical_opt_projection_bytes(projection) != canonical_opt_projection_bytes(
        plan_projection
    ):
        raise ValueError(
            "materialised opt_pool metadata does not match the rebuilt r2 plan"
        )
    folds = build_opt_fold_plan(
        projection,
        legacy_selected_query_ids=legacy_ids,
        seed=arguments.seed,
        fold_count=4,
        source_queries_sha256=source_sha256,
        trusted_plan_sha256=context.trusted_plan_sha256,
        legacy_selection_bytes_sha256=legacy_sha256,
    )
    validate_opt_fold_plan(
        folds,
        projection,
        legacy_selected_query_ids=legacy_ids,
        source_queries_sha256=source_sha256,
        trusted_plan_sha256=context.trusted_plan_sha256,
        legacy_selection_bytes_sha256=legacy_sha256,
    )
    selection = build_opt_creator_selection_v2_from_publication(
        context,
        folds,
        seed=arguments.seed,
    )
    validate_opt_creator_selection_v2_from_publication(
        selection,
        context,
        folds,
    )

    files = {
        "opt-projection.jsonl": canonical_opt_projection_bytes(projection),
        "fold-mapping.jsonl": canonical_opt_fold_mapping_bytes(folds.assignments),
        "fold-manifest.json": canonical_json_bytes(folds.manifest),
        "fold-audit.json": canonical_json_bytes(folds.audit),
        "fold-policy.json": canonical_json_bytes(folds.policy),
        "creator-v2-index.jsonl": creator_selection_v2_index_bytes(selection),
        "creator-v2-manifest.json": canonical_json_bytes(selection.manifest),
        "creator-v2-audit.json": canonical_json_bytes(selection.audit),
        "creator-v2-policy.json": canonical_json_bytes(selection.policy),
    }
    bundle_manifest = {
        "schema_version": 1,
        "artifact_kind": "portfolio-core-opt800-folds-and-creator-v2",
        "seed": arguments.seed,
        "source_queries_sha256": source_sha256,
        "plan_sha256": context.trusted_plan_sha256,
        "realism_assignments_sha256": (
            context.realism_sidecar.manifest.assignments_sha256
        ),
        "r2_publication_manifest_file_sha256": (
            published.publication_manifest_file_sha256
        ),
        "superseded_legacy_selection_sha256": legacy_sha256,
        "fold_mapping_sha256": folds.manifest.mapping_sha256,
        "creator_selection_sha256": selection.manifest.selection_bytes_sha256,
        "files": [
            {
                "relative_path": name,
                "bytes": len(content),
                "sha256": sha256_bytes(content),
            }
            for name, content in sorted(files.items())
        ],
    }
    files["bundle-manifest.json"] = canonical_json_bytes(bundle_manifest)
    return files


def publish_create_only(output_directory: Path, files: Mapping[str, bytes]) -> Path:
    """Atomically publish a complete file set without replacing any path."""

    destination = Path(output_directory)
    staging = new_staging_directory(destination)
    try:
        for relative_path, content in sorted(files.items()):
            relative = Path(relative_path)
            if (
                relative.is_absolute()
                or relative.name != relative_path
                or relative_path in {"", ".", ".."}
            ):
                raise ValueError(f"invalid bundle file name: {relative_path!r}")
            atomic_create_file(staging / relative, content)
        return atomic_publish_new_directory(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-publication-root", type=Path, required=True)
    parser.add_argument(
        "--expected-r2-publication-manifest-sha256",
        required=True,
        help="Caller-trusted file SHA-256 of pre-generation-manifest.json.",
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--expected-queries-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(20260808,), default=20260808)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        # Reject cheaply before rebuilding, while the publisher still protects
        # against a writer racing between this check and the final rename.
        if os.path.lexists(arguments.output_dir):
            raise FileExistsError(
                f"output directory already exists; refusing overwrite: "
                f"{arguments.output_dir}"
            )
        files = _bundle_files(arguments)
        published = publish_create_only(arguments.output_dir, files)
    except (
        FileExistsError,
        OSError,
        OptFoldError,
        TypeError,
        ValueError,
    ) as error:
        print(f"build-portfolio-opt-folds: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "creator_selection_sha256": sha256_bytes(
                    files["creator-v2-index.jsonl"]
                ),
                "fold_manifest_file_sha256": sha256_bytes(
                    files["fold-manifest.json"]
                ),
                "fold_mapping_sha256": sha256_bytes(files["fold-mapping.jsonl"]),
                "output_dir": str(published.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
