"""Train and publish the offline, text-only Portfolio Hybrid Router.

Only materialised ``opt_pool`` queries are projected into model examples.  The
command consumes a previously published group-safe fold mapping, performs four
fold OOF evaluation, fits one opt800 candidate, and publishes an immutable
offline-only bundle.  It neither calls a model service nor integrates the
candidate into the assistant runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Annotated, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pydantic import (  # noqa: E402
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
)

from skillchain.evolution.hybrid_router import (  # noqa: E402
    CAPABILITY_ORDER,
    DEFAULT_RANDOM_SEED,
    HybridRouterError,
    fit_hybrid_router,
    make_hybrid_route_example,
    publish_hybrid_router_bundle,
    run_grouped_oof,
)
from skillchain.schemas import Query  # noqa: E402
from skillchain.synthesis.portfolio_opt_folds import (  # noqa: E402
    OPT_BATCHES_PER_FOLD,
    OPT_BATCH_SIZE,
    OPT_FOLD_IDS,
    OPT_FOLD_SIZE,
    OPT_QUERY_COUNT,
    OptFoldAssignment,
    OptFoldManifest,
    OptFoldQuery,
    canonical_opt_fold_mapping_bytes,
    canonical_opt_projection_bytes,
)
from skillchain.synthesis.splitting import GROUP_FIELDS  # noqa: E402
from skillchain.synthesis.store import (  # noqa: E402
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import (  # noqa: E402
    parse_canonical_jsonl,
    read_stable_regular_file,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_MAX_QUERY_BYTES = 64 * 1024 * 1024
_MAX_FOLD_BYTES = 4 * 1024 * 1024
_MAX_BASELINE_BYTES = 8 * 1024 * 1024


class _BaselineRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    query_id: str
    selected_capability: str | None

    @field_validator("schema_version")
    @classmethod
    def _schema_is_one(cls, value: int) -> int:
        if value != 1:
            raise ValueError("baseline route schema_version must be 1")
        return value

    @field_validator("query_id")
    @classmethod
    def _query_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("baseline route query_id must be non-blank and trimmed")
        return value

    @field_validator("selected_capability")
    @classmethod
    def _capability(cls, value: str | None) -> str | None:
        if value is not None and value not in CAPABILITY_ORDER:
            raise ValueError("baseline selected_capability is outside the frozen enum")
        return value


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


def load_opt_queries(
    path: Path, *, expected_file_sha256: str
) -> tuple[Query, ...]:
    """Typed-load only opt_pool rows from one externally bound Core artifact."""

    content = _read_exact(
        path,
        expected_file_sha256,
        "materialised Core query artifact",
        limit=_MAX_QUERY_BYTES,
    )
    rows = parse_canonical_jsonl(content, label="materialised Core query artifact")
    queries: list[Query] = []
    for line_number, value in enumerate(rows, start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"Core query row {line_number} must be an object")
        # split is the only field inspected on non-opt rows.  Their turns and
        # labels never cross the feature-projection boundary.
        if value.get("split") != "opt_pool":
            continue
        try:
            query = Query.model_validate(value, strict=True)
        except ValidationError as error:
            raise ValueError(f"opt_pool query row {line_number} is invalid") from error
        if query.canonical_capability not in CAPABILITY_ORDER:
            raise ValueError(
                f"opt_pool query {query.query_id} has an unresolved/unknown capability"
            )
        queries.append(query)
    if len(queries) != OPT_QUERY_COUNT:
        raise ValueError(
            f"materialised Core artifact must contain exactly {OPT_QUERY_COUNT} "
            "opt_pool rows"
        )
    ordered = tuple(sorted(queries, key=lambda item: item.query_id))
    if len({item.query_id for item in ordered}) != len(ordered):
        raise ValueError("opt_pool query artifact contains duplicate query IDs")
    return ordered


def _ids_sha256(values: list[str]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(values)))


def _query_fold_projection(queries: tuple[Query, ...]) -> tuple[OptFoldQuery, ...]:
    return tuple(
        OptFoldQuery(
            query_id=query.query_id,
            leakage_group_id=query.leakage_group_id,
            boundary_group_id=query.boundary_group_id,
            template_family=query.template_family,
            generator_batch_id=query.generator_batch_id,
            canonical_capability=query.canonical_capability,
            is_boundary=query.is_boundary,
        )
        for query in queries
    )


def load_verified_fold_mapping(
    fold_directory: Path,
    *,
    expected_manifest_sha256: str,
    expected_mapping_sha256: str,
    source_queries_sha256: str,
    queries: tuple[Query, ...],
) -> tuple[OptFoldManifest, tuple[OptFoldAssignment, ...]]:
    manifest_bytes = _read_exact(
        fold_directory / "fold-manifest.json",
        expected_manifest_sha256,
        "fold manifest",
        limit=_MAX_FOLD_BYTES,
    )
    mapping_bytes = _read_exact(
        fold_directory / "fold-mapping.jsonl",
        expected_mapping_sha256,
        "fold mapping",
        limit=_MAX_FOLD_BYTES,
    )
    try:
        manifest = OptFoldManifest.model_validate_json(manifest_bytes, strict=True)
        raw_assignments = parse_canonical_jsonl(mapping_bytes, label="fold mapping")
        assignments = tuple(
            OptFoldAssignment.model_validate(value, strict=True)
            for value in raw_assignments
        )
    except (ValidationError, ValueError) as error:
        raise ValueError("fold artifacts are invalid") from error
    if canonical_opt_fold_mapping_bytes(assignments) != mapping_bytes:
        raise ValueError("fold mapping is not canonical and query-sorted")
    if manifest.mapping_sha256 != sha256_bytes(mapping_bytes):
        raise ValueError("fold manifest does not bind the supplied mapping")
    if manifest.source_queries_sha256 != source_queries_sha256:
        raise ValueError("fold manifest is bound to a different query artifact")

    projections = _query_fold_projection(queries)
    if manifest.opt_projection_sha256 != sha256_bytes(
        canonical_opt_projection_bytes(projections)
    ):
        raise ValueError("fold manifest is bound to different opt_pool metadata")
    query_by_id = {query.query_id: query for query in queries}
    assignment_by_id = {item.query_id: item for item in assignments}
    if len(assignments) != OPT_QUERY_COUNT or set(assignment_by_id) != set(query_by_id):
        raise ValueError("fold mapping must cover every opt_pool query exactly once")

    fold_counts = Counter(item.fold_id for item in assignments)
    batches_by_fold: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        query = query_by_id[assignment.query_id]
        if assignment.atomic_batch_id != query.generator_batch_id:
            raise ValueError("fold mapping atomic batch differs from query metadata")
        batches_by_fold[assignment.fold_id].add(assignment.atomic_batch_id)
    if any(fold_counts[fold] != OPT_FOLD_SIZE for fold in OPT_FOLD_IDS):
        raise ValueError("fold mapping must contain four 200-query folds")
    if any(
        len(batches_by_fold[fold]) != OPT_BATCHES_PER_FOLD for fold in OPT_FOLD_IDS
    ):
        raise ValueError("each fold must contain exactly eight atomic batches")
    if any(
        sum(
            1
            for item in assignments
            if item.atomic_batch_id == batch_id
        )
        != OPT_BATCH_SIZE
        for batch_id in {item.atomic_batch_id for item in assignments}
    ):
        raise ValueError("each mapped atomic batch must contain exactly 25 queries")

    fold_by_query = {item.query_id: item.fold_id for item in assignments}
    for field_name in GROUP_FIELDS:
        folds_by_value: dict[str, set[str]] = defaultdict(set)
        for query in queries:
            value = getattr(query, field_name)
            if value is not None:
                folds_by_value[value].add(fold_by_query[query.query_id])
        if any(len(folds) != 1 for folds in folds_by_value.values()):
            raise ValueError(f"fold mapping splits GROUP_FIELDS field {field_name}")
    discovery_ids = [item.query_id for item in assignments if item.role == "discovery"]
    replay_ids = [item.query_id for item in assignments if item.role == "replay"]
    if manifest.discovery_query_ids_sha256 != _ids_sha256(discovery_ids):
        raise ValueError("fold manifest discovery ID binding drifted")
    if manifest.replay_query_ids_sha256 != _ids_sha256(replay_ids):
        raise ValueError("fold manifest replay ID binding drifted")
    return manifest, assignments


def load_baseline_routes(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_query_ids: set[str],
) -> dict[str, str | None]:
    content = _read_exact(
        path,
        expected_file_sha256,
        "baseline route artifact",
        limit=_MAX_BASELINE_BYTES,
    )
    rows = parse_canonical_jsonl(content, label="baseline route artifact")
    try:
        routes = tuple(_BaselineRoute.model_validate(value, strict=True) for value in rows)
    except ValidationError as error:
        raise ValueError("baseline route artifact is invalid") from error
    if canonical_jsonl_bytes(routes) != content:
        raise ValueError("baseline route artifact must be canonical JSONL")
    by_id = {route.query_id: route.selected_capability for route in routes}
    if len(by_id) != len(routes) or set(by_id) != expected_query_ids:
        raise ValueError("baseline routes must cover opt800 exactly once")
    return by_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--expected-queries-sha256", required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--expected-fold-manifest-sha256", required=True)
    parser.add_argument("--expected-fold-mapping-sha256", required=True)
    parser.add_argument("--baseline-routes", type=Path)
    parser.add_argument("--expected-baseline-routes-sha256")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if os.path.lexists(arguments.output_dir):
            raise FileExistsError(
                f"output directory already exists; refusing overwrite: "
                f"{arguments.output_dir}"
            )
        if (arguments.baseline_routes is None) != (
            arguments.expected_baseline_routes_sha256 is None
        ):
            raise ValueError(
                "--baseline-routes and --expected-baseline-routes-sha256 must "
                "be supplied together"
            )
        source_sha256 = _require_sha256(
            arguments.expected_queries_sha256,
            "expected materialised query artifact SHA-256",
        )
        if arguments.seed != DEFAULT_RANDOM_SEED:
            raise ValueError(
                f"offline router seed is frozen at {DEFAULT_RANDOM_SEED}"
            )
        queries = load_opt_queries(
            arguments.queries,
            expected_file_sha256=source_sha256,
        )
        manifest, assignments = load_verified_fold_mapping(
            arguments.fold_dir,
            expected_manifest_sha256=arguments.expected_fold_manifest_sha256,
            expected_mapping_sha256=arguments.expected_fold_mapping_sha256,
            source_queries_sha256=source_sha256,
            queries=queries,
        )
        if manifest.seed != arguments.seed:
            raise ValueError("router seed differs from the frozen fold manifest")
        fold_by_id = {item.query_id: item.fold_id for item in assignments}
        baseline_by_id: dict[str, str | None]
        if arguments.baseline_routes is None:
            baseline_by_id = {query.query_id: None for query in queries}
        else:
            baseline_by_id = load_baseline_routes(
                arguments.baseline_routes,
                expected_file_sha256=arguments.expected_baseline_routes_sha256,
                expected_query_ids={query.query_id for query in queries},
            )
        examples = tuple(
            make_hybrid_route_example(
                query,
                fold_id=fold_by_id[query.query_id],
                baseline_capability=baseline_by_id[query.query_id],
                baseline_observed=arguments.baseline_routes is not None,
            )
            for query in queries
        )
        oof = run_grouped_oof(examples, random_seed=arguments.seed)
        pipeline = fit_hybrid_router(examples, random_seed=arguments.seed)
        router_manifest = publish_hybrid_router_bundle(
            arguments.output_dir,
            pipeline=pipeline,
            oof_result=oof,
            source_queries_sha256=source_sha256,
            fold_plan_sha256=manifest.mapping_sha256,
        )
        manifest_file_sha256 = sha256_bytes(
            read_stable_regular_file(
                arguments.output_dir / "model-manifest.json",
                label="published router manifest",
                max_bytes=128_000,
            )
        )
    except (
        FileExistsError,
        HybridRouterError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"build-portfolio-hybrid-router: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "authorization_status": router_manifest.authorization_status,
                "macro_f1": oof.metrics.macro_f1,
                "manifest_file_sha256": manifest_file_sha256,
                "output_dir": str(arguments.output_dir.resolve()),
                "query_accuracy": oof.metrics.query_accuracy,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
