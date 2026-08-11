"""Verified, split-aware Portfolio Core inputs for zero-call launch planning.

The Core r3 corpus has a different publication lineage from the historical
``dev_mini`` accepted-batch corpus.  In particular, r3 queries intentionally
have no ``synthesis_batch_id``.  Their immutable 25-query execution unit is the
``generator_batch_id`` inherited from the r2 plan, while the final split comes
from the separately frozen final-splits sidecar.

This module verifies that complete chain without invoking a provider.  The
returned Assistant projection contains only the opaque asset token, text and
turns; catalog paths, source metadata, template families and labels remain in
the private typed query handle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from skillchain.data.asset_catalog import AssetCatalog, load_asset_catalog
from skillchain.data.portfolio_remote_processing import (
    VerifiedPortfolioRemoteProcessingRuntime,
    verify_portfolio_remote_processing_runtime,
)
from skillchain.evaluation.assistant_runs import (
    AssistantQueryInput,
    build_assistant_query_input,
)
from skillchain.evaluation.portfolio_inputs import (
    CORE_PORTFOLIO_PROCESSOR_ORDER,
    ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
    PortfolioQueryAssetBinding,
    PortfolioRemoteProcessingFiles,
)
from skillchain.schemas import Query
from skillchain.synthesis.models import CorpusPlan, PlannedQuery
from skillchain.synthesis.planning import (
    CORE_R2_CAPABILITY_BOUNDARY_COUNTS,
    CORE_R2_CAPABILITY_COUNTS,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)


CoreSplit = Literal["dev_mini", "opt_pool", "val", "test_frozen"]
CORE_SPLIT_ORDER: tuple[CoreSplit, ...] = (
    "dev_mini",
    "opt_pool",
    "val",
    "test_frozen",
)
CORE_SPLIT_COUNTS: dict[CoreSplit, int] = {
    "dev_mini": 200,
    "opt_pool": 800,
    "val": 200,
    "test_frozen": 300,
}
CORE_QUERY_COUNT = 1500
CORE_BATCH_SIZE = 25
CORE_BATCH_COUNT = 60
CORE_R3_MATERIALIZATION_MANIFEST_FILE_SHA256 = (
    "9b68fbb5e48ff5dc7b96701897391f4918013be1603b0e9e37c9dcbddadccf43"
)

_VERIFIED_CORE_INPUTS_MARKER = object()
_CATALOG_FILES = ("assets.jsonl", "components.jsonl", "manifest.json")


class PortfolioCoreInputError(ValueError):
    """The Core r2/r3, split, catalog, or permission chain drifted."""


@dataclass(frozen=True)
class PortfolioCoreInputFiles:
    """Exact Core artifacts required by the split-aware loader."""

    plan_path: Path
    expected_plan_sha256: str
    pre_generation_manifest_path: Path
    expected_pre_generation_manifest_file_sha256: str
    split_assignment_path: Path
    expected_split_assignment_sha256: str
    query_artifact_path: Path
    expected_query_artifact_sha256: str
    materialization_manifest_path: Path
    expected_materialization_manifest_file_sha256: str
    capability_assignments_path: Path
    expected_capability_assignments_sha256: str
    base_catalog_dir: Path
    expected_base_catalog_sha256: str
    asset_root: Path
    runtime_catalog_dir: Path | None = None
    expected_runtime_catalog_sha256: str | None = None
    remote_files: PortfolioRemoteProcessingFiles | None = None


@dataclass(frozen=True)
class PortfolioCoreBatch:
    """One immutable 25-query Core execution unit."""

    batch_id: str
    split: CoreSplit
    query_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedPortfolioCoreInputs:
    """Deeply verified Portfolio-only Core r3 input handle."""

    files: PortfolioCoreInputFiles
    plan: CorpusPlan
    queries: tuple[Query, ...]
    assistant_queries: tuple[AssistantQueryInput, ...]
    query_assets: tuple[PortfolioQueryAssetBinding, ...]
    batches: tuple[PortfolioCoreBatch, ...]
    remote_runtimes: tuple[VerifiedPortfolioRemoteProcessingRuntime, ...] = field(
        repr=False,
        compare=False,
    )
    _snapshots: tuple[tuple[Path, bytes], ...] = field(
        repr=False,
        compare=False,
    )
    _marker: object = field(repr=False, compare=False)
    _verified_catalogs: tuple[AssetCatalog, AssetCatalog] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    formal_eligible: Literal[False] = False

    @property
    def expected_plan_sha256(self) -> str:
        return self.files.expected_plan_sha256

    @property
    def expected_plan_manifest_file_sha256(self) -> str:
        return self.files.expected_pre_generation_manifest_file_sha256

    @property
    def expected_query_artifact_sha256(self) -> str:
        return self.files.expected_query_artifact_sha256

    @property
    def expected_capability_assignments_sha256(self) -> str:
        return self.files.expected_capability_assignments_sha256

    @property
    def expected_base_catalog_sha256(self) -> str:
        return self.files.expected_base_catalog_sha256

    @property
    def expected_output_catalog_sha256(self) -> str:
        return (
            self.files.expected_runtime_catalog_sha256
            or self.files.expected_base_catalog_sha256
        )

    def runtime_for(self, processor: str) -> VerifiedPortfolioRemoteProcessingRuntime:
        verified = require_verified_portfolio_core_inputs(self)
        for runtime in verified.remote_runtimes:
            if runtime.processor == processor:
                return runtime
        raise PortfolioCoreInputError(
            f"missing verified Core processor runtime: {processor}"
        )

    def runtime_asset_catalog(self) -> AssetCatalog:
        """Return the exact already-verified runtime catalog for this process.

        Core loading decodes every catalog image to verify its pHash.  Consumers
        in the same process must reuse that proof instead of decoding the full
        catalog again; metadata snapshots and all permission bindings are still
        rebuilt by ``require_verified_portfolio_core_inputs``.
        """

        verified = require_verified_portfolio_core_inputs(self)
        if verified._verified_catalogs is None:
            raise PortfolioCoreInputError(
                "verified Core inputs lack their runtime catalog proof"
            )
        catalog = verified._verified_catalogs[1]
        catalog.require_verified_files()
        return catalog


@dataclass(frozen=True)
class _LoadedCoreState:
    plan: CorpusPlan
    queries: tuple[Query, ...]
    assistant_queries: tuple[AssistantQueryInput, ...]
    query_assets: tuple[PortfolioQueryAssetBinding, ...]
    batches: tuple[PortfolioCoreBatch, ...]
    remote_runtimes: tuple[VerifiedPortfolioRemoteProcessingRuntime, ...]
    verified_catalogs: tuple[AssetCatalog, AssetCatalog]
    snapshots: tuple[tuple[Path, bytes], ...]


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PortfolioCoreInputError(f"{label} must be a lowercase SHA-256")
    return value


def _normalized_files(value: PortfolioCoreInputFiles) -> PortfolioCoreInputFiles:
    if not isinstance(value, PortfolioCoreInputFiles):
        raise TypeError("Core inputs require PortfolioCoreInputFiles")
    runtime_dir = (
        None
        if value.runtime_catalog_dir is None
        else Path(value.runtime_catalog_dir).absolute()
    )
    runtime_sha = value.expected_runtime_catalog_sha256
    if (runtime_dir is None) != (runtime_sha is None):
        raise PortfolioCoreInputError(
            "runtime catalog path and logical SHA-256 must be supplied together"
        )
    return PortfolioCoreInputFiles(
        plan_path=Path(value.plan_path).absolute(),
        expected_plan_sha256=_require_sha256(
            value.expected_plan_sha256, "expected Core plan SHA-256"
        ),
        pre_generation_manifest_path=Path(
            value.pre_generation_manifest_path
        ).absolute(),
        expected_pre_generation_manifest_file_sha256=_require_sha256(
            value.expected_pre_generation_manifest_file_sha256,
            "expected Core pre-generation manifest file SHA-256",
        ),
        split_assignment_path=Path(value.split_assignment_path).absolute(),
        expected_split_assignment_sha256=_require_sha256(
            value.expected_split_assignment_sha256,
            "expected Core split-assignment SHA-256",
        ),
        query_artifact_path=Path(value.query_artifact_path).absolute(),
        expected_query_artifact_sha256=_require_sha256(
            value.expected_query_artifact_sha256,
            "expected Core query artifact SHA-256",
        ),
        materialization_manifest_path=Path(
            value.materialization_manifest_path
        ).absolute(),
        expected_materialization_manifest_file_sha256=_require_sha256(
            value.expected_materialization_manifest_file_sha256,
            "expected Core materialization manifest file SHA-256",
        ),
        capability_assignments_path=Path(value.capability_assignments_path).absolute(),
        expected_capability_assignments_sha256=_require_sha256(
            value.expected_capability_assignments_sha256,
            "expected Core capability-assignments SHA-256",
        ),
        base_catalog_dir=Path(value.base_catalog_dir).absolute(),
        expected_base_catalog_sha256=_require_sha256(
            value.expected_base_catalog_sha256,
            "expected Core base-catalog SHA-256",
        ),
        asset_root=Path(value.asset_root).absolute(),
        runtime_catalog_dir=runtime_dir,
        expected_runtime_catalog_sha256=(
            None
            if runtime_sha is None
            else _require_sha256(runtime_sha, "expected Core runtime-catalog SHA-256")
        ),
        remote_files=value.remote_files,
    )


def load_verified_portfolio_core_inputs(
    files: PortfolioCoreInputFiles,
) -> VerifiedPortfolioCoreInputs:
    """Load and verify the exact 1,500-query Core r3 corpus without model calls."""

    normalized = _normalized_files(files)
    state = _load_core_state(normalized)
    return VerifiedPortfolioCoreInputs(
        files=normalized,
        plan=state.plan,
        queries=state.queries,
        assistant_queries=state.assistant_queries,
        query_assets=state.query_assets,
        batches=state.batches,
        remote_runtimes=state.remote_runtimes,
        _snapshots=state.snapshots,
        _marker=_VERIFIED_CORE_INPUTS_MARKER,
        _verified_catalogs=state.verified_catalogs,
        formal_eligible=False,
    )


def require_verified_portfolio_core_inputs(
    value: object,
) -> VerifiedPortfolioCoreInputs:
    """Rebuild the full proof and reject stale or forged Core handles."""

    if (
        type(value) is not VerifiedPortfolioCoreInputs
        or value._marker is not _VERIFIED_CORE_INPUTS_MARKER
        or value.formal_eligible is not False
    ):
        raise TypeError("Core inputs require the dedicated external-digest loader")
    rebuilt = _load_core_state(
        value.files,
        _verified_catalogs=value._verified_catalogs,
    )
    held = (
        value.plan,
        value.queries,
        value.assistant_queries,
        value.query_assets,
        value.batches,
        _runtime_identities(value.remote_runtimes),
        value._snapshots,
    )
    current = (
        rebuilt.plan,
        rebuilt.queries,
        rebuilt.assistant_queries,
        rebuilt.query_assets,
        rebuilt.batches,
        _runtime_identities(rebuilt.remote_runtimes),
        rebuilt.snapshots,
    )
    if current != held:
        raise PortfolioCoreInputError("verified Core inputs changed or were mutated")
    return value


def _read(path: Path, label: str, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    try:
        return read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    except (ArtifactFormatError, OSError, ValueError) as error:
        raise PortfolioCoreInputError(f"unable to read {label}") from error


def _read_exact(path: Path, expected_sha256: str, label: str) -> bytes:
    content = _read(path, label)
    if sha256_bytes(content) != expected_sha256:
        raise PortfolioCoreInputError(f"{label} external digest mismatch")
    return content


def _canonical_object(content: bytes, label: str) -> dict:
    try:
        value = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise PortfolioCoreInputError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != content:
        raise PortfolioCoreInputError(f"{label} must be one canonical JSON object")
    return value


def _canonical_rows(content: bytes, label: str) -> tuple[dict, ...]:
    try:
        raw = parse_canonical_jsonl(content, label=label)
    except ArtifactFormatError as error:
        raise PortfolioCoreInputError(f"{label} is not canonical JSONL") from error
    if not raw or any(not isinstance(item, dict) for item in raw):
        raise PortfolioCoreInputError(f"{label} must contain canonical objects")
    rows = tuple(raw)  # type: ignore[arg-type]
    if canonical_jsonl_bytes(rows) != content:
        raise PortfolioCoreInputError(f"{label} is not canonical JSONL")
    return rows


def _manifest_file_binding(manifest: dict, relative_path: str) -> tuple[str, int]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise PortfolioCoreInputError("Core manifest files binding is invalid")
    matches = [
        item
        for item in files
        if isinstance(item, dict)
        and (item.get("relative_path") or item.get("path")) == relative_path
    ]
    if len(matches) != 1:
        raise PortfolioCoreInputError(
            f"Core manifest does not uniquely bind {relative_path}"
        )
    digest = matches[0].get("sha256")
    byte_count = matches[0].get("bytes")
    if not isinstance(digest, str) or not isinstance(byte_count, int):
        raise PortfolioCoreInputError("Core manifest file binding is malformed")
    return _require_sha256(digest, f"{relative_path} manifest SHA-256"), byte_count


def _parse_queries(content: bytes) -> tuple[Query, ...]:
    rows = _canonical_rows(content, "Core r3 query artifact")
    try:
        queries = tuple(Query.model_validate(item, strict=True) for item in rows)
    except ValidationError as error:
        raise PortfolioCoreInputError(
            "Core r3 query artifact violates Query v2"
        ) from error
    if (
        canonical_jsonl_bytes(tuple(query.model_dump(mode="json") for query in queries))
        != content
    ):
        raise PortfolioCoreInputError("Core r3 query artifact typed bytes drifted")
    return queries


def _split_map(content: bytes) -> dict[str, CoreSplit]:
    rows = _canonical_rows(content, "Core final-splits sidecar")
    result: dict[str, CoreSplit] = {}
    for row in rows:
        if set(row) != {"plan_id", "final_split"}:
            raise PortfolioCoreInputError("Core final-splits row has extra fields")
        plan_id = row.get("plan_id")
        split = row.get("final_split")
        if (
            not isinstance(plan_id, str)
            or split not in CORE_SPLIT_ORDER
            or plan_id in result
        ):
            raise PortfolioCoreInputError(
                "Core final-splits row is invalid or duplicated"
            )
        result[plan_id] = split
    return result


def _validate_query_binding(
    planned: PlannedQuery,
    query: Query,
    *,
    final_split: CoreSplit,
    plan_sha256: str,
) -> None:
    if (
        query.schema_version != 2
        or query.query_id != planned.plan_id
        or query.taxonomy_version != planned.taxonomy_version
        or query.task_spec_version != planned.task_spec_version
        or query.asset_id != planned.asset_id
        or query.image_path != planned.image_path
        or query.leakage_group_id != planned.leakage_group_id
        or query.boundary_group_id != planned.boundary_group_id
        or query.template_family != planned.template_family
        or query.generator_batch_id != planned.generator_batch_id
        or query.canonical_intent != planned.canonical_intent
        or query.canonical_capability != planned.canonical_capability
        or query.acceptable_capabilities != planned.acceptable_capabilities
        or query.is_boundary != planned.is_boundary
        or query.boundary_strategy != planned.boundary_strategy
        or query.requires_card != planned.requires_card
        or query.split != final_split
        or not query.label_provenance
        or query.label_provenance[-1].source_artifact_sha256 != plan_sha256
    ):
        raise PortfolioCoreInputError(
            f"Core r3 query drifted from r2 plan/final split: {query.query_id}"
        )


def _validate_core_semantics(
    *,
    plan: CorpusPlan,
    plan_sha256: str,
    queries: tuple[Query, ...],
    splits: dict[str, CoreSplit],
) -> tuple[PortfolioCoreBatch, ...]:
    if plan.scope != "core" or len(plan.queries) != CORE_QUERY_COUNT:
        raise PortfolioCoreInputError("Core loader requires the exact 1,500-query plan")
    if len(queries) != CORE_QUERY_COUNT or len(splits) != CORE_QUERY_COUNT:
        raise PortfolioCoreInputError(
            "Core r3/split sidecar must each cover 1,500 queries"
        )
    expected_ids = tuple(item.plan_id for item in plan.queries)
    actual_ids = tuple(item.query_id for item in queries)
    if actual_ids != expected_ids or set(splits) != set(expected_ids):
        raise PortfolioCoreInputError(
            "Core query/split order or membership differs from plan"
        )
    for planned, query in zip(plan.queries, queries, strict=True):
        _validate_query_binding(
            planned,
            query,
            final_split=splits[query.query_id],
            plan_sha256=plan_sha256,
        )

    split_counts = Counter(query.split for query in queries)
    if split_counts != Counter(CORE_SPLIT_COUNTS):
        raise PortfolioCoreInputError("Core final split counts drifted")
    capability_counts: dict[str, Counter[str]] = defaultdict(Counter)
    boundary_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for query in queries:
        assert query.canonical_capability is not None
        capability_counts[query.split][query.canonical_capability] += 1
        if query.is_boundary:
            boundary_counts[query.split][query.canonical_capability] += 1
    for split in CORE_SPLIT_ORDER:
        if dict(capability_counts[split]) != CORE_R2_CAPABILITY_COUNTS[split]:
            raise PortfolioCoreInputError(f"Core capability counts drifted: {split}")
        if dict(boundary_counts[split]) != CORE_R2_CAPABILITY_BOUNDARY_COUNTS[split]:
            raise PortfolioCoreInputError(f"Core boundary counts drifted: {split}")

    grouped: dict[str, list[Query]] = defaultdict(list)
    batch_order: list[str] = []
    for query in queries:
        if query.generator_batch_id not in grouped:
            batch_order.append(query.generator_batch_id)
        grouped[query.generator_batch_id].append(query)
    if len(batch_order) != CORE_BATCH_COUNT:
        raise PortfolioCoreInputError("Core generator-batch count drifted")
    batches: list[PortfolioCoreBatch] = []
    for batch_id in batch_order:
        members = grouped[batch_id]
        member_splits = {query.split for query in members}
        if len(members) != CORE_BATCH_SIZE or len(member_splits) != 1:
            raise PortfolioCoreInputError(
                f"Core batch is not one split-atomic 25-query unit: {batch_id}"
            )
        split = next(iter(member_splits))
        if split not in CORE_SPLIT_ORDER:  # pragma: no cover - Query schema guards it
            raise PortfolioCoreInputError("Core batch has an invalid split")
        batches.append(
            PortfolioCoreBatch(
                batch_id=batch_id,
                split=split,  # type: ignore[arg-type]
                query_ids=tuple(query.query_id for query in members),
            )
        )
    return tuple(batches)


def _assert_opaque_public_projection(
    query: Query,
    projected: AssistantQueryInput,
) -> None:
    try:
        public = parse_canonical_json(
            projected.public_input_json.encode("utf-8"),
            label=f"Core public query {query.query_id}",
        )
    except ArtifactFormatError as error:  # pragma: no cover - model already checks
        raise PortfolioCoreInputError("Core public query is not canonical") from error
    if not isinstance(public, dict) or set(public) != {
        "asset_id",
        "text",
        "turns",
    }:
        raise PortfolioCoreInputError(
            "Core provider query DTO is not the opaque allowlist"
        )
    if public["asset_id"] in {query.asset_id, query.image_path}:
        raise PortfolioCoreInputError(
            "Core provider query DTO exposes a private asset identity"
        )
    serialized = canonical_json_bytes(public).decode("utf-8")
    for forbidden_key in (
        "image_path",
        "source",
        "source_dataset",
        "source_record_id",
        "template_family",
        "canonical_intent",
        "canonical_capability",
        "acceptable_capabilities",
        "split",
        "label_provenance",
    ):
        if f'"{forbidden_key}"' in serialized:
            raise PortfolioCoreInputError(
                f"Core provider query DTO exposes private field: {forbidden_key}"
            )


def _runtime_identities(
    runtimes: tuple[VerifiedPortfolioRemoteProcessingRuntime, ...],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            runtime.processor,
            runtime.authorization_file_sha256,
            runtime.receipt_file_sha256,
            runtime.catalog.catalog_sha256,
        )
        for runtime in runtimes
    )


def _load_core_state(
    files: PortfolioCoreInputFiles,
    *,
    _verified_catalogs: tuple[AssetCatalog, AssetCatalog] | None = None,
) -> _LoadedCoreState:
    plan_bytes = _read_exact(
        files.plan_path, files.expected_plan_sha256, "Core r2 plan"
    )
    pregen_bytes = _read_exact(
        files.pre_generation_manifest_path,
        files.expected_pre_generation_manifest_file_sha256,
        "Core pre-generation manifest",
    )
    split_bytes = _read_exact(
        files.split_assignment_path,
        files.expected_split_assignment_sha256,
        "Core final-splits sidecar",
    )
    query_bytes = _read_exact(
        files.query_artifact_path,
        files.expected_query_artifact_sha256,
        "Core r3 query artifact",
    )
    materialization_bytes = _read_exact(
        files.materialization_manifest_path,
        files.expected_materialization_manifest_file_sha256,
        "Core r3 materialization manifest",
    )
    capability_bytes = _read_exact(
        files.capability_assignments_path,
        files.expected_capability_assignments_sha256,
        "Core capability assignments",
    )

    try:
        plan = CorpusPlan.model_validate_json(plan_bytes, strict=True)
    except ValidationError as error:
        raise PortfolioCoreInputError("Core r2 plan violates CorpusPlan") from error
    if canonical_json_bytes(plan.model_dump(mode="json")) != plan_bytes:
        raise PortfolioCoreInputError("Core r2 plan is not canonical JSON")
    pregen = _canonical_object(pregen_bytes, "Core pre-generation manifest")
    materialization = _canonical_object(
        materialization_bytes, "Core r3 materialization manifest"
    )
    declared_materialization_sha256 = materialization.get("manifest_sha256")
    materialization_content = dict(materialization)
    materialization_content.pop("manifest_sha256", None)
    if (
        not isinstance(declared_materialization_sha256, str)
        or sha256_bytes(canonical_json_bytes(materialization_content))
        != declared_materialization_sha256
    ):
        raise PortfolioCoreInputError(
            "Core r3 materialization manifest self-hash drifted"
        )
    _canonical_rows(capability_bytes, "Core capability assignments")
    splits = _split_map(split_bytes)
    queries = _parse_queries(query_bytes)

    plan_binding = _manifest_file_binding(pregen, "plan/core.json")
    split_binding = _manifest_file_binding(pregen, "sidecars/final-splits.jsonl")
    query_binding = _manifest_file_binding(materialization, "queries.jsonl")
    if (
        pregen.get("r2_plan_sha256") != files.expected_plan_sha256
        or plan_binding != (files.expected_plan_sha256, len(plan_bytes))
        or split_binding != (files.expected_split_assignment_sha256, len(split_bytes))
        or materialization.get("queries_sha256") != files.expected_query_artifact_sha256
        or materialization.get("source_query_count") != CORE_QUERY_COUNT
        or materialization.get("parent_batch_count") != CORE_BATCH_COUNT
        or query_binding != (files.expected_query_artifact_sha256, len(query_bytes))
    ):
        raise PortfolioCoreInputError("Core publication manifest bindings drifted")
    if (
        plan.asset_catalog_sha256 != files.expected_base_catalog_sha256
        or plan.capability_assignments_sha256
        != files.expected_capability_assignments_sha256
    ):
        raise PortfolioCoreInputError("Core plan catalog/assignment bindings drifted")

    batches = _validate_core_semantics(
        plan=plan,
        plan_sha256=files.expected_plan_sha256,
        queries=queries,
        splits=splits,
    )

    if _verified_catalogs is None:
        try:
            base_catalog = load_asset_catalog(
                files.base_catalog_dir, files.asset_root, verify_files=True
            )
            base_catalog.require_verified_files()
        except (ArtifactFormatError, OSError, ValueError) as error:
            raise PortfolioCoreInputError(
                "Core base catalog verification failed"
            ) from error
    else:
        if (
            not isinstance(_verified_catalogs, tuple)
            or len(_verified_catalogs) != 2
            or any(type(item) is not AssetCatalog for item in _verified_catalogs)
        ):
            raise PortfolioCoreInputError("cached Core catalog proof is invalid")
        base_catalog = _verified_catalogs[0]
        try:
            base_catalog.require_verified_files()
        except (ArtifactFormatError, OSError, ValueError) as error:
            raise PortfolioCoreInputError(
                "cached Core base catalog proof is invalid"
            ) from error
        if (
            base_catalog.root.resolve() != files.base_catalog_dir.resolve()
            or base_catalog.asset_root != files.asset_root.resolve()
        ):
            raise PortfolioCoreInputError(
                "cached Core base catalog paths differ from input binding"
            )
    if base_catalog.catalog_sha256 != files.expected_base_catalog_sha256:
        raise PortfolioCoreInputError("Core base catalog logical SHA-256 mismatch")

    runtime_catalog = base_catalog
    if files.runtime_catalog_dir is not None and _verified_catalogs is None:
        try:
            runtime_catalog = load_asset_catalog(
                files.runtime_catalog_dir, files.asset_root, verify_files=True
            )
            runtime_catalog.require_verified_files()
        except (ArtifactFormatError, OSError, ValueError) as error:
            raise PortfolioCoreInputError(
                "Core runtime catalog verification failed"
            ) from error
        if runtime_catalog.catalog_sha256 != files.expected_runtime_catalog_sha256:
            raise PortfolioCoreInputError(
                "Core runtime catalog logical SHA-256 mismatch"
            )
    elif files.runtime_catalog_dir is not None:
        runtime_catalog = _verified_catalogs[1]
        try:
            runtime_catalog.require_verified_files()
        except (ArtifactFormatError, OSError, ValueError) as error:
            raise PortfolioCoreInputError(
                "cached Core runtime catalog proof is invalid"
            ) from error
        if (
            runtime_catalog.root.resolve() != files.runtime_catalog_dir.resolve()
            or runtime_catalog.asset_root != files.asset_root.resolve()
        ):
            raise PortfolioCoreInputError(
                "cached Core runtime catalog paths differ from input binding"
            )
        if runtime_catalog.catalog_sha256 != files.expected_runtime_catalog_sha256:
            raise PortfolioCoreInputError(
                "Core runtime catalog logical SHA-256 mismatch"
            )
    elif _verified_catalogs is not None and _verified_catalogs[1] is not base_catalog:
        raise PortfolioCoreInputError(
            "cached Core catalog pair differs from single-catalog input binding"
        )

    query_assets: list[PortfolioQueryAssetBinding] = []
    try:
        for query in queries:
            base = base_catalog.verify_reference(
                query.asset_id,
                query.image_path,
                leakage_group_id=query.leakage_group_id,
            )
            active = runtime_catalog.verify_reference(
                query.asset_id,
                query.image_path,
                leakage_group_id=query.leakage_group_id,
            )
            if (
                active.asset.asset_id != base.asset.asset_id
                or active.asset.local_path != base.asset.local_path
                or active.asset.sha256 != base.asset.sha256
                or active.leakage_group_id != base.leakage_group_id
            ):
                raise PortfolioCoreInputError(
                    "Core runtime catalog changed an asset identity or leakage binding"
                )
            query_assets.append(
                PortfolioQueryAssetBinding(
                    query_id=query.query_id,
                    asset_id=query.asset_id,
                    image_path=query.image_path,
                    image_sha256=active.asset.sha256,
                )
            )
    except (ArtifactFormatError, OSError, ValueError) as error:
        if isinstance(error, PortfolioCoreInputError):
            raise
        raise PortfolioCoreInputError("Core query asset binding failed") from error

    assistant_queries = tuple(build_assistant_query_input(query) for query in queries)
    for query, projected in zip(queries, assistant_queries, strict=True):
        _assert_opaque_public_projection(query, projected)

    runtimes: tuple[VerifiedPortfolioRemoteProcessingRuntime, ...] = ()
    if files.remote_files is not None:
        remote = files.remote_files
        if (
            remote.base_catalog_dir.absolute() != files.base_catalog_dir
            or remote.output_catalog_dir.absolute()
            != (files.runtime_catalog_dir or files.base_catalog_dir)
            or remote.asset_root.absolute() != files.asset_root
            or remote.processor_order
            not in {
                CORE_PORTFOLIO_PROCESSOR_ORDER,
                ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
            }
        ):
            raise PortfolioCoreInputError(
                "Core remote files differ from loader catalogs"
            )
        try:
            runtimes = tuple(
                verify_portfolio_remote_processing_runtime(
                    authorization_file=remote.authorization_file,
                    expected_authorization_sha256=(
                        remote.expected_authorization_file_sha256
                    ),
                    receipt_file=remote.receipt_file,
                    expected_receipt_file_sha256=remote.expected_receipt_file_sha256,
                    selection_manifest=remote.selection_manifest,
                    dataset_assets=remote.dataset_assets,
                    base_catalog=remote.base_catalog_dir,
                    output_catalog=remote.output_catalog_dir,
                    asset_root=remote.asset_root,
                    plan_file=files.plan_path,
                    queries_file=files.query_artifact_path,
                    processor=processor,
                    _verified_catalogs=(base_catalog, runtime_catalog),
                )
                for processor in remote.processor_order
            )
        except (ArtifactFormatError, OSError, ValueError) as error:
            raise PortfolioCoreInputError(
                "Core remote-processing runtime verification failed"
            ) from error
        if (
            tuple(item.processor for item in runtimes) != remote.processor_order
            or any(item.plan_sha256 != files.expected_plan_sha256 for item in runtimes)
            or any(
                item.query_artifact_sha256 != files.expected_query_artifact_sha256
                for item in runtimes
            )
            or any(
                item.catalog.catalog_sha256
                != (
                    files.expected_runtime_catalog_sha256
                    or files.expected_base_catalog_sha256
                )
                for item in runtimes
            )
        ):
            raise PortfolioCoreInputError("Core remote runtime bindings drifted")

    snapshot_paths = (
        files.plan_path,
        files.pre_generation_manifest_path,
        files.split_assignment_path,
        files.query_artifact_path,
        files.materialization_manifest_path,
        files.capability_assignments_path,
        *(files.base_catalog_dir / name for name in _CATALOG_FILES),
        *(
            ()
            if files.runtime_catalog_dir is None
            or files.runtime_catalog_dir == files.base_catalog_dir
            else tuple(files.runtime_catalog_dir / name for name in _CATALOG_FILES)
        ),
    )
    snapshots = tuple(
        (path, _read(path, f"Core snapshot {path.name}")) for path in snapshot_paths
    )
    if (
        snapshots[0][1] != plan_bytes
        or snapshots[1][1] != pregen_bytes
        or snapshots[2][1] != split_bytes
        or snapshots[3][1] != query_bytes
        or snapshots[4][1] != materialization_bytes
        or snapshots[5][1] != capability_bytes
    ):
        raise PortfolioCoreInputError("Core inputs changed during verified loading")
    return _LoadedCoreState(
        plan=plan,
        queries=queries,
        assistant_queries=assistant_queries,
        query_assets=tuple(query_assets),
        batches=batches,
        remote_runtimes=runtimes,
        verified_catalogs=(base_catalog, runtime_catalog),
        snapshots=snapshots,
    )


__all__ = [
    "CORE_BATCH_COUNT",
    "CORE_BATCH_SIZE",
    "CORE_QUERY_COUNT",
    "CORE_R3_MATERIALIZATION_MANIFEST_FILE_SHA256",
    "CORE_SPLIT_COUNTS",
    "CORE_SPLIT_ORDER",
    "CoreSplit",
    "PortfolioCoreBatch",
    "PortfolioCoreInputError",
    "PortfolioCoreInputFiles",
    "VerifiedPortfolioCoreInputs",
    "load_verified_portfolio_core_inputs",
    "require_verified_portfolio_core_inputs",
]
