"""Verified Portfolio ``dev_mini`` inputs without weakening Formal Phase 4.

The accepted Portfolio corpus is a local, owner-reviewed artifact with a
different lifecycle from the frozen Formal Research split.  This module keeps
those boundaries separate: it verifies the eight accepted batches, their
derived 200-query view, the historical v1 plan/catalog, capability assignments,
    and the owner-authorized remote-processing overlay.

Loading these inputs never invokes a model and never grants formal eligibility.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
)

from skillchain.data.asset_catalog import load_asset_catalog
from skillchain.data.portfolio_remote_processing import (
    PortfolioProcessor,
    VerifiedPortfolioRemoteProcessingRuntime,
    require_verified_portfolio_remote_processing_runtime,
    verify_portfolio_remote_processing_runtime,
)
from skillchain.evaluation.assistant_runs import (
    AssistantQueryInput,
    build_assistant_query_input,
)
from skillchain.schemas import Query
from skillchain.synthesis.batches import (
    AcceptedLedgerEntry,
    verify_accepted_corpus,
)
from skillchain.synthesis.models import CorpusPlan, PlanManifest, PlannedQuery
from skillchain.synthesis.planning import (
    CapabilityAssignment,
    DEV_MINI_CAPABILITY_COUNTS,
    load_capability_assignments,
    load_plan,
)
from skillchain.synthesis.store import canonical_json_bytes, canonical_jsonl_bytes
from skillchain.tools.serialization import (
    ArtifactFormatError,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

# Historical v2 snapshot order remains the default because the accepted
# zero-call checklist and Stage 1 handoff bind that exact authorization.
# New launch preparation may pass an explicit forward processor order without
# rewriting those historical artifacts.
PORTFOLIO_PROCESSOR_ORDER: tuple[PortfolioProcessor, ...] = (
    "dashscope-qwen-assistant",
    "dashscope-kimi-feedback",
    "dashscope-kimi-judge",
)
ACTIVE_PORTFOLIO_PROCESSOR_ORDER: tuple[PortfolioProcessor, ...] = (
    "dashscope-qwen-assistant",
    "dashscope-kimi-feedback",
    "aifast-gemini-judge",
)
# Core v9 has its own owner authorization.  Keep this separate from the active
# dev_mini order so the historical AIFast/Gemini launch contract remains
# byte-for-byte readable while Core stays inside its DashScope-only scope.
CORE_PORTFOLIO_PROCESSOR_ORDER: tuple[PortfolioProcessor, ...] = (
    "dashscope-qwen-assistant",
    "dashscope-kimi-feedback",
    "dashscope-kimi-judge",
)
ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER: tuple[PortfolioProcessor, ...] = (
    "dashscope-qwen-assistant",
    "dashscope-kimi-feedback",
    "aifast-gemini-judge",
)
PORTFOLIO_DEV_MINI_COUNT = 200
PORTFOLIO_DEV_MINI_BATCH_COUNT = 8
PORTFOLIO_DEV_MINI_BATCH_SIZE = 25

_VERIFIED_PORTFOLIO_INPUTS_MARKER = object()
_CATALOG_FILES = ("assets.jsonl", "components.jsonl", "manifest.json")


class PortfolioInputError(ValueError):
    """The Portfolio corpus, plan, catalog, or permission chain drifted."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PortfolioQueryAssetBinding(_StrictFrozenModel):
    """The image identity used by one accepted query."""

    query_id: str
    asset_id: str
    image_path: str
    image_sha256: Sha256

    @field_validator("query_id", "asset_id", "image_path")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        if not value or value != value.strip():
            raise ValueError(f"{info.field_name} must be non-blank and trimmed")
        return value


@dataclass(frozen=True)
class PortfolioRemoteProcessingFiles:
    """Exact local artifacts used to rebuild the permission preflight."""

    authorization_file: Path
    expected_authorization_file_sha256: str
    receipt_file: Path
    expected_receipt_file_sha256: str
    selection_manifest: Path
    dataset_assets: Path
    base_catalog_dir: Path
    output_catalog_dir: Path
    asset_root: Path
    processor_order: tuple[PortfolioProcessor, ...] = PORTFOLIO_PROCESSOR_ORDER


@dataclass(frozen=True)
class VerifiedPortfolioDevMiniInputs:
    """Deeply verified, Portfolio-only 200-query input handle."""

    queries_root: Path
    plan_path: Path
    plan_manifest_path: Path
    ledger_path: Path
    queries_path: Path
    capability_assignments_path: Path
    expected_plan_sha256: str
    expected_plan_manifest_file_sha256: str
    expected_accepted_ledger_sha256: str
    expected_query_artifact_sha256: str
    expected_capability_assignments_sha256: str
    expected_seed_set_sha256: str
    expected_base_catalog_sha256: str
    expected_output_catalog_sha256: str
    remote_files: PortfolioRemoteProcessingFiles
    plan: CorpusPlan
    plan_manifest: PlanManifest
    ledger: tuple[AcceptedLedgerEntry, ...]
    queries: tuple[Query, ...]
    assistant_queries: tuple[AssistantQueryInput, ...]
    query_assets: tuple[PortfolioQueryAssetBinding, ...]
    remote_runtimes: tuple[VerifiedPortfolioRemoteProcessingRuntime, ...] = field(
        repr=False,
        compare=False,
    )
    _snapshots: tuple[tuple[Path, bytes], ...] = field(
        repr=False,
        compare=False,
    )
    _marker: object = field(repr=False, compare=False)
    formal_eligible: Literal[False] = False

    def runtime_for(
        self,
        processor: PortfolioProcessor,
    ) -> VerifiedPortfolioRemoteProcessingRuntime:
        """Return the already verified runtime for one exact processor."""

        verified = require_verified_portfolio_dev_mini_inputs(self)
        for runtime in verified.remote_runtimes:
            if runtime.processor == processor:
                return runtime
        raise PortfolioInputError(f"missing verified processor runtime: {processor}")


@dataclass(frozen=True)
class _LoadedPortfolioState:
    plan: CorpusPlan
    plan_manifest: PlanManifest
    ledger: tuple[AcceptedLedgerEntry, ...]
    queries: tuple[Query, ...]
    assistant_queries: tuple[AssistantQueryInput, ...]
    query_assets: tuple[PortfolioQueryAssetBinding, ...]
    remote_runtimes: tuple[VerifiedPortfolioRemoteProcessingRuntime, ...]
    snapshots: tuple[tuple[Path, bytes], ...]


def load_verified_portfolio_dev_mini_inputs(
    *,
    queries_root: str | Path,
    plan_path: str | Path,
    capability_assignments_path: str | Path,
    expected_plan_sha256: str,
    expected_plan_manifest_file_sha256: str,
    expected_accepted_ledger_sha256: str,
    expected_query_artifact_sha256: str,
    expected_capability_assignments_sha256: str,
    expected_seed_set_sha256: str,
    expected_base_catalog_sha256: str,
    expected_output_catalog_sha256: str,
    remote_files: PortfolioRemoteProcessingFiles,
) -> VerifiedPortfolioDevMiniInputs:
    """Load the accepted 200-query Portfolio corpus and all permission bindings."""

    root = Path(queries_root).absolute()
    plan = Path(plan_path).absolute()
    assignments = Path(capability_assignments_path).absolute()
    manifest = plan.with_name(f"{plan.stem}.manifest.json")
    ledger = root / "accepted-ledger.jsonl"
    queries = root / "queries.jsonl"
    expected_hashes = tuple(
        _require_sha256(value, label)
        for value, label in (
            (expected_plan_sha256, "expected plan SHA-256"),
            (
                expected_plan_manifest_file_sha256,
                "expected plan manifest file SHA-256",
            ),
            (
                expected_accepted_ledger_sha256,
                "expected accepted ledger SHA-256",
            ),
            (
                expected_query_artifact_sha256,
                "expected query artifact SHA-256",
            ),
            (
                expected_capability_assignments_sha256,
                "expected capability assignments SHA-256",
            ),
            (expected_seed_set_sha256, "expected seed set SHA-256"),
            (expected_base_catalog_sha256, "expected base catalog SHA-256"),
            (expected_output_catalog_sha256, "expected output catalog SHA-256"),
        )
    )
    normalized_remote = _normalize_remote_files(remote_files)
    state = _load_portfolio_state(
        queries_root=root,
        plan_path=plan,
        plan_manifest_path=manifest,
        ledger_path=ledger,
        queries_path=queries,
        capability_assignments_path=assignments,
        expected_plan_sha256=expected_hashes[0],
        expected_plan_manifest_file_sha256=expected_hashes[1],
        expected_accepted_ledger_sha256=expected_hashes[2],
        expected_query_artifact_sha256=expected_hashes[3],
        expected_capability_assignments_sha256=expected_hashes[4],
        expected_seed_set_sha256=expected_hashes[5],
        expected_base_catalog_sha256=expected_hashes[6],
        expected_output_catalog_sha256=expected_hashes[7],
        remote_files=normalized_remote,
    )
    return VerifiedPortfolioDevMiniInputs(
        queries_root=root,
        plan_path=plan,
        plan_manifest_path=manifest,
        ledger_path=ledger,
        queries_path=queries,
        capability_assignments_path=assignments,
        expected_plan_sha256=expected_hashes[0],
        expected_plan_manifest_file_sha256=expected_hashes[1],
        expected_accepted_ledger_sha256=expected_hashes[2],
        expected_query_artifact_sha256=expected_hashes[3],
        expected_capability_assignments_sha256=expected_hashes[4],
        expected_seed_set_sha256=expected_hashes[5],
        expected_base_catalog_sha256=expected_hashes[6],
        expected_output_catalog_sha256=expected_hashes[7],
        remote_files=normalized_remote,
        plan=state.plan,
        plan_manifest=state.plan_manifest,
        ledger=state.ledger,
        queries=state.queries,
        assistant_queries=state.assistant_queries,
        query_assets=state.query_assets,
        remote_runtimes=state.remote_runtimes,
        _snapshots=state.snapshots,
        _marker=_VERIFIED_PORTFOLIO_INPUTS_MARKER,
        formal_eligible=False,
    )


def require_verified_portfolio_dev_mini_inputs(
    value: object,
) -> VerifiedPortfolioDevMiniInputs:
    """Rebuild the complete local proof and reject forged or stale handles."""

    if (
        type(value) is not VerifiedPortfolioDevMiniInputs
        or value._marker is not _VERIFIED_PORTFOLIO_INPUTS_MARKER
        or value.formal_eligible is not False
    ):
        raise TypeError("Portfolio inputs require the dedicated external-digest loader")
    rebuilt = _load_portfolio_state(
        queries_root=value.queries_root,
        plan_path=value.plan_path,
        plan_manifest_path=value.plan_manifest_path,
        ledger_path=value.ledger_path,
        queries_path=value.queries_path,
        capability_assignments_path=value.capability_assignments_path,
        expected_plan_sha256=value.expected_plan_sha256,
        expected_plan_manifest_file_sha256=(value.expected_plan_manifest_file_sha256),
        expected_accepted_ledger_sha256=value.expected_accepted_ledger_sha256,
        expected_query_artifact_sha256=value.expected_query_artifact_sha256,
        expected_capability_assignments_sha256=(
            value.expected_capability_assignments_sha256
        ),
        expected_seed_set_sha256=value.expected_seed_set_sha256,
        expected_base_catalog_sha256=value.expected_base_catalog_sha256,
        expected_output_catalog_sha256=value.expected_output_catalog_sha256,
        remote_files=value.remote_files,
    )
    held = (
        value.plan,
        value.plan_manifest,
        value.ledger,
        value.queries,
        value.assistant_queries,
        value.query_assets,
        _remote_runtime_identities(value.remote_runtimes),
        value._snapshots,
    )
    current = (
        rebuilt.plan,
        rebuilt.plan_manifest,
        rebuilt.ledger,
        rebuilt.queries,
        rebuilt.assistant_queries,
        rebuilt.query_assets,
        _remote_runtime_identities(rebuilt.remote_runtimes),
        rebuilt.snapshots,
    )
    if current != held:
        raise PortfolioInputError(
            "verified Portfolio inputs changed or the typed handle was mutated"
        )
    return value


def _load_portfolio_state(
    *,
    queries_root: Path,
    plan_path: Path,
    plan_manifest_path: Path,
    ledger_path: Path,
    queries_path: Path,
    capability_assignments_path: Path,
    expected_plan_sha256: str,
    expected_plan_manifest_file_sha256: str,
    expected_accepted_ledger_sha256: str,
    expected_query_artifact_sha256: str,
    expected_capability_assignments_sha256: str,
    expected_seed_set_sha256: str,
    expected_base_catalog_sha256: str,
    expected_output_catalog_sha256: str,
    remote_files: PortfolioRemoteProcessingFiles,
) -> _LoadedPortfolioState:
    core_paths = (
        plan_path,
        plan_manifest_path,
        ledger_path,
        queries_path,
        capability_assignments_path,
    )
    core_contents = _read_regular_files(core_paths)
    for label, content, expected in zip(
        (
            "Portfolio dev_mini plan",
            "Portfolio plan manifest",
            "Portfolio accepted ledger",
            "Portfolio query artifact",
            "Portfolio capability assignments",
        ),
        core_contents,
        (
            expected_plan_sha256,
            expected_plan_manifest_file_sha256,
            expected_accepted_ledger_sha256,
            expected_query_artifact_sha256,
            expected_capability_assignments_sha256,
        ),
        strict=True,
    ):
        if sha256_bytes(content) != expected:
            raise PortfolioInputError(f"{label} external digest mismatch")

    ledger = _parse_ledger(core_contents[2])
    accepted_paths = tuple(
        path
        for entry in ledger
        for path in (
            queries_root / "accepted" / entry.batch_id / "manifest.json",
            queries_root / "accepted" / entry.batch_id / "results.jsonl",
        )
    )
    remote_paths = (
        remote_files.authorization_file,
        remote_files.receipt_file,
        remote_files.selection_manifest,
        remote_files.dataset_assets,
        *(remote_files.base_catalog_dir / name for name in _CATALOG_FILES),
        *(remote_files.output_catalog_dir / name for name in _CATALOG_FILES),
    )
    snapshot_paths = _unique_paths((*core_paths, *accepted_paths, *remote_paths))
    before = _read_regular_files(snapshot_paths)
    if before[: len(core_paths)] != core_contents:
        raise PortfolioInputError(
            "Portfolio core inputs changed before verified loading"
        )

    try:
        accepted_queries = tuple(verify_accepted_corpus(queries_root))
        plan, plan_manifest = load_plan(plan_path)
        assignments = load_capability_assignments(
            capability_assignments_path,
            expected_sha256=expected_capability_assignments_sha256,
        )
        base_catalog = load_asset_catalog(
            remote_files.base_catalog_dir,
            remote_files.asset_root,
            verify_files=True,
        )
        base_catalog.require_verified_files()
    except (ArtifactFormatError, OSError, ValueError) as error:
        raise PortfolioInputError(
            "Portfolio accepted input verification failed"
        ) from error

    if core_contents[0] != canonical_json_bytes(plan.model_dump(mode="json")):
        raise PortfolioInputError("Portfolio plan is not canonical JSON")
    if core_contents[1] != canonical_json_bytes(plan_manifest.model_dump(mode="json")):
        raise PortfolioInputError("Portfolio plan manifest is not canonical JSON")
    if core_contents[3] != canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in accepted_queries)
    ):
        raise PortfolioInputError("Portfolio query artifact is not canonical JSONL")

    _validate_portfolio_semantics(
        plan=plan,
        plan_manifest=plan_manifest,
        ledger=ledger,
        queries=accepted_queries,
        assignments=assignments,
        expected_plan_sha256=expected_plan_sha256,
        expected_seed_set_sha256=expected_seed_set_sha256,
        expected_capability_assignments_sha256=(expected_capability_assignments_sha256),
        expected_base_catalog_sha256=expected_base_catalog_sha256,
    )
    if base_catalog.catalog_sha256 != expected_base_catalog_sha256:
        raise PortfolioInputError("Portfolio base catalog SHA-256 mismatch")

    query_assets: list[PortfolioQueryAssetBinding] = []
    try:
        for query in accepted_queries:
            resolution = base_catalog.verify_reference(
                query.asset_id,
                query.image_path,
                leakage_group_id=query.leakage_group_id,
            )
            query_assets.append(
                PortfolioQueryAssetBinding(
                    query_id=query.query_id,
                    asset_id=query.asset_id,
                    image_path=query.image_path,
                    image_sha256=resolution.asset.sha256,
                )
            )
    except (ArtifactFormatError, OSError, ValueError) as error:
        raise PortfolioInputError(
            "Portfolio query asset binding verification failed"
        ) from error

    image_paths, image_before = _read_verified_query_images(
        tuple(query_assets),
        asset_root=remote_files.asset_root,
    )

    try:
        runtimes = tuple(
            verify_portfolio_remote_processing_runtime(
                authorization_file=remote_files.authorization_file,
                expected_authorization_sha256=(
                    remote_files.expected_authorization_file_sha256
                ),
                receipt_file=remote_files.receipt_file,
                expected_receipt_file_sha256=(
                    remote_files.expected_receipt_file_sha256
                ),
                selection_manifest=remote_files.selection_manifest,
                dataset_assets=remote_files.dataset_assets,
                base_catalog=remote_files.base_catalog_dir,
                output_catalog=remote_files.output_catalog_dir,
                asset_root=remote_files.asset_root,
                plan_file=plan_path,
                queries_file=queries_path,
                processor=processor,
            )
            for processor in remote_files.processor_order
        )
        _validate_remote_runtimes(
            runtimes,
            expected_plan_sha256=expected_plan_sha256,
            expected_query_artifact_sha256=expected_query_artifact_sha256,
            expected_base_catalog_sha256=expected_base_catalog_sha256,
            expected_output_catalog_sha256=expected_output_catalog_sha256,
            remote_files=remote_files,
        )
    except (ArtifactFormatError, OSError, ValueError) as error:
        raise PortfolioInputError(
            "Portfolio remote-processing runtime verification failed"
        ) from error

    after = _read_regular_files(snapshot_paths)
    image_after = _read_regular_files(image_paths)
    if after != before or image_after != image_before:
        raise PortfolioInputError("Portfolio inputs changed during verified loading")
    assistant_queries = tuple(
        build_assistant_query_input(query) for query in accepted_queries
    )
    return _LoadedPortfolioState(
        plan=plan,
        plan_manifest=plan_manifest,
        ledger=ledger,
        queries=accepted_queries,
        assistant_queries=assistant_queries,
        query_assets=tuple(query_assets),
        remote_runtimes=runtimes,
        snapshots=(
            *tuple(zip(snapshot_paths, before, strict=True)),
            *tuple(zip(image_paths, image_before, strict=True)),
        ),
    )


def _validate_portfolio_semantics(
    *,
    plan: CorpusPlan,
    plan_manifest: PlanManifest,
    ledger: tuple[AcceptedLedgerEntry, ...],
    queries: tuple[Query, ...],
    assignments: tuple[CapabilityAssignment, ...],
    expected_plan_sha256: str,
    expected_seed_set_sha256: str,
    expected_capability_assignments_sha256: str,
    expected_base_catalog_sha256: str,
) -> None:
    if (
        plan.scope != "dev_mini"
        or plan_manifest.scope != "dev_mini"
        or len(plan.queries) != PORTFOLIO_DEV_MINI_COUNT
        or len(queries) != PORTFOLIO_DEV_MINI_COUNT
        or plan_manifest.count != PORTFOLIO_DEV_MINI_COUNT
    ):
        raise PortfolioInputError(
            "Portfolio loader requires the exact 200-query dev_mini plan"
        )
    if (
        plan_manifest.plan_sha256 != expected_plan_sha256
        or plan.asset_catalog_sha256 != expected_base_catalog_sha256
        or plan_manifest.asset_catalog_sha256 != expected_base_catalog_sha256
        or plan.capability_assignments_sha256 != expected_capability_assignments_sha256
        or plan_manifest.capability_assignments_sha256
        != expected_capability_assignments_sha256
    ):
        raise PortfolioInputError(
            "Portfolio plan/manifest external bindings are inconsistent"
        )

    planned_batches = tuple(dict.fromkeys(item.batch_id for item in plan.queries))
    if (
        len(ledger) != PORTFOLIO_DEV_MINI_BATCH_COUNT
        or tuple(item.base_batch_id for item in ledger) != planned_batches
        or any(item.count != PORTFOLIO_DEV_MINI_BATCH_SIZE for item in ledger)
        or sum(item.count for item in ledger) != PORTFOLIO_DEV_MINI_COUNT
        or any(item.plan_sha256 != expected_plan_sha256 for item in ledger)
        or any(item.seed_set_sha256 != expected_seed_set_sha256 for item in ledger)
    ):
        raise PortfolioInputError(
            "Portfolio accepted ledger is not the exact 8x25 plan prefix"
        )

    expected_query_ids = tuple(item.plan_id for item in plan.queries)
    actual_query_ids = tuple(item.query_id for item in queries)
    if actual_query_ids != expected_query_ids or len(actual_query_ids) != len(
        set(actual_query_ids)
    ):
        raise PortfolioInputError(
            "Portfolio accepted query order differs from the plan"
        )
    ledger_batch_by_base = {item.base_batch_id: item.batch_id for item in ledger}
    for planned, query in zip(plan.queries, queries, strict=True):
        _validate_planned_query_binding(
            planned,
            query,
            accepted_batch_id=ledger_batch_by_base[planned.batch_id],
            expected_seed_set_sha256=expected_seed_set_sha256,
        )

    capability_counts = Counter(query.canonical_capability for query in queries)
    if capability_counts != Counter(DEV_MINI_CAPABILITY_COUNTS):
        raise PortfolioInputError("Portfolio accepted capability distribution drifted")
    indexed_assignments = {
        (
            item.asset_id,
            item.image_path,
            item.canonical_intent,
        ): item
        for item in assignments
    }
    if len(indexed_assignments) != len(assignments):
        raise PortfolioInputError("Portfolio capability assignments are duplicated")
    for planned in plan.queries:
        assignment = indexed_assignments.get(
            (
                planned.asset_id,
                planned.image_path,
                planned.canonical_intent,
            )
        )
        if assignment is None or (
            planned.canonical_capability != assignment.canonical_capability
            or tuple(planned.acceptable_capabilities)
            != assignment.acceptable_capabilities
            or planned.requires_card != assignment.requires_card
            or planned.capability_assignment_id != assignment.assignment_id
            or planned.capability_assignment_source_sha256
            != assignment.source_artifact_sha256
        ):
            raise PortfolioInputError(
                f"Portfolio plan drifted from capability assignment: {planned.plan_id}"
            )


def _validate_planned_query_binding(
    planned: PlannedQuery,
    query: Query,
    *,
    accepted_batch_id: str,
    expected_seed_set_sha256: str,
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
        or query.split != planned.provisional_split
        or query.synthesis_batch_id != accepted_batch_id
        or query.synthesis_prompt_id != planned.plan_id
        or query.seed_set_sha256 != expected_seed_set_sha256
    ):
        raise PortfolioInputError(
            f"accepted query drifted from plan/ledger/seed: {query.query_id}"
        )


def _validate_remote_runtimes(
    runtimes: tuple[VerifiedPortfolioRemoteProcessingRuntime, ...],
    *,
    expected_plan_sha256: str,
    expected_query_artifact_sha256: str,
    expected_base_catalog_sha256: str,
    expected_output_catalog_sha256: str,
    remote_files: PortfolioRemoteProcessingFiles,
) -> None:
    if tuple(item.processor for item in runtimes) != remote_files.processor_order:
        raise PortfolioInputError(
            "Portfolio remote runtimes do not cover the exact processor order"
        )
    for runtime in runtimes:
        require_verified_portfolio_remote_processing_runtime(
            runtime,
            processor=runtime.processor,
            catalog_sha256=expected_output_catalog_sha256,
        )
        if (
            runtime.plan_sha256 != expected_plan_sha256
            or runtime.query_artifact_sha256 != expected_query_artifact_sha256
            or runtime.catalog.catalog_sha256 != expected_output_catalog_sha256
            or runtime.receipt.base_catalog_sha256 != expected_base_catalog_sha256
            or runtime.authorization_file_sha256
            != remote_files.expected_authorization_file_sha256
            or runtime.receipt_file_sha256 != remote_files.expected_receipt_file_sha256
        ):
            raise PortfolioInputError(
                f"Portfolio remote runtime binding drifted: {runtime.processor}"
            )


def _remote_runtime_identities(
    runtimes: tuple[VerifiedPortfolioRemoteProcessingRuntime, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            runtime.processor,
            sha256_bytes(
                canonical_json_bytes(runtime.authorization.model_dump(mode="json"))
            ),
            runtime.authorization_file_sha256,
            runtime.receipt_file_sha256,
            sha256_bytes(canonical_json_bytes(runtime.receipt.model_dump(mode="json"))),
            runtime.dataset_assets_sha256,
            runtime.plan_sha256,
            runtime.query_artifact_sha256,
            _catalog_runtime_identity(runtime),
        )
        for runtime in runtimes
    )


def _catalog_runtime_identity(
    runtime: VerifiedPortfolioRemoteProcessingRuntime,
) -> tuple[object, ...]:
    catalog = runtime.catalog

    def resolution_identity(value) -> tuple[str, str]:
        return (
            value.asset.asset_id,
            value.leakage_group_id,
        )

    return (
        catalog.root.absolute(),
        catalog.asset_root.absolute(),
        sha256_bytes(canonical_json_bytes(catalog.manifest.model_dump(mode="json"))),
        sha256_bytes(
            canonical_jsonl_bytes(
                tuple(item.model_dump(mode="json") for item in catalog.assets)
            )
        ),
        sha256_bytes(
            canonical_jsonl_bytes(
                tuple(item.model_dump(mode="json") for item in catalog.components)
            )
        ),
        tuple(
            (key, *resolution_identity(value))
            for key, value in sorted(catalog._by_asset_id.items())
        ),
        tuple(
            (key, *resolution_identity(value))
            for key, value in sorted(catalog._by_path.items())
        ),
        tuple(
            (
                key,
                tuple(item.asset_id for item in value),
            )
            for key, value in sorted(catalog._members_by_component.items())
        ),
        catalog._files_verified,
    )


def _parse_ledger(content: bytes) -> tuple[AcceptedLedgerEntry, ...]:
    entries: list[AcceptedLedgerEntry] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        try:
            entries.append(AcceptedLedgerEntry.model_validate_json(line, strict=True))
        except ValidationError as error:
            raise PortfolioInputError(
                f"Portfolio accepted ledger line {line_number} is invalid"
            ) from error
    if not entries or content != canonical_jsonl_bytes(entries):
        raise PortfolioInputError(
            "Portfolio accepted ledger must be non-empty canonical JSONL"
        )
    return tuple(entries)


def _normalize_remote_files(
    value: PortfolioRemoteProcessingFiles,
) -> PortfolioRemoteProcessingFiles:
    if type(value) is not PortfolioRemoteProcessingFiles:
        raise TypeError("remote_files must be PortfolioRemoteProcessingFiles")
    return PortfolioRemoteProcessingFiles(
        authorization_file=Path(value.authorization_file).absolute(),
        expected_authorization_file_sha256=_require_sha256(
            value.expected_authorization_file_sha256,
            "expected remote authorization file SHA-256",
        ),
        receipt_file=Path(value.receipt_file).absolute(),
        expected_receipt_file_sha256=_require_sha256(
            value.expected_receipt_file_sha256,
            "expected remote receipt file SHA-256",
        ),
        selection_manifest=Path(value.selection_manifest).absolute(),
        dataset_assets=Path(value.dataset_assets).absolute(),
        base_catalog_dir=Path(value.base_catalog_dir).absolute(),
        output_catalog_dir=Path(value.output_catalog_dir).absolute(),
        asset_root=Path(value.asset_root).absolute(),
        processor_order=_validate_processor_order(value.processor_order),
    )


def _validate_processor_order(
    value: tuple[PortfolioProcessor, ...],
) -> tuple[PortfolioProcessor, ...]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("Portfolio processor_order must contain exactly three items")
    allowed = {
        "aifast-gemini-feedback",
        "aifast-gemini-judge",
        "dashscope-kimi-feedback",
        "dashscope-kimi-judge",
        "dashscope-qwen-assistant",
    }
    evaluator_pair = value[1:] if len(value) == 3 else ()
    if (
        any(type(item) is not str or item not in allowed for item in value)
        or len(set(value)) != len(value)
        or value[0] != "dashscope-qwen-assistant"
        or evaluator_pair
        not in {
            ("dashscope-kimi-feedback", "dashscope-kimi-judge"),
            ("aifast-gemini-feedback", "dashscope-kimi-judge"),
            ("dashscope-kimi-feedback", "aifast-gemini-judge"),
        }
    ):
        raise ValueError(
            "Portfolio processor_order must be Assistant, one Feedback, then Judge"
        )
    return value


def _read_verified_query_images(
    bindings: tuple[PortfolioQueryAssetBinding, ...],
    *,
    asset_root: Path,
) -> tuple[tuple[Path, ...], tuple[bytes, ...]]:
    expected_by_path: dict[Path, str] = {}
    for binding in bindings:
        path = (asset_root / Path(binding.image_path)).absolute()
        previous = expected_by_path.setdefault(path, binding.image_sha256)
        if previous != binding.image_sha256:
            raise PortfolioInputError(
                f"Portfolio image path has conflicting hashes: {binding.image_path}"
            )
    paths = tuple(expected_by_path)
    contents = _read_regular_files(paths)
    for path, content in zip(paths, contents, strict=True):
        if sha256_bytes(content) != expected_by_path[path]:
            raise PortfolioInputError(
                f"Portfolio query image SHA-256 mismatch: {path.name}"
            )
    return paths, contents


def _read_regular_files(paths: tuple[Path, ...]) -> tuple[bytes, ...]:
    values: list[bytes] = []
    for path in paths:
        try:
            values.append(
                read_stable_regular_file(
                    path,
                    label=f"Portfolio input {path.name}",
                    max_bytes=128 * 1024 * 1024,
                )
            )
        except ArtifactFormatError as error:
            raise PortfolioInputError(str(error)) from error
    return tuple(values)


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    absolute = tuple(path.absolute() for path in paths)
    if len(absolute) != len(set(absolute)):
        raise PortfolioInputError(
            "Portfolio input roles must use distinct regular files"
        )
    return absolute


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


__all__ = [
    "ACTIVE_PORTFOLIO_PROCESSOR_ORDER",
    "CORE_PORTFOLIO_PROCESSOR_ORDER",
    "PORTFOLIO_DEV_MINI_BATCH_COUNT",
    "PORTFOLIO_DEV_MINI_BATCH_SIZE",
    "PORTFOLIO_DEV_MINI_COUNT",
    "PORTFOLIO_PROCESSOR_ORDER",
    "ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER",
    "PortfolioInputError",
    "PortfolioQueryAssetBinding",
    "PortfolioRemoteProcessingFiles",
    "VerifiedPortfolioDevMiniInputs",
    "load_verified_portfolio_dev_mini_inputs",
    "require_verified_portfolio_dev_mini_inputs",
]
