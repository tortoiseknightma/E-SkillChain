"""Verified local tool sources for the Portfolio Core runtime.

The historical Portfolio runtime was assembled around ``dev_mini`` paths.
Core v9 has a different selection/catalog lineage and does not publish the
RPC ``scenes.jsonl`` or the Core-name iNaturalist manifest expected by that
runtime.  This module derives only those local projections, binds every input
to the already verified Core remote-processing chain (or an explicit digest),
and publishes a create-only receipt.  It performs no provider calls.

The private selection, annotation and label-bearing artifacts stay inside the
local tool runtime.  They are never added to the Assistant query DTO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import stat
from typing import Any
import zipfile

from skillchain.data import rpc as rpc_source
from skillchain.data.source_lock import SourceLockError, stable_file_digest
from skillchain.evaluation.portfolio_core_inputs import (
    PortfolioCoreInputError,
    VerifiedPortfolioCoreInputs,
    require_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_inputs import (
    CORE_PORTFOLIO_PROCESSOR_ORDER,
    ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
)
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.portfolio_runtime import (
    PORTFOLIO_TOOL_RUNTIME_POLICY,
    PortfolioRuntimeSources,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    parse_strict_json,
    read_stable_regular_file,
    sha256_bytes,
)


CORE_RUNTIME_SOURCE_POLICY = "portfolio-core-runtime-sources-v1"
CORE_RUNTIME_SOURCE_RECEIPT = "receipt.json"
CORE_RUNTIME_RPC_SCENES = "rpc-scenes.jsonl"
CORE_RUNTIME_INATURALIST = "inaturalist-runtime.jsonl"
CORE_RUNTIME_RECIPES = "recipe-evidence-runtime.jsonl"
CORE_RUNTIME_CAPTION_DIR = "fashioniq-captions"
CORE_RUNTIME_STYLE_COORDINATION_GRAPH = "style-coordination-graph.json"
_RPC_SOURCE = "rpc"
_INATURALIST_SOURCE = "inaturalist"
_RECIPE_SOURCE = "isia_food500"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SELECTION_BYTES = 32 * 1024 * 1024
_MAX_DATASET_ASSETS_BYTES = 64 * 1024 * 1024
_MAX_CATALOG_ASSETS_BYTES = 64 * 1024 * 1024
_MAX_AUXILIARY_BYTES = 128 * 1024 * 1024
_STYLE_COORDINATION_GRAPH_KIND = "portfolio-style-coordination-graph"
_STYLE_COORDINATION_GRAPH_POLICY = "portfolio-style-coordination-graph-v1"
_STYLE_COORDINATION_ANNOTATION_POLICY = (
    "portfolio-style-coordination-candidate-review-v1"
)
_STYLE_COORDINATION_FORBIDDEN_FIELDS = frozenset(
    {
        "config",
        "final_score",
        "query_id",
        "query_text",
        "score",
        "split",
        "user_text",
    }
)
_VERIFIED_CORE_RUNTIME_SOURCES = object()


class PortfolioCoreRuntimeSourceError(ValueError):
    """The Core local runtime-source chain is missing or inconsistent."""


@dataclass(frozen=True)
class BoundPortfolioRuntimeFile:
    """One optional runtime source with an externally frozen digest."""

    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class PortfolioCoreRuntimeAuxiliaryFiles:
    """Local sources not carried directly by ``VerifiedPortfolioCoreInputs``."""

    rpc_archive: Path
    rpc_adapter_manifest: Path
    expected_rpc_adapter_manifest_sha256: str
    expected_rpc_annotation_sha256: str
    inaturalist_manifest: Path
    expected_inaturalist_manifest_sha256: str
    recipe_evidence: Path
    expected_recipe_evidence_sha256: str
    fashioniq_captions: tuple[BoundPortfolioRuntimeFile, ...] = ()
    verify_rpc_archive_full_sha256: bool = False
    rpc_annotation_file: Path | None = None
    style_coordination_graph: BoundPortfolioRuntimeFile | None = None


@dataclass(frozen=True)
class VerifiedPortfolioCoreRuntimeSources:
    """A create-only Core source projection ready for the local tool runtime."""

    sources: PortfolioRuntimeSources
    output_dir: Path
    receipt_path: Path
    receipt_file_sha256: str
    source_file_sha256s: tuple[str, ...]
    rpc_scene_count: int
    inaturalist_row_count: int
    recipe_row_count: int
    provider_call_count: int = 0
    formal_eligible: bool = False
    _marker: object = field(repr=False, compare=False, default=None)


@dataclass(frozen=True)
class _PrimarySources:
    selection_path: Path
    selection_content: bytes
    selection: dict[str, Any]
    dataset_assets_path: Path
    dataset_assets_content: bytes
    runtime_catalog_assets_path: Path
    runtime_catalog_assets_content: bytes
    asset_root: Path
    core_catalog_sha256: str
    remote_receipt_file_sha256: str


@dataclass(frozen=True)
class _DerivedSources:
    rpc_scenes: bytes
    inaturalist: bytes
    recipes: bytes
    rpc_scene_count: int
    inaturalist_row_count: int
    recipe_row_count: int
    fashioniq_captions: tuple[tuple[str, bytes], ...]
    style_coordination_graph: bytes | None
    input_records: dict[str, Any]


def materialize_verified_portfolio_core_runtime_sources(
    core_inputs: VerifiedPortfolioCoreInputs,
    auxiliary: PortfolioCoreRuntimeAuxiliaryFiles,
    *,
    output_dir: str | Path,
) -> VerifiedPortfolioCoreRuntimeSources:
    """Verify and publish Core tool-runtime sources without model calls.

    ``output_dir`` is create-only.  Existing output must be loaded through
    :func:`load_verified_portfolio_core_runtime_sources` with its external
    receipt digest rather than overwritten.
    """

    normalized = _normalize_auxiliary(auxiliary)
    # Reject malformed cheap bindings before rebuilding the comparatively
    # expensive 1,500-query Core proof.
    _preflight_auxiliary_files(normalized)
    verified = _require_core_inputs(core_inputs)
    primary = _primary_sources(verified)
    _assert_opaque_assistant_projection(verified)
    derived = _derive_sources(primary, normalized)

    destination = Path(output_dir).absolute()
    staging = new_staging_directory(destination)
    try:
        payloads = {
            CORE_RUNTIME_RPC_SCENES: derived.rpc_scenes,
            CORE_RUNTIME_INATURALIST: derived.inaturalist,
            CORE_RUNTIME_RECIPES: derived.recipes,
            **{
                f"{CORE_RUNTIME_CAPTION_DIR}/{name}": content
                for name, content in derived.fashioniq_captions
            },
        }
        if derived.style_coordination_graph is not None:
            payloads[CORE_RUNTIME_STYLE_COORDINATION_GRAPH] = (
                derived.style_coordination_graph
            )
        for name, content in payloads.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        receipt = _receipt(
            verified=verified,
            primary=primary,
            auxiliary=normalized,
            derived=derived,
            payloads=payloads,
        )
        receipt_bytes = canonical_json_bytes(receipt)
        (staging / CORE_RUNTIME_SOURCE_RECEIPT).write_bytes(receipt_bytes)
        atomic_publish_new_directory(staging, destination)
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise

    return _load_materialized(
        verified,
        normalized,
        destination,
        expected_receipt_file_sha256=sha256_bytes(receipt_bytes),
        preverified_derived=derived,
    )


def load_verified_portfolio_core_runtime_sources(
    core_inputs: VerifiedPortfolioCoreInputs,
    auxiliary: PortfolioCoreRuntimeAuxiliaryFiles | None = None,
    *,
    output_dir: str | Path,
    expected_receipt_file_sha256: str,
    _allow_legacy_tool_runtime_policy: bool = False,
) -> VerifiedPortfolioCoreRuntimeSources:
    """Strictly reload an existing Core runtime-source projection."""

    verified = _require_core_inputs(core_inputs)
    normalized = None if auxiliary is None else _normalize_auxiliary(auxiliary)
    _assert_opaque_assistant_projection(verified)
    return _load_materialized(
        verified,
        normalized,
        Path(output_dir).absolute(),
        expected_receipt_file_sha256=_require_sha256(
            expected_receipt_file_sha256,
            "runtime-source receipt file SHA-256",
        ),
        _allow_legacy_tool_runtime_policy=_allow_legacy_tool_runtime_policy,
    )


def require_verified_portfolio_core_runtime_sources(
    value: object,
) -> VerifiedPortfolioCoreRuntimeSources:
    """Reject forged handles and drifted materialized source bytes."""

    if (
        type(value) is not VerifiedPortfolioCoreRuntimeSources
        or value._marker is not _VERIFIED_CORE_RUNTIME_SOURCES
        or value.provider_call_count != 0
        or value.formal_eligible is not False
    ):
        raise TypeError("Core tool runtime requires a verified source handle")
    observed = tuple(_stable_sha(path, path.name) for path in value.sources.files())
    if observed != value.source_file_sha256s:
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime source bytes changed after verification"
        )
    if _stable_sha(value.receipt_path, "Core runtime-source receipt") != (
        value.receipt_file_sha256
    ):
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source receipt changed after verification"
        )
    return value


def _require_core_inputs(
    value: VerifiedPortfolioCoreInputs,
) -> VerifiedPortfolioCoreInputs:
    try:
        verified = require_verified_portfolio_core_inputs(value)
    except (PortfolioCoreInputError, TypeError, ValueError) as error:
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime sources require the verified Core input chain"
        ) from error
    runtimes = verified.remote_runtimes
    if tuple(item.processor for item in runtimes) not in {
        CORE_PORTFOLIO_PROCESSOR_ORDER,
        ROLE_SWAPPED_CORE_PORTFOLIO_PROCESSOR_ORDER,
    }:
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime sources require one complete schema-versioned processor set"
        )
    return verified


def _normalize_auxiliary(
    value: PortfolioCoreRuntimeAuxiliaryFiles,
) -> PortfolioCoreRuntimeAuxiliaryFiles:
    if not isinstance(value, PortfolioCoreRuntimeAuxiliaryFiles):
        raise TypeError("auxiliary sources require PortfolioCoreRuntimeAuxiliaryFiles")
    captions = tuple(
        BoundPortfolioRuntimeFile(
            path=Path(item.path).absolute(),
            expected_sha256=_require_sha256(
                item.expected_sha256, "FashionIQ caption SHA-256"
            ),
        )
        for item in value.fashioniq_captions
    )
    caption_paths = tuple(item.path for item in captions)
    if caption_paths != tuple(sorted(set(caption_paths), key=str)):
        raise PortfolioCoreRuntimeSourceError(
            "FashionIQ caption paths must be sorted and unique"
        )
    caption_names = tuple(item.path.name for item in captions)
    if len(set(caption_names)) != len(caption_names):
        raise PortfolioCoreRuntimeSourceError(
            "FashionIQ caption file names must be unique"
        )
    if type(value.verify_rpc_archive_full_sha256) is not bool:
        raise PortfolioCoreRuntimeSourceError(
            "verify_rpc_archive_full_sha256 must be boolean"
        )
    coordination_graph = (
        None
        if value.style_coordination_graph is None
        else BoundPortfolioRuntimeFile(
            path=Path(value.style_coordination_graph.path).absolute(),
            expected_sha256=_require_sha256(
                value.style_coordination_graph.expected_sha256,
                "style coordination graph SHA-256",
            ),
        )
    )
    return PortfolioCoreRuntimeAuxiliaryFiles(
        rpc_archive=Path(value.rpc_archive).absolute(),
        rpc_adapter_manifest=Path(value.rpc_adapter_manifest).absolute(),
        expected_rpc_adapter_manifest_sha256=_require_sha256(
            value.expected_rpc_adapter_manifest_sha256,
            "RPC adapter manifest SHA-256",
        ),
        expected_rpc_annotation_sha256=_require_sha256(
            value.expected_rpc_annotation_sha256,
            "RPC annotation SHA-256",
        ),
        inaturalist_manifest=Path(value.inaturalist_manifest).absolute(),
        expected_inaturalist_manifest_sha256=_require_sha256(
            value.expected_inaturalist_manifest_sha256,
            "iNaturalist manifest SHA-256",
        ),
        recipe_evidence=Path(value.recipe_evidence).absolute(),
        expected_recipe_evidence_sha256=_require_sha256(
            value.expected_recipe_evidence_sha256,
            "recipe evidence SHA-256",
        ),
        fashioniq_captions=captions,
        style_coordination_graph=coordination_graph,
        verify_rpc_archive_full_sha256=value.verify_rpc_archive_full_sha256,
        rpc_annotation_file=(
            None
            if value.rpc_annotation_file is None
            else Path(value.rpc_annotation_file).absolute()
        ),
    )


def _read_style_coordination_graph(
    bound: BoundPortfolioRuntimeFile,
) -> tuple[bytes, dict[str, Any]]:
    content = _read(
        bound.path,
        "style coordination graph",
        max_bytes=_MAX_AUXILIARY_BYTES,
    )
    if sha256_bytes(content) != bound.expected_sha256:
        raise PortfolioCoreRuntimeSourceError(
            "style coordination graph SHA-256 mismatch"
        )
    try:
        value = parse_canonical_json(content, label="style coordination graph")
    except ArtifactFormatError as error:
        raise PortfolioCoreRuntimeSourceError(str(error)) from error
    if not isinstance(value, dict):
        raise PortfolioCoreRuntimeSourceError(
            "style coordination graph must be one canonical JSON object"
        )
    _validate_style_coordination_graph(value)
    return content, value


def _validate_style_coordination_graph(value: dict[str, Any]) -> None:
    _reject_style_coordination_forbidden_fields(value)
    expected_top_level = {
        "edges",
        "graph_sha256",
        "kind",
        "policy_version",
        "schema_version",
        "source_bindings",
    }
    if set(value) != expected_top_level:
        raise PortfolioCoreRuntimeSourceError(
            "style coordination graph top-level schema is invalid"
        )
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("kind") != _STYLE_COORDINATION_GRAPH_KIND
        or value.get("policy_version") != _STYLE_COORDINATION_GRAPH_POLICY
    ):
        raise PortfolioCoreRuntimeSourceError(
            "style coordination graph identity or policy is invalid"
        )

    source_bindings = value.get("source_bindings")
    expected_source_bindings = {
        "abo_listings_archive_sha256",
        "candidate_seed_sha256",
        "runtime_catalog_assets_sha256",
        "selection_manifest_sha256",
    }
    if not isinstance(source_bindings, dict) or set(source_bindings) != (
        expected_source_bindings
    ):
        raise PortfolioCoreRuntimeSourceError(
            "style coordination graph source bindings are invalid"
        )
    for name in sorted(expected_source_bindings):
        _require_sha256(source_bindings.get(name), f"style graph {name}")

    declared_graph_sha256 = _require_sha256(
        value.get("graph_sha256"), "style coordination graph self SHA-256"
    )
    unsigned = dict(value)
    unsigned.pop("graph_sha256")
    if declared_graph_sha256 != sha256_bytes(canonical_json_bytes(unsigned)):
        raise PortfolioCoreRuntimeSourceError(
            "style coordination graph self hash is invalid"
        )

    edges = value.get("edges")
    if not isinstance(edges, list) or not edges:
        raise PortfolioCoreRuntimeSourceError(
            "style coordination graph must contain non-empty edges"
        )
    edge_ids: set[str] = set()
    previous_edge_id = ""
    for edge_index, edge in enumerate(edges):
        _validate_style_coordination_edge(edge, edge_index=edge_index)
        edge_id = edge["edge_id"]
        if edge_id in edge_ids or edge_id <= previous_edge_id:
            raise PortfolioCoreRuntimeSourceError(
                "style coordination graph edge_id is duplicated or unsorted"
            )
        edge_ids.add(edge_id)
        previous_edge_id = edge_id


def _validate_style_coordination_edge(value: object, *, edge_index: int) -> None:
    expected_edge_keys = {
        "anchor",
        "annotation_policy_version",
        "candidate",
        "confidence",
        "edge_id",
        "facets",
        "relation_kind",
    }
    if not isinstance(value, dict) or set(value) != expected_edge_keys:
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} schema is invalid"
        )
    _require_nonempty_string(value.get("edge_id"), f"style graph edge {edge_index} id")
    if value.get("relation_kind") != "portfolio_curated_coordination_rule":
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} relation is invalid"
        )
    if value.get("annotation_policy_version") != (
        _STYLE_COORDINATION_ANNOTATION_POLICY
    ):
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} annotation policy is invalid"
        )
    _require_unit_confidence(
        value.get("confidence"), f"style graph edge {edge_index} confidence"
    )

    anchor = value.get("anchor")
    expected_anchor_keys = {
        "asset_id",
        "category_l1",
        "color_families",
        "image_path",
        "image_sha256",
        "product_id",
        "source_dataset",
        "source_record_id",
    }
    if not isinstance(anchor, dict) or set(anchor) != expected_anchor_keys:
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} anchor schema is invalid"
        )
    for name in (
        "asset_id",
        "category_l1",
        "image_path",
        "product_id",
        "source_dataset",
        "source_record_id",
    ):
        _require_nonempty_string(
            anchor.get(name), f"style graph edge {edge_index} anchor {name}"
        )
    _require_sha256(
        anchor.get("image_sha256"),
        f"style graph edge {edge_index} anchor image SHA-256",
    )
    _require_nonempty_unique_strings(
        anchor.get("color_families"),
        f"style graph edge {edge_index} anchor color_families",
    )

    candidate = value.get("candidate")
    expected_candidate_keys = {
        "asset_id",
        "audience",
        "category_l1",
        "color_families",
        "display_title",
        "feature_tags",
        "image_path",
        "image_sha256",
        "product_id",
        "source_dataset",
        "source_record_id",
    }
    if not isinstance(candidate, dict) or set(candidate) != expected_candidate_keys:
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} candidate schema is invalid"
        )
    for name in (
        "asset_id",
        "category_l1",
        "display_title",
        "image_path",
        "product_id",
        "source_dataset",
        "source_record_id",
    ):
        _require_nonempty_string(
            candidate.get(name), f"style graph edge {edge_index} candidate {name}"
        )
    if candidate.get("audience") not in {"men", "unisex", "women"}:
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} candidate audience is invalid"
        )
    _require_sha256(
        candidate.get("image_sha256"),
        f"style graph edge {edge_index} candidate image SHA-256",
    )
    for name in ("feature_tags", "color_families"):
        _require_nonempty_unique_strings(
            candidate.get(name),
            f"style graph edge {edge_index} candidate {name}",
        )
    if (
        candidate["audience"] not in {"women", "unisex"}
        or candidate["category_l1"] not in {"footwear", "bag", "jewelry"}
        or candidate["category_l1"] == anchor["category_l1"]
    ):
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} candidate scope is invalid"
        )

    facets = value.get("facets")
    if not isinstance(facets, list) or not facets:
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} facets are invalid"
        )
    seen_facets: set[tuple[str, str]] = set()
    normalized_facets: list[dict[str, object]] = []
    for facet_index, facet in enumerate(facets):
        if not isinstance(facet, dict) or set(facet) != {
            "confidence",
            "facet",
            "value",
        }:
            raise PortfolioCoreRuntimeSourceError(
                f"style coordination graph edge {edge_index} facet schema is invalid"
            )
        facet_name = _require_nonempty_string(
            facet.get("facet"),
            f"style graph edge {edge_index} facet {facet_index} name",
        )
        facet_value = _require_nonempty_string(
            facet.get("value"),
            f"style graph edge {edge_index} facet {facet_index} value",
        )
        _require_unit_confidence(
            facet.get("confidence"),
            f"style graph edge {edge_index} facet {facet_index} confidence",
        )
        identity = (facet_name, facet_value)
        if identity in seen_facets:
            raise PortfolioCoreRuntimeSourceError(
                f"style coordination graph edge {edge_index} facet is duplicated"
            )
        seen_facets.add(identity)
        normalized_facets.append(facet)

    expected_names = [
        "category",
        "palette",
        "verified_attributes",
        "coordination_rule",
    ]
    facets_by_name = {str(facet["facet"]): facet for facet in normalized_facets}
    rule = facets_by_name.get("coordination_rule", {}).get("value")
    if (
        [facet["facet"] for facet in normalized_facets] != expected_names
        or facets_by_name["category"]["value"] != candidate["category_l1"]
        or facets_by_name["category"]["confidence"] != 1.0
        or facets_by_name["palette"]["value"] != ",".join(candidate["color_families"])
        or facets_by_name["palette"]["confidence"] != 0.95
        or facets_by_name["verified_attributes"]["value"]
        != ",".join(candidate["feature_tags"])
        or facets_by_name["verified_attributes"]["confidence"] != 0.95
        or rule not in {"neutral_palette_rule", "compatible_palette_rule"}
        or facets_by_name["coordination_rule"]["confidence"] != value["confidence"]
    ):
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} facet contract is invalid"
        )
    expected_edge_id = "style.edge.v1." + sha256_bytes(
        canonical_json_bytes(
            {
                "anchor_asset_id": anchor["asset_id"],
                "annotation_policy_version": value["annotation_policy_version"],
                "candidate_asset_id": candidate["asset_id"],
                "relation_kind": value["relation_kind"],
                "rule": rule,
            }
        )
    )
    if value["edge_id"] != expected_edge_id:
        raise PortfolioCoreRuntimeSourceError(
            f"style coordination graph edge {edge_index} identity is invalid"
        )


def _reject_style_coordination_forbidden_fields(value: object) -> None:
    if isinstance(value, dict):
        for name, nested in value.items():
            if name.casefold() in _STYLE_COORDINATION_FORBIDDEN_FIELDS:
                raise PortfolioCoreRuntimeSourceError(
                    f"style coordination graph contains forbidden field: {name}"
                )
            _reject_style_coordination_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_style_coordination_forbidden_fields(nested)


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PortfolioCoreRuntimeSourceError(f"{label} must be a non-empty string")
    return value


def _require_nonempty_unique_strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PortfolioCoreRuntimeSourceError(
            f"{label} must be a non-empty string array"
        )
    normalized = tuple(
        _require_nonempty_string(item, f"{label} item") for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise PortfolioCoreRuntimeSourceError(f"{label} must be unique")
    return normalized


def _require_unit_confidence(value: object, label: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise PortfolioCoreRuntimeSourceError(f"{label} must be within [0, 1]")
    return float(value)


def _preflight_auxiliary_files(
    auxiliary: PortfolioCoreRuntimeAuxiliaryFiles,
) -> None:
    """Reject cheap auxiliary drift before rebuilding the Core proof.

    The full derivation intentionally rereads these files after Core
    verification so that selection-dependent mappings and TOCTOU drift still
    fail closed.  For the real Core path the large RPC archive is only
    inspected with ``lstat`` here; annotation semantics come from the
    separately hash-bound extracted member.
    """

    if auxiliary.style_coordination_graph is not None:
        _read_style_coordination_graph(auxiliary.style_coordination_graph)

    manifest_content = _read(
        auxiliary.rpc_adapter_manifest,
        "Core RPC adapter manifest",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(manifest_content) != auxiliary.expected_rpc_adapter_manifest_sha256:
        raise PortfolioCoreRuntimeSourceError("RPC adapter manifest SHA-256 mismatch")
    try:
        manifest = parse_canonical_json(
            manifest_content, label="Core RPC adapter manifest"
        )
    except ArtifactFormatError as error:
        raise PortfolioCoreRuntimeSourceError(str(error)) from error
    archives = manifest.get("archives") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("source_id") != _RPC_SOURCE
        or not isinstance(manifest.get("source_revision"), str)
        or type(manifest.get("include_count")) is not int
        or manifest["include_count"] <= 0
        or not isinstance(archives, list)
        or len(archives) != 1
        or not isinstance(archives[0], dict)
    ):
        raise PortfolioCoreRuntimeSourceError("RPC adapter manifest schema is invalid")
    archive_record = archives[0]
    declared_archive_sha256 = _require_sha256(
        archive_record.get("sha256"), "RPC archive SHA-256"
    )
    declared_archive_bytes = archive_record.get("bytes")
    if type(declared_archive_bytes) is not int or declared_archive_bytes <= 0:
        raise PortfolioCoreRuntimeSourceError("RPC archive byte count is invalid")
    if (
        _regular_file_snapshot(auxiliary.rpc_archive, "Core RPC archive")[2]
        != declared_archive_bytes
    ):
        raise PortfolioCoreRuntimeSourceError(
            "RPC archive size differs from the selected-source adapter"
        )
    if not _SHA256.fullmatch(declared_archive_sha256):  # pragma: no cover
        raise PortfolioCoreRuntimeSourceError("RPC archive SHA-256 is invalid")

    try:
        if auxiliary.rpc_annotation_file is not None:
            annotation_content = _read(
                auxiliary.rpc_annotation_file,
                "externally hash-bound Core RPC validation annotations",
                max_bytes=_MAX_AUXILIARY_BYTES,
            )
        else:
            with zipfile.ZipFile(auxiliary.rpc_archive) as archive:
                members = rpc_source._validate_archive_members(archive)
                annotation_content, _paths, _roots, _canonical = (
                    rpc_source._load_annotation_replicas(archive, members)
                )
        if sha256_bytes(annotation_content) != auxiliary.expected_rpc_annotation_sha256:
            raise PortfolioCoreRuntimeSourceError(
                "RPC annotation member SHA-256 mismatch"
            )
        annotation = parse_strict_json(
            annotation_content, label="Core RPC validation annotations"
        )
        rpc_source._validate_annotations(annotation)
    except (ArtifactFormatError, OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, PortfolioCoreRuntimeSourceError):
            raise
        raise PortfolioCoreRuntimeSourceError(
            "RPC annotation member is invalid"
        ) from error

    inaturalist_content = _read(
        auxiliary.inaturalist_manifest,
        "Core iNaturalist manifest",
        max_bytes=_MAX_AUXILIARY_BYTES,
    )
    if (
        sha256_bytes(inaturalist_content)
        != auxiliary.expected_inaturalist_manifest_sha256
    ):
        raise PortfolioCoreRuntimeSourceError("iNaturalist manifest SHA-256 mismatch")
    inaturalist_rows, _ignored = _parse_strict_jsonl(
        inaturalist_content, label="Core iNaturalist manifest"
    )
    photo_ids: set[int] = set()
    for row in inaturalist_rows:
        photo_id = row.get("photo_id") if isinstance(row, dict) else None
        image = row.get("image") if isinstance(row, dict) else None
        if type(photo_id) is not int or not isinstance(image, str) or not image:
            raise PortfolioCoreRuntimeSourceError("iNaturalist manifest row is invalid")
        if photo_id in photo_ids:
            raise PortfolioCoreRuntimeSourceError(
                "iNaturalist manifest photo_id is duplicated"
            )
        photo_ids.add(photo_id)

    recipe_content = _read(
        auxiliary.recipe_evidence,
        "Core recipe evidence",
        max_bytes=_MAX_AUXILIARY_BYTES,
    )
    if sha256_bytes(recipe_content) != auxiliary.expected_recipe_evidence_sha256:
        raise PortfolioCoreRuntimeSourceError("recipe evidence SHA-256 mismatch")
    recipe_rows, _ignored = _parse_strict_jsonl(
        recipe_content,
        label="Core recipe evidence",
        allow_blank_lines=True,
    )
    _validate_recipe_evidence_rows(recipe_rows)

    for caption in auxiliary.fashioniq_captions:
        content = _read(
            caption.path,
            f"FashionIQ caption {caption.path.name}",
            max_bytes=_MAX_AUXILIARY_BYTES,
        )
        if sha256_bytes(content) != caption.expected_sha256:
            raise PortfolioCoreRuntimeSourceError(
                f"FashionIQ caption SHA-256 mismatch: {caption.path.name}"
            )
        try:
            parsed = parse_strict_json(
                content, label=f"FashionIQ caption {caption.path.name}"
            )
        except ArtifactFormatError as error:
            raise PortfolioCoreRuntimeSourceError(str(error)) from error
        if not isinstance(parsed, list):
            raise PortfolioCoreRuntimeSourceError(
                f"FashionIQ caption is not an array: {caption.path.name}"
            )


def _primary_sources(verified: VerifiedPortfolioCoreInputs) -> _PrimarySources:
    remote = verified.files.remote_files
    if remote is None:
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime sources require the remote-processing receipt binding"
        )
    selection_path = remote.selection_manifest.absolute()
    dataset_assets_path = remote.dataset_assets.absolute()
    runtime_catalog_dir = remote.output_catalog_dir.absolute()
    runtime_catalog_assets_path = runtime_catalog_dir / "assets.jsonl"
    selection_content = _read(
        selection_path,
        "Core v9 selection manifest",
        max_bytes=_MAX_SELECTION_BYTES,
    )
    dataset_assets_content = _read(
        dataset_assets_path,
        "Core v9 dataset assets",
        max_bytes=_MAX_DATASET_ASSETS_BYTES,
    )
    runtime_catalog_assets_content = _read(
        runtime_catalog_assets_path,
        "Core runtime catalog assets",
        max_bytes=_MAX_CATALOG_ASSETS_BYTES,
    )
    try:
        selection_value = parse_canonical_json(
            selection_content, label="Core v9 selection manifest"
        )
        parse_canonical_jsonl(dataset_assets_content, label="Core v9 dataset assets")
        parse_canonical_jsonl(
            runtime_catalog_assets_content,
            label="Core runtime catalog assets",
        )
    except ArtifactFormatError as error:
        raise PortfolioCoreRuntimeSourceError(str(error)) from error
    if not isinstance(selection_value, dict):
        raise PortfolioCoreRuntimeSourceError("Core selection is not one object")

    selection_sha256 = sha256_bytes(selection_content)
    dataset_assets_sha256 = sha256_bytes(dataset_assets_content)
    receipts = tuple(item.receipt for item in verified.remote_runtimes)
    receipt_file_hashes = {
        item.receipt_file_sha256 for item in verified.remote_runtimes
    }
    if len(receipt_file_hashes) != 1 or any(
        receipt.base_selection_manifest_sha256 != selection_sha256
        or receipt.base_dataset_assets_sha256 != dataset_assets_sha256
        or receipt.output_catalog_sha256 != verified.expected_output_catalog_sha256
        for receipt in receipts
    ):
        raise PortfolioCoreRuntimeSourceError(
            "Core primary runtime sources differ from the permission receipt"
        )
    return _PrimarySources(
        selection_path=selection_path,
        selection_content=selection_content,
        selection=selection_value,
        dataset_assets_path=dataset_assets_path,
        dataset_assets_content=dataset_assets_content,
        runtime_catalog_assets_path=runtime_catalog_assets_path,
        runtime_catalog_assets_content=runtime_catalog_assets_content,
        asset_root=verified.files.asset_root.absolute(),
        core_catalog_sha256=verified.expected_output_catalog_sha256,
        remote_receipt_file_sha256=next(iter(receipt_file_hashes)),
    )


def _assert_opaque_assistant_projection(
    verified: VerifiedPortfolioCoreInputs,
) -> None:
    private_strings = {query.asset_id for query in verified.queries} | {
        query.image_path for query in verified.queries
    }
    for projected in verified.assistant_queries:
        try:
            public = parse_canonical_json(
                projected.public_input_json.encode("utf-8"),
                label="Core Assistant public input",
            )
        except ArtifactFormatError as error:
            raise PortfolioCoreRuntimeSourceError(str(error)) from error
        if not isinstance(public, dict) or set(public) != {
            "asset_id",
            "text",
            "turns",
        }:
            raise PortfolioCoreRuntimeSourceError(
                "Core runtime-source resolution found a non-opaque Assistant DTO"
            )
        serialized = canonical_json_bytes(public).decode("utf-8")
        if any(private and private in serialized for private in private_strings):
            raise PortfolioCoreRuntimeSourceError(
                "Core Assistant DTO exposes a private asset identity"
            )
        for forbidden in (
            "canonical_capability",
            "canonical_intent",
            "acceptable_capabilities",
            "label_provenance",
            "template_family",
            "split",
            "source_record_id",
        ):
            if f'"{forbidden}"' in serialized:
                raise PortfolioCoreRuntimeSourceError(
                    f"Core Assistant DTO exposes private label field: {forbidden}"
                )


def _derive_sources(
    primary: _PrimarySources,
    auxiliary: PortfolioCoreRuntimeAuxiliaryFiles,
) -> _DerivedSources:
    selections = primary.selection.get("selections")
    if not isinstance(selections, list) or not selections:
        raise PortfolioCoreRuntimeSourceError("Core selection has no rows")
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for value in selections:
        if not isinstance(value, dict) or not isinstance(value.get("draft"), dict):
            raise PortfolioCoreRuntimeSourceError("Core selection row is invalid")
        draft = value["draft"]
        source = draft.get("source_dataset")
        if not isinstance(source, str):
            raise PortfolioCoreRuntimeSourceError("Core selection source is invalid")
        source_rows.setdefault(source, []).append(value)
    for required in (_RPC_SOURCE, _INATURALIST_SOURCE, _RECIPE_SOURCE):
        if not source_rows.get(required):
            raise PortfolioCoreRuntimeSourceError(
                f"Core selection lacks required runtime source: {required}"
            )

    inaturalist_bytes, inaturalist_record = _inaturalist_projection(
        source_rows[_INATURALIST_SOURCE],
        auxiliary.inaturalist_manifest,
        expected_sha256=auxiliary.expected_inaturalist_manifest_sha256,
    )
    recipe_bytes, recipe_record = _recipe_projection(
        source_rows[_RECIPE_SOURCE],
        auxiliary.recipe_evidence,
        expected_sha256=auxiliary.expected_recipe_evidence_sha256,
    )
    caption_records = []
    caption_payloads: list[tuple[str, bytes]] = []
    for caption in auxiliary.fashioniq_captions:
        content = _read(
            caption.path,
            f"FashionIQ caption {caption.path.name}",
            max_bytes=_MAX_AUXILIARY_BYTES,
        )
        if sha256_bytes(content) != caption.expected_sha256:
            raise PortfolioCoreRuntimeSourceError(
                f"FashionIQ caption SHA-256 mismatch: {caption.path.name}"
            )
        try:
            parsed = parse_strict_json(
                content, label=f"FashionIQ caption {caption.path.name}"
            )
        except ArtifactFormatError as error:
            raise PortfolioCoreRuntimeSourceError(str(error)) from error
        if not isinstance(parsed, list):
            raise PortfolioCoreRuntimeSourceError(
                f"FashionIQ caption is not an array: {caption.path.name}"
            )
        caption_records.append(
            {
                "bytes": len(content),
                "name": caption.path.name,
                "output_path": (f"{CORE_RUNTIME_CAPTION_DIR}/{caption.path.name}"),
                "rows": len(parsed),
                "sha256": caption.expected_sha256,
            }
        )
        caption_payloads.append((caption.path.name, content))
    coordination_graph_content: bytes | None = None
    coordination_graph_record: dict[str, Any] | None = None
    if auxiliary.style_coordination_graph is not None:
        coordination_graph_content, coordination_graph = _read_style_coordination_graph(
            auxiliary.style_coordination_graph
        )
        source_bindings = coordination_graph["source_bindings"]
        if source_bindings["selection_manifest_sha256"] != sha256_bytes(
            primary.selection_content
        ) or source_bindings["runtime_catalog_assets_sha256"] != sha256_bytes(
            primary.runtime_catalog_assets_content
        ):
            raise PortfolioCoreRuntimeSourceError(
                "style coordination graph source bindings differ from verified Core inputs"
            )
        coordination_graph_record = {
            "bytes": len(coordination_graph_content),
            "edge_count": len(coordination_graph["edges"]),
            "graph_sha256": coordination_graph["graph_sha256"],
            "output_path": CORE_RUNTIME_STYLE_COORDINATION_GRAPH,
            "policy_version": coordination_graph["policy_version"],
            "sha256": auxiliary.style_coordination_graph.expected_sha256,
        }
    # The RPC archive is much larger than every other auxiliary source.  Run
    # all cheap exact-digest/schema checks first so a bad small input cannot
    # waste the archive verification/read.
    rpc_bytes, rpc_record = _rpc_projection(
        source_rows[_RPC_SOURCE], auxiliary, asset_root=primary.asset_root
    )
    input_records: dict[str, Any] = {
        "rpc": rpc_record,
        "inaturalist": inaturalist_record,
        "recipe_evidence": recipe_record,
        "fashioniq_captions": caption_records,
    }
    if coordination_graph_record is not None:
        input_records["style_coordination_graph"] = coordination_graph_record
    return _DerivedSources(
        rpc_scenes=rpc_bytes,
        inaturalist=inaturalist_bytes,
        recipes=recipe_bytes,
        rpc_scene_count=rpc_record["selected_scene_count"],
        inaturalist_row_count=inaturalist_record["selected_row_count"],
        recipe_row_count=recipe_record["selected_row_count"],
        fashioniq_captions=tuple(caption_payloads),
        style_coordination_graph=coordination_graph_content,
        input_records=input_records,
    )


def _rpc_projection(
    selections: list[dict[str, Any]],
    auxiliary: PortfolioCoreRuntimeAuxiliaryFiles,
    *,
    asset_root: Path,
) -> tuple[bytes, dict[str, Any]]:
    manifest_content = _read(
        auxiliary.rpc_adapter_manifest,
        "Core RPC adapter manifest",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(manifest_content) != auxiliary.expected_rpc_adapter_manifest_sha256:
        raise PortfolioCoreRuntimeSourceError("RPC adapter manifest SHA-256 mismatch")
    try:
        manifest = parse_canonical_json(
            manifest_content, label="Core RPC adapter manifest"
        )
    except ArtifactFormatError as error:
        raise PortfolioCoreRuntimeSourceError(str(error)) from error
    if not isinstance(manifest, dict):
        raise PortfolioCoreRuntimeSourceError("RPC adapter manifest is not an object")
    source_revisions = {row["draft"].get("source_revision") for row in selections}
    archives = manifest.get("archives")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_id") != _RPC_SOURCE
        or len(source_revisions) != 1
        or manifest.get("source_revision") not in source_revisions
        or manifest.get("include_count") != len(selections)
        or not isinstance(archives, list)
        or len(archives) != 1
        or not isinstance(archives[0], dict)
    ):
        raise PortfolioCoreRuntimeSourceError(
            "RPC adapter manifest does not bind the selected Core RPC rows"
        )
    archive_record = archives[0]
    declared_archive_sha256 = _require_sha256(
        archive_record.get("sha256"), "RPC archive SHA-256"
    )
    declared_archive_bytes = archive_record.get("bytes")
    if type(declared_archive_bytes) is not int or declared_archive_bytes <= 0:
        raise PortfolioCoreRuntimeSourceError("RPC archive byte count is invalid")
    archive_snapshot = _regular_file_snapshot(auxiliary.rpc_archive, "Core RPC archive")
    archive_bytes = archive_snapshot[2]
    if archive_bytes != declared_archive_bytes:
        raise PortfolioCoreRuntimeSourceError(
            "RPC archive size differs from the selected-source adapter"
        )
    if auxiliary.verify_rpc_archive_full_sha256:
        try:
            observed_archive_sha256, observed_archive_bytes = stable_file_digest(
                auxiliary.rpc_archive, label="Core RPC archive"
            )
        except (OSError, SourceLockError, ValueError) as error:
            raise PortfolioCoreRuntimeSourceError(
                "unable to verify the full RPC archive"
            ) from error
        if (observed_archive_sha256, observed_archive_bytes) != (
            declared_archive_sha256,
            declared_archive_bytes,
        ):
            raise PortfolioCoreRuntimeSourceError(
                "RPC archive differs from the selected-source adapter"
            )

    try:
        if auxiliary.rpc_annotation_file is None:
            with zipfile.ZipFile(auxiliary.rpc_archive) as archive:
                members = rpc_source._validate_archive_members(archive)
                (
                    annotation_content,
                    annotation_paths,
                    _replica_roots,
                    _canonical_root,
                ) = rpc_source._load_annotation_replicas(archive, members)
            annotation_source_mode = "archive_member"
        else:
            annotation_content = _read(
                auxiliary.rpc_annotation_file,
                "externally hash-bound Core RPC validation annotations",
                max_bytes=_MAX_AUXILIARY_BYTES,
            )
            annotation_paths = ("instances_val2019.json",)
            annotation_source_mode = "externally_hash_bound_extracted_annotation"
        if (
            _regular_file_snapshot(auxiliary.rpc_archive, "Core RPC archive")
            != archive_snapshot
        ):
            raise PortfolioCoreRuntimeSourceError(
                "RPC archive changed while reading its annotation member"
            )
        annotation_sha256 = sha256_bytes(annotation_content)
        if annotation_sha256 != auxiliary.expected_rpc_annotation_sha256:
            raise PortfolioCoreRuntimeSourceError(
                "RPC annotation member SHA-256 mismatch"
            )
        annotation = parse_strict_json(
            annotation_content, label="Core RPC validation annotations"
        )
        _images, categories, instances_by_image = rpc_source._validate_annotations(
            annotation
        )
    except (
        ArtifactFormatError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        if isinstance(error, PortfolioCoreRuntimeSourceError):
            raise
        raise PortfolioCoreRuntimeSourceError(
            "RPC annotation member is invalid"
        ) from error

    scenes: list[dict[str, Any]] = []
    selected_image_bindings: list[dict[str, str]] = []
    seen_records: set[str] = set()
    for selection in selections:
        draft = selection["draft"]
        source_record_id = draft.get("source_record_id")
        if (
            not isinstance(source_record_id, str)
            or not source_record_id.startswith("val2019:")
            or source_record_id in seen_records
        ):
            raise PortfolioCoreRuntimeSourceError(
                "Core RPC selection has an invalid or duplicate record ID"
            )
        try:
            image_id = int(source_record_id.split(":", 1)[1])
        except ValueError as error:
            raise PortfolioCoreRuntimeSourceError(
                "Core RPC record ID has a non-integer image ID"
            ) from error
        raw_instances = instances_by_image.get(image_id)
        if raw_instances is None or len(raw_instances) < 2:
            raise PortfolioCoreRuntimeSourceError(
                f"Core RPC scene lacks two verified instances: {source_record_id}"
            )
        expected_image_sha256 = _require_sha256(
            selection.get("expected_sha256"), "Core RPC image SHA-256"
        )
        destination = selection.get("destination_path")
        if not isinstance(destination, str) or not destination:
            raise PortfolioCoreRuntimeSourceError(
                "Core RPC selection lacks a destination path"
            )
        image_path = _resolve_under(asset_root, destination, "Core RPC image")
        image_sha256 = _stable_sha(image_path, f"Core RPC image {source_record_id}")
        if image_sha256 != expected_image_sha256:
            raise PortfolioCoreRuntimeSourceError(
                f"Core RPC selected image SHA-256 mismatch: {source_record_id}"
            )
        selected_image_bindings.append(
            {
                "image_sha256": image_sha256,
                "source_record_id": source_record_id,
            }
        )
        instances = [
            {
                "annotation_id": item["annotation_id"],
                "bbox_area": item["bbox_xywh"][2] * item["bbox_xywh"][3],
                "bbox_xywh": list(item["bbox_xywh"]),
                "category_id": item["category_id"],
                "category_name": categories[item["category_id"]],
                "sku_product_id": f"rpc:{item['category_id']}",
            }
            for item in raw_instances
        ]
        scenes.append(
            {
                "image_sha256": expected_image_sha256,
                "instances": instances,
                "schema_version": 1,
                "source_record_id": source_record_id,
            }
        )
        seen_records.add(source_record_id)
    scene_bytes = canonical_jsonl_bytes(tuple(scenes))
    return scene_bytes, {
        "adapter_manifest_bytes": len(manifest_content),
        "adapter_manifest_sha256": auxiliary.expected_rpc_adapter_manifest_sha256,
        "annotation_document_sha256": annotation_sha256,
        "annotation_member_paths": list(annotation_paths),
        "annotation_source_mode": annotation_source_mode,
        "archive_bytes": archive_bytes,
        "archive_full_sha256_reverified": (auxiliary.verify_rpc_archive_full_sha256),
        "archive_sha256_declared_by_adapter_manifest": declared_archive_sha256,
        "selected_image_binding_sha256": sha256_bytes(
            canonical_json_bytes(selected_image_bindings)
        ),
        "selected_image_bytes_verified": True,
        "verification_boundary": (
            "externally_hashed_adapter_manifest+archive_size_and_declared_identity+"
            "exact_annotation_member_sha256+verified_selected_image_bytes"
        ),
        "selected_scene_count": len(scenes),
    }


def _inaturalist_projection(
    selections: list[dict[str, Any]],
    manifest_path: Path,
    *,
    expected_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    content = _read(
        manifest_path,
        "Core iNaturalist manifest",
        max_bytes=_MAX_AUXILIARY_BYTES,
    )
    source_revisions = {row["draft"].get("source_revision") for row in selections}
    if len(source_revisions) != 1:
        raise PortfolioCoreRuntimeSourceError(
            "Core iNaturalist selection has multiple source revisions"
        )
    revision = next(iter(source_revisions))
    if not isinstance(revision, str) or not revision.startswith("manifest-"):
        raise PortfolioCoreRuntimeSourceError(
            "Core iNaturalist revision does not bind a manifest digest"
        )
    manifest_sha256 = revision.removeprefix("manifest-")
    if manifest_sha256 != expected_sha256 or sha256_bytes(content) != expected_sha256:
        raise PortfolioCoreRuntimeSourceError(
            "iNaturalist manifest differs from the Core selection revision"
        )
    rows, _ignored_blank_lines = _parse_strict_jsonl(
        content, label="Core iNaturalist manifest"
    )
    by_photo_id: dict[int, dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, dict) or type(value.get("photo_id")) is not int:
            raise PortfolioCoreRuntimeSourceError(
                "iNaturalist manifest row lacks an integer photo_id"
            )
        photo_id = value["photo_id"]
        if photo_id in by_photo_id:
            raise PortfolioCoreRuntimeSourceError(
                "iNaturalist manifest photo_id is duplicated"
            )
        by_photo_id[photo_id] = value
    projected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for selection in selections:
        draft = selection["draft"]
        record = draft.get("source_record_id")
        destination = selection.get("destination_path")
        if (
            not isinstance(record, str)
            or not record.startswith("photo:")
            or not isinstance(destination, str)
            or not destination
        ):
            raise PortfolioCoreRuntimeSourceError(
                "Core iNaturalist selection row is invalid"
            )
        try:
            photo_id = int(record.split(":", 1)[1])
        except ValueError as error:
            raise PortfolioCoreRuntimeSourceError(
                "Core iNaturalist record has a non-integer photo ID"
            ) from error
        source = by_photo_id.get(photo_id)
        if source is None or photo_id in seen:
            raise PortfolioCoreRuntimeSourceError(
                "Core iNaturalist selection is absent or duplicated in its manifest"
            )
        original_image = source.get("image")
        if not isinstance(original_image, str) or not original_image:
            raise PortfolioCoreRuntimeSourceError(
                "Core iNaturalist source row lacks its original image"
            )
        projected.append(
            {
                **source,
                "image": Path(destination).name,
                "source_image": original_image,
            }
        )
        seen.add(photo_id)
    projected_bytes = canonical_jsonl_bytes(tuple(projected))
    return projected_bytes, {
        "manifest_bytes": len(content),
        "manifest_sha256": manifest_sha256,
        "source_row_count": len(rows),
        "selected_row_count": len(projected),
    }


def _recipe_projection(
    selections: list[dict[str, Any]],
    evidence_path: Path,
    *,
    expected_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    content = _read(
        evidence_path,
        "Core recipe evidence",
        max_bytes=_MAX_AUXILIARY_BYTES,
    )
    if sha256_bytes(content) != expected_sha256:
        raise PortfolioCoreRuntimeSourceError("recipe evidence SHA-256 mismatch")
    rows, ignored_blank_line_count = _parse_strict_jsonl(
        content,
        label="Core recipe evidence",
        allow_blank_lines=True,
    )
    normalized = _validate_recipe_evidence_rows(rows)
    selected_categories: set[str] = set()
    source_revisions = {row["draft"].get("source_revision") for row in selections}
    if len(source_revisions) != 1:
        raise PortfolioCoreRuntimeSourceError(
            "Core recipe selections have multiple image-source revisions"
        )
    for selection in selections:
        record = selection["draft"].get("source_record_id")
        marker = "/images/"
        if not isinstance(record, str) or marker not in record:
            raise PortfolioCoreRuntimeSourceError(
                "Core recipe selection record is invalid"
            )
        selected_categories.add(record.split(marker, 1)[1].split("/", 1)[0])
    selected = tuple(
        sorted(
            (row for row in normalized if row["category"] in selected_categories),
            key=lambda row: (str(row["category"]), str(row["source_record_id"])),
        )
    )
    covered = {str(row["category"]) for row in selected}
    missing = sorted(selected_categories - covered)
    if missing:
        raise PortfolioCoreRuntimeSourceError(
            "recipe evidence does not cover Core image categories: "
            + ", ".join(missing)
        )
    projected_bytes = canonical_jsonl_bytes(selected)
    return projected_bytes, {
        "evidence_bytes": len(content),
        "evidence_sha256": expected_sha256,
        "ignored_blank_line_count": ignored_blank_line_count,
        "image_source_revision": next(iter(source_revisions)),
        "selected_categories": sorted(selected_categories),
        "selected_row_count": len(selected),
        "source_row_count": len(rows),
    }


def _validate_recipe_evidence_rows(
    rows: tuple[object, ...],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    source_record_ids: set[str] = set()
    for value in rows:
        if not isinstance(value, dict):
            raise PortfolioCoreRuntimeSourceError(
                "recipe evidence row is not an object"
            )
        required = (
            "category",
            "license_id",
            "name",
            "recipeIngredient",
            "recipeInstructions",
            "source_archive_sha256",
            "source_record_id",
            "source_uri",
        )
        if any(key not in value for key in required):
            raise PortfolioCoreRuntimeSourceError("recipe evidence row is incomplete")
        source_record_id = value.get("source_record_id")
        if not isinstance(value["category"], str):
            raise PortfolioCoreRuntimeSourceError("recipe category is invalid")
        if not isinstance(source_record_id, str) or not source_record_id:
            raise PortfolioCoreRuntimeSourceError(
                "recipe evidence source_record_id is invalid"
            )
        if source_record_id in source_record_ids:
            raise PortfolioCoreRuntimeSourceError(
                "recipe evidence source_record_id is duplicated"
            )
        source_record_ids.add(source_record_id)
        normalized.append(value)
    return normalized


def _receipt(
    *,
    verified: VerifiedPortfolioCoreInputs,
    primary: _PrimarySources,
    auxiliary: PortfolioCoreRuntimeAuxiliaryFiles,
    derived: _DerivedSources,
    payloads: dict[str, bytes],
) -> dict[str, Any]:
    source_hashes = (
        sha256_bytes(primary.selection_content),
        sha256_bytes(primary.dataset_assets_content),
        sha256_bytes(primary.runtime_catalog_assets_content),
        sha256_bytes(payloads[CORE_RUNTIME_RPC_SCENES]),
        sha256_bytes(payloads[CORE_RUNTIME_INATURALIST]),
        sha256_bytes(payloads[CORE_RUNTIME_RECIPES]),
        *(item.expected_sha256 for item in auxiliary.fashioniq_captions),
        *(
            ()
            if auxiliary.style_coordination_graph is None
            else (auxiliary.style_coordination_graph.expected_sha256,)
        ),
    )
    runtime_data_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                "source_sha256s": list(source_hashes),
            }
        )
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "portfolio-core-runtime-sources-receipt",
        "policy_version": CORE_RUNTIME_SOURCE_POLICY,
        "scope": "core-v9-local-tool-runtime",
        "formal_eligible": False,
        "provider_call_count": 0,
        "verification_boundary": {
            "claim": "exact inputs consumed by the local Core tool runtime",
            "primary_sources": ("verified Core inputs plus remote-processing receipt"),
            "rpc": (
                "externally hashed adapter manifest, archive size and declared "
                "identity, exact annotation-member SHA-256, and each selected "
                "image expected SHA-256"
            ),
            "rpc_full_archive_sha256_reverified": (
                auxiliary.verify_rpc_archive_full_sha256
            ),
            "downstream_reload_reads_raw_rpc_archive": False,
            "provider_effect_claimed": False,
        },
        "assistant_projection": {
            "field_allowlist": ["asset_id", "text", "turns"],
            "private_labels_exposed": False,
            "query_count_verified": len(verified.queries),
        },
        "core_bindings": {
            "plan_sha256": verified.expected_plan_sha256,
            "query_artifact_sha256": verified.expected_query_artifact_sha256,
            "catalog_sha256": primary.core_catalog_sha256,
            "remote_receipt_file_sha256": primary.remote_receipt_file_sha256,
            "selection_manifest_sha256": sha256_bytes(primary.selection_content),
            "dataset_assets_sha256": sha256_bytes(primary.dataset_assets_content),
            "runtime_catalog_assets_file_sha256": sha256_bytes(
                primary.runtime_catalog_assets_content
            ),
        },
        "auxiliary_inputs": derived.input_records,
        "outputs": {
            name: _output_descriptor(name, content, derived=derived)
            for name, content in sorted(payloads.items())
        },
        "runtime_source_sha256s": list(source_hashes),
        "runtime_data_sha256": runtime_data_sha256,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def _output_descriptor(
    name: str,
    content: bytes,
    *,
    derived: _DerivedSources | None = None,
) -> dict[str, Any]:
    try:
        if name == CORE_RUNTIME_STYLE_COORDINATION_GRAPH:
            value = parse_canonical_json(content, label=name)
            if not isinstance(value, dict):
                raise PortfolioCoreRuntimeSourceError(
                    "Core style coordination graph output is not an object"
                )
            _validate_style_coordination_graph(value)
            rows = len(value["edges"])
            if derived is not None:
                record = derived.input_records.get("style_coordination_graph")
                if (
                    not isinstance(record, dict)
                    or record.get("output_path") != name
                    or record.get("edge_count") != rows
                ):
                    raise PortfolioCoreRuntimeSourceError(
                        "Core style coordination graph output binding differs"
                    )
        elif name.startswith(f"{CORE_RUNTIME_CAPTION_DIR}/"):
            value = parse_strict_json(content, label=name)
            if not isinstance(value, list):
                raise PortfolioCoreRuntimeSourceError(
                    f"Core runtime caption output is not an array: {name}"
                )
            rows = len(value)
            if derived is not None:
                matches = [
                    item
                    for item in derived.input_records["fashioniq_captions"]
                    if item["output_path"] == name
                ]
                if len(matches) != 1 or matches[0]["rows"] != rows:
                    raise PortfolioCoreRuntimeSourceError(
                        f"Core runtime caption row binding differs: {name}"
                    )
        else:
            rows = len(parse_canonical_jsonl(content, label=name))
    except ArtifactFormatError as error:
        raise PortfolioCoreRuntimeSourceError(str(error)) from error
    return {
        "bytes": len(content),
        "rows": rows,
        "sha256": sha256_bytes(content),
    }


def _load_materialized(
    verified: VerifiedPortfolioCoreInputs,
    auxiliary: PortfolioCoreRuntimeAuxiliaryFiles | None,
    output_dir: Path,
    *,
    expected_receipt_file_sha256: str,
    preverified_derived: _DerivedSources | None = None,
    _allow_legacy_tool_runtime_policy: bool = False,
) -> VerifiedPortfolioCoreRuntimeSources:
    if type(_allow_legacy_tool_runtime_policy) is not bool:
        raise TypeError("legacy tool-runtime policy flag must be boolean")
    primary = _primary_sources(verified)
    receipt_path = output_dir / CORE_RUNTIME_SOURCE_RECEIPT
    receipt_content = _read(
        receipt_path,
        "Core runtime-source receipt",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(receipt_content) != expected_receipt_file_sha256:
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source receipt external digest mismatch"
        )
    try:
        receipt = parse_canonical_json(
            receipt_content, label="Core runtime-source receipt"
        )
    except ArtifactFormatError as error:
        raise PortfolioCoreRuntimeSourceError(str(error)) from error
    if not isinstance(receipt, dict):
        raise PortfolioCoreRuntimeSourceError("Core runtime-source receipt is invalid")
    declared_self_hash = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if (
        receipt.get("kind") != "portfolio-core-runtime-sources-receipt"
        or receipt.get("policy_version") != CORE_RUNTIME_SOURCE_POLICY
        or receipt.get("provider_call_count") != 0
        or receipt.get("formal_eligible") is not False
        or declared_self_hash != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source receipt boundary or self hash is invalid"
        )
    expected_core = {
        "plan_sha256": verified.expected_plan_sha256,
        "query_artifact_sha256": verified.expected_query_artifact_sha256,
        "catalog_sha256": primary.core_catalog_sha256,
        "remote_receipt_file_sha256": primary.remote_receipt_file_sha256,
        "selection_manifest_sha256": sha256_bytes(primary.selection_content),
        "dataset_assets_sha256": sha256_bytes(primary.dataset_assets_content),
        "runtime_catalog_assets_file_sha256": sha256_bytes(
            primary.runtime_catalog_assets_content
        ),
    }
    if receipt.get("core_bindings") != expected_core:
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source receipt differs from verified Core inputs"
        )

    auxiliary_inputs = receipt.get("auxiliary_inputs")
    if not isinstance(auxiliary_inputs, dict):
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source receipt lacks auxiliary input bindings"
        )
    caption_records = auxiliary_inputs.get("fashioniq_captions")
    if not isinstance(caption_records, list):
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source caption bindings are invalid"
        )
    caption_output_names: list[str] = []
    for item in caption_records:
        if not isinstance(item, dict):
            raise PortfolioCoreRuntimeSourceError(
                "Core runtime-source caption binding is not an object"
            )
        name = item.get("name")
        output_name = item.get("output_path")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or output_name != f"{CORE_RUNTIME_CAPTION_DIR}/{name}"
        ):
            raise PortfolioCoreRuntimeSourceError(
                "Core runtime-source caption output path is unsafe"
            )
        caption_output_names.append(output_name)
    if len(set(caption_output_names)) != len(caption_output_names):
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source caption outputs are duplicated"
        )
    coordination_graph_record: dict[str, Any] | None = None
    if "style_coordination_graph" in auxiliary_inputs:
        candidate_record = auxiliary_inputs["style_coordination_graph"]
        if (
            not isinstance(candidate_record, dict)
            or set(candidate_record)
            != {
                "bytes",
                "edge_count",
                "graph_sha256",
                "output_path",
                "policy_version",
                "sha256",
            }
            or candidate_record.get("output_path")
            != CORE_RUNTIME_STYLE_COORDINATION_GRAPH
            or candidate_record.get("policy_version")
            != _STYLE_COORDINATION_GRAPH_POLICY
            or type(candidate_record.get("bytes")) is not int
            or candidate_record["bytes"] <= 0
            or type(candidate_record.get("edge_count")) is not int
            or candidate_record["edge_count"] <= 0
        ):
            raise PortfolioCoreRuntimeSourceError(
                "Core runtime-source style coordination graph binding is invalid"
            )
        _require_sha256(
            candidate_record.get("sha256"),
            "Core runtime-source style coordination graph SHA-256",
        )
        _require_sha256(
            candidate_record.get("graph_sha256"),
            "Core runtime-source style coordination graph self SHA-256",
        )
        coordination_graph_record = candidate_record
    expected_output_names = {
        CORE_RUNTIME_RPC_SCENES,
        CORE_RUNTIME_INATURALIST,
        CORE_RUNTIME_RECIPES,
        *caption_output_names,
    }
    if coordination_graph_record is not None:
        expected_output_names.add(CORE_RUNTIME_STYLE_COORDINATION_GRAPH)
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != expected_output_names:
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source receipt output set is invalid"
        )
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source output must be a real directory"
        )
    actual_file_names = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if actual_file_names != {CORE_RUNTIME_SOURCE_RECEIPT, *expected_output_names}:
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime-source output file set differs from receipt"
        )
    output_paths = {
        name: output_dir.joinpath(*name.split("/"))
        for name in sorted(expected_output_names)
    }
    for name, path in output_paths.items():
        content = _read(
            path, f"Core runtime output {name}", max_bytes=_MAX_AUXILIARY_BYTES
        )
        descriptor = outputs[name]
        observed_descriptor = _output_descriptor(name, content)
        if not isinstance(descriptor, dict) or descriptor != observed_descriptor:
            raise PortfolioCoreRuntimeSourceError(
                f"Core runtime output differs from receipt: {name}"
            )
        if name == CORE_RUNTIME_STYLE_COORDINATION_GRAPH:
            try:
                graph = parse_canonical_json(content, label=name)
            except (
                ArtifactFormatError
            ) as error:  # pragma: no cover - descriptor parsed it
                raise PortfolioCoreRuntimeSourceError(str(error)) from error
            if not isinstance(graph, dict):  # pragma: no cover - descriptor checked it
                raise PortfolioCoreRuntimeSourceError(
                    "Core style coordination graph output is not an object"
                )
            expected_graph_record = {
                "bytes": len(content),
                "edge_count": len(graph["edges"]),
                "graph_sha256": graph["graph_sha256"],
                "output_path": CORE_RUNTIME_STYLE_COORDINATION_GRAPH,
                "policy_version": graph["policy_version"],
                "sha256": sha256_bytes(content),
            }
            if coordination_graph_record != expected_graph_record:
                raise PortfolioCoreRuntimeSourceError(
                    "Core runtime-source style coordination graph binding differs"
                )

    # Re-derive from current auxiliary inputs.  This makes an old receipt fail
    # closed if an annotation/evidence file has drifted even when output bytes
    # were left untouched.
    derived = preverified_derived
    if derived is None and auxiliary is not None:
        derived = _derive_sources(primary, auxiliary)
    if derived is not None:
        if receipt.get("auxiliary_inputs") != derived.input_records:
            raise PortfolioCoreRuntimeSourceError(
                "Core runtime-source auxiliary inputs differ from receipt"
            )
        expected_derived = {
            CORE_RUNTIME_RPC_SCENES: derived.rpc_scenes,
            CORE_RUNTIME_INATURALIST: derived.inaturalist,
            CORE_RUNTIME_RECIPES: derived.recipes,
            **{
                f"{CORE_RUNTIME_CAPTION_DIR}/{name}": content
                for name, content in derived.fashioniq_captions
            },
        }
        if derived.style_coordination_graph is not None:
            expected_derived[CORE_RUNTIME_STYLE_COORDINATION_GRAPH] = (
                derived.style_coordination_graph
            )
        for name, expected in expected_derived.items():
            if (
                _read(
                    output_paths[name],
                    f"Core runtime output {name}",
                    max_bytes=_MAX_AUXILIARY_BYTES,
                )
                != expected
            ):
                raise PortfolioCoreRuntimeSourceError(
                    f"Core runtime output cannot be reproduced: {name}"
                )

    captions = tuple(output_paths[name] for name in caption_output_names)
    coordination_graph_path = (
        None
        if coordination_graph_record is None
        else output_paths[CORE_RUNTIME_STYLE_COORDINATION_GRAPH]
    )
    sources = PortfolioRuntimeSources(
        selection_manifest=primary.selection_path,
        dataset_assets=primary.dataset_assets_path,
        runtime_catalog_assets=primary.runtime_catalog_assets_path,
        rpc_scenes=output_paths[CORE_RUNTIME_RPC_SCENES],
        inaturalist_manifest=output_paths[CORE_RUNTIME_INATURALIST],
        recipe_evidence=output_paths[CORE_RUNTIME_RECIPES],
        asset_root=primary.asset_root,
        fashioniq_captions=captions,
        style_coordination_graph=coordination_graph_path,
    )
    source_hashes = tuple(_stable_sha(path, path.name) for path in sources.files())
    if receipt.get("runtime_source_sha256s") != list(source_hashes):
        raise PortfolioCoreRuntimeSourceError(
            "Core runtime source order or digest differs from receipt"
        )
    runtime_data_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                "source_sha256s": list(source_hashes),
            }
        )
    )
    if receipt.get("runtime_data_sha256") != runtime_data_sha256:
        legacy_runtime_data_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "policy_version": "portfolio-public-data-tools-v2",
                    "source_sha256s": list(source_hashes),
                }
            )
        )
        legacy_shape = (
            coordination_graph_record is None and "contract_refresh" not in receipt
        )
        if not (
            _allow_legacy_tool_runtime_policy
            and legacy_shape
            and receipt.get("runtime_data_sha256") == legacy_runtime_data_sha256
        ):
            raise PortfolioCoreRuntimeSourceError(
                "Core tool runtime-data binding differs from receipt"
            )
    return VerifiedPortfolioCoreRuntimeSources(
        sources=sources,
        output_dir=output_dir,
        receipt_path=receipt_path,
        receipt_file_sha256=expected_receipt_file_sha256,
        source_file_sha256s=source_hashes,
        rpc_scene_count=int(outputs[CORE_RUNTIME_RPC_SCENES]["rows"]),
        inaturalist_row_count=int(outputs[CORE_RUNTIME_INATURALIST]["rows"]),
        recipe_row_count=int(outputs[CORE_RUNTIME_RECIPES]["rows"]),
        provider_call_count=0,
        formal_eligible=False,
        _marker=_VERIFIED_CORE_RUNTIME_SOURCES,
    )


def _read(path: Path, label: str, *, max_bytes: int) -> bytes:
    try:
        return read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    except (ArtifactFormatError, OSError, ValueError) as error:
        raise PortfolioCoreRuntimeSourceError(f"unable to read {label}") from error


def _parse_strict_jsonl(
    content: bytes,
    *,
    label: str,
    allow_blank_lines: bool = False,
) -> tuple[tuple[object, ...], int]:
    if not content or not content.endswith(b"\n"):
        raise PortfolioCoreRuntimeSourceError(
            f"{label} must be non-empty JSONL ending with a newline"
        )
    rows: list[object] = []
    ignored_blank_line_count = 0
    for line_number, line in enumerate(content.splitlines(keepends=True), 1):
        if not line.strip():
            if allow_blank_lines:
                ignored_blank_line_count += 1
                continue
            raise PortfolioCoreRuntimeSourceError(
                f"{label} contains blank line {line_number}"
            )
        try:
            rows.append(
                parse_strict_json(
                    line,
                    label=f"{label} line {line_number}",
                )
            )
        except ArtifactFormatError as error:
            raise PortfolioCoreRuntimeSourceError(str(error)) from error
    if not rows:
        raise PortfolioCoreRuntimeSourceError(f"{label} has no JSON records")
    return tuple(rows), ignored_blank_line_count


def _regular_file_snapshot(path: Path, label: str) -> tuple[int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PortfolioCoreRuntimeSourceError(f"unable to inspect {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PortfolioCoreRuntimeSourceError(
            f"{label} must be a regular non-symlink file"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _resolve_under(root: Path, relative: str, label: str) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        path = (resolved_root / relative).resolve(strict=True)
        path.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise PortfolioCoreRuntimeSourceError(
            f"{label} resolves outside the verified Core asset root or is missing"
        ) from error
    _regular_file_snapshot(path, label)
    return path


def _stable_sha(path: Path, label: str) -> str:
    try:
        digest, _size = stable_file_digest(path, label=label)
    except (OSError, SourceLockError, ValueError) as error:
        raise PortfolioCoreRuntimeSourceError(f"unable to hash {label}") from error
    return digest


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PortfolioCoreRuntimeSourceError(f"{label} must be lowercase SHA-256")
    return value


__all__ = [
    "BoundPortfolioRuntimeFile",
    "CORE_RUNTIME_INATURALIST",
    "CORE_RUNTIME_RECIPES",
    "CORE_RUNTIME_RPC_SCENES",
    "CORE_RUNTIME_STYLE_COORDINATION_GRAPH",
    "CORE_RUNTIME_SOURCE_POLICY",
    "CORE_RUNTIME_SOURCE_RECEIPT",
    "PortfolioCoreRuntimeAuxiliaryFiles",
    "PortfolioCoreRuntimeSourceError",
    "VerifiedPortfolioCoreRuntimeSources",
    "load_verified_portfolio_core_runtime_sources",
    "materialize_verified_portfolio_core_runtime_sources",
    "require_verified_portfolio_core_runtime_sources",
]
