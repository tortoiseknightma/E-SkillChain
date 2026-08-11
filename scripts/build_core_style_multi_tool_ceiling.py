"""Build a deterministic local Gate 0 ceiling for Core Style and Multi tools.

This receipt is deliberately an annotation-oracle *tool ceiling*.  It verifies
that the available local evidence can satisfy the output contracts; it does
not claim detector, retrieval, assistant, or end-to-end model effectiveness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile

from skillchain.synthesis.store import atomic_create_file
from skillchain.evaluation.portfolio_core_runtime_sources import (
    PortfolioCoreRuntimeSourceError,
    _validate_style_coordination_graph,
)
from skillchain.tools.portfolio_runtime import (
    PORTFOLIO_TOOL_RUNTIME_POLICY,
    _is_style_coordination_query,
    _negated_style_coordination_categories,
    _requested_style_coordination_categories,
    _style_coordination_allowed_categories,
    _style_coordination_candidate_matches,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


SCHEMA_VERSION = 2
POLICY_VERSION = "core-style-multi-tool-ceiling-v2"
EVIDENCE_KIND = "annotation_oracle_tool_ceiling"
CORE_QUERY_COUNT = 1_500
STYLE_ELIGIBLE_SAMPLE_SIZE = 20
MULTI_SAMPLE_SIZE = 40
RPC_ANNOTATION_MEMBER = "instances_val2019.json"
MAX_SMALL_INPUT_BYTES = 128 * 1024 * 1024
MAX_COORDINATION_GRAPH_BYTES = 32 * 1024 * 1024
STYLE_CEILING_SPLITS = frozenset({"dev_mini", "opt_pool"})
STYLE_COORDINATION_GRAPH_KIND = "portfolio-style-coordination-graph"
STYLE_COORDINATION_GRAPH_POLICY = "portfolio-style-coordination-graph-v1"
STYLE_COORDINATION_ANNOTATION_POLICY = (
    "portfolio-style-coordination-candidate-review-v1"
)
STYLE_COORDINATION_RELATION = "portfolio_curated_coordination_rule"
STYLE_COORDINATION_CATEGORIES = frozenset({"footwear", "bag", "jewelry"})
STYLE_GRAPH_FORBIDDEN_FIELDS = frozenset(
    {
        "query_id",
        "query_text",
        "split",
        "user_text",
        "expected_answer",
        "answer",
        "final_score",
        "score",
        "config",
    }
)

FASHIONIQ_CATEGORIES = ("dress", "shirt", "toptee")
FASHIONIQ_SPLITS = ("test", "train", "val")
FASHIONIQ_CAPTION_FILES = tuple(
    f"cap.{category}.{split_name}.json"
    for category in FASHIONIQ_CATEGORIES
    for split_name in FASHIONIQ_SPLITS
)

# Thresholds are frozen in code before the evidence is generated.  Cross
# requests without an exact edge are explicitly excluded from eligible rates.
THRESHOLDS: dict[str, float] = {
    "style_same_category_nonempty_rate": 0.95,
    "style_same_category_category_consistency_rate": 1.0,
    "style_same_category_facet_complete_rate": 1.0,
    "style_same_category_provenance_complete_rate": 1.0,
    "style_same_category_non_speculative_candidate_rate": 1.0,
    "style_cross_category_mapped_nonempty_rate": 1.0,
    "style_cross_category_mapped_category_consistency_rate": 1.0,
    "style_cross_category_mapped_requested_family_complete_rate": 1.0,
    "style_cross_category_mapped_facet_complete_rate": 1.0,
    "style_cross_category_mapped_provenance_complete_rate": 1.0,
    "style_cross_category_mapped_non_speculative_candidate_rate": 1.0,
    "style_cross_category_uncovered_zero_hit_rate": 1.0,
    "style_cross_category_uncovered_not_counted_rate": 1.0,
    "style_cross_category_semantic_evidence_rate": 1.0,
    "multi_nonempty_scene_rate": 1.0,
    "multi_object_resolution_status_typed_rate": 1.0,
    "multi_matched_object_type_complete_rate": 1.0,
    "multi_candidate_dedup_rate": 1.0,
    "multi_item_refs_alignment_rate": 1.0,
    "multi_quantity_alignment_rate": 1.0,
    "multi_card_alignment_rate": 1.0,
    "multi_non_speculative_candidate_rate": 1.0,
}


class GateEvidenceError(ValueError):
    """Raised when a local input cannot support a trustworthy ceiling receipt."""


def _read_small(path: Path, *, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise GateEvidenceError(f"unable to inspect {label}: {path}") from exc
    if not path.is_file() or size > MAX_SMALL_INPUT_BYTES:
        raise GateEvidenceError(f"{label} must be a regular file <= 128 MiB: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise GateEvidenceError(f"unable to read {label}: {path}") from exc
    if len(content) != size:
        raise GateEvidenceError(f"{label} changed while being read: {path}")
    return content


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateEvidenceError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GateEvidenceError(f"{label} must contain a JSON object")
    return value


def _json_array(content: bytes, *, label: str) -> list[Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateEvidenceError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, list):
        raise GateEvidenceError(f"{label} must contain a JSON array")
    return value


def _load_queries(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    content = _read_small(path, label="Core queries")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GateEvidenceError(
                f"Core queries line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise GateEvidenceError(f"Core queries line {line_number} is not an object")
        rows.append(row)
    if len(rows) != CORE_QUERY_COUNT:
        raise GateEvidenceError(
            f"expected {CORE_QUERY_COUNT} Core queries, observed {len(rows)}"
        )
    query_ids = [row.get("query_id") for row in rows]
    if any(not isinstance(item, str) or not item for item in query_ids):
        raise GateEvidenceError("every Core query must have a non-empty query_id")
    if len(set(query_ids)) != len(query_ids):
        raise GateEvidenceError("Core query_id values must be unique")
    return content, rows


def _load_selection(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    content = _read_small(path, label="Core selection manifest")
    value = _json_object(content, label="Core selection manifest")
    rows = value.get("selections")
    if not isinstance(rows, list) or not rows:
        raise GateEvidenceError("Core selection manifest has no selections")
    if any(not isinstance(row, dict) for row in rows):
        raise GateEvidenceError("Core selections must be JSON objects")
    return content, rows


def _selection_by_destination(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        destination = row.get("destination_path")
        if not isinstance(destination, str) or not destination:
            raise GateEvidenceError("selection destination_path must be non-empty")
        if destination in result:
            raise GateEvidenceError(f"duplicate selection destination: {destination}")
        result[destination] = row
    return result


def _bind_query_selection(
    query: dict[str, Any],
    by_destination: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    image_path = query.get("image_path")
    if not isinstance(image_path, str) or image_path not in by_destination:
        raise GateEvidenceError(
            f"query {query.get('query_id')} has no selected image binding"
        )
    return by_destination[image_path]


def _fashioniq_anchor(
    selection: dict[str, Any],
) -> tuple[str, str] | None:
    draft = selection.get("draft")
    if not isinstance(draft, dict) or draft.get("source_dataset") != "fashioniq":
        return None
    source_record_id = draft.get("source_record_id")
    if not isinstance(source_record_id, str) or ":" not in source_record_id:
        raise GateEvidenceError("FashionIQ selection has an invalid source_record_id")
    category, anchor_id = source_record_id.split(":", 1)
    if category not in FASHIONIQ_CATEGORIES or not anchor_id:
        raise GateEvidenceError("FashionIQ selection has an invalid category or anchor")
    return category, anchor_id


def _reject_graph_forbidden_fields(value: object, *, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in STYLE_GRAPH_FORBIDDEN_FIELDS:
                raise GateEvidenceError(
                    f"style coordination graph contains forbidden field {key!r} "
                    f"at {location}"
                )
            _reject_graph_forbidden_fields(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_graph_forbidden_fields(child, location=f"{location}[{index}]")


def _exact_graph_keys(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise GateEvidenceError(f"style coordination graph {label} fields differ")
    return value


def _load_style_coordination_graph(
    path: Path,
    *,
    selection_content: bytes,
    by_destination: dict[str, dict[str, Any]],
) -> tuple[bytes, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load a self-hashed, query-blind graph and bind every anchor to selection."""

    content = _read_small(path, label="Style coordination graph")
    if len(content) > MAX_COORDINATION_GRAPH_BYTES:
        raise GateEvidenceError("Style coordination graph exceeds 32 MiB")
    graph = _json_object(content, label="Style coordination graph")
    if canonical_json_bytes(graph) != content:
        raise GateEvidenceError("Style coordination graph must be canonical JSON")
    try:
        _validate_style_coordination_graph(graph)
    except PortfolioCoreRuntimeSourceError as error:
        raise GateEvidenceError(
            f"Style coordination graph violates the Core runtime contract: {error}"
        ) from error

    # The shared validator establishes the portable graph contract.  The
    # checks below additionally bind every anchor to this exact Core selection
    # before constructing the query-time index.
    _reject_graph_forbidden_fields(graph, location="graph")
    graph = _exact_graph_keys(
        graph,
        frozenset(
            {
                "kind",
                "schema_version",
                "policy_version",
                "source_bindings",
                "edges",
                "graph_sha256",
            }
        ),
        label="root",
    )
    if (
        graph["kind"] != STYLE_COORDINATION_GRAPH_KIND
        or graph["schema_version"] != 1
        or graph["policy_version"] != STYLE_COORDINATION_GRAPH_POLICY
    ):
        raise GateEvidenceError(
            "Style coordination graph kind, schema, or policy differs"
        )
    observed_graph_sha = graph["graph_sha256"]
    unsigned = {key: value for key, value in graph.items() if key != "graph_sha256"}
    if (
        not isinstance(observed_graph_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", observed_graph_sha)
        or observed_graph_sha != sha256_bytes(canonical_json_bytes(unsigned))
    ):
        raise GateEvidenceError("Style coordination graph self hash differs")

    bindings = _exact_graph_keys(
        graph["source_bindings"],
        frozenset(
            {
                "selection_manifest_sha256",
                "runtime_catalog_assets_sha256",
                "abo_listings_archive_sha256",
                "candidate_seed_sha256",
            }
        ),
        label="source bindings",
    )
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in bindings.values()
    ):
        raise GateEvidenceError("Style coordination graph source binding is invalid")
    if bindings["selection_manifest_sha256"] != sha256_bytes(selection_content):
        raise GateEvidenceError(
            "Style coordination graph selection binding differs from Core selection"
        )

    raw_edges = graph["edges"]
    if not isinstance(raw_edges, list) or not raw_edges:
        raise GateEvidenceError("Style coordination graph edges are empty or invalid")
    edge_keys = frozenset(
        {
            "edge_id",
            "anchor",
            "candidate",
            "relation_kind",
            "confidence",
            "facets",
            "annotation_policy_version",
        }
    )
    anchor_keys = frozenset(
        {
            "asset_id",
            "product_id",
            "source_dataset",
            "source_record_id",
            "image_sha256",
            "image_path",
            "category_l1",
            "color_families",
        }
    )
    candidate_keys = frozenset(
        {
            "asset_id",
            "product_id",
            "source_dataset",
            "source_record_id",
            "image_sha256",
            "image_path",
            "category_l1",
            "display_title",
            "audience",
            "feature_tags",
            "color_families",
        }
    )
    facet_keys = frozenset({"facet", "value", "confidence"})
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previous_edge_id = ""
    seen_edge_ids: set[str] = set()
    for ordinal, raw_edge in enumerate(raw_edges, 1):
        edge = _exact_graph_keys(raw_edge, edge_keys, label=f"edge {ordinal}")
        edge_id = edge["edge_id"]
        confidence = edge["confidence"]
        if (
            not isinstance(edge_id, str)
            or not edge_id
            or edge_id in seen_edge_ids
            or edge_id <= previous_edge_id
            or edge["relation_kind"] != STYLE_COORDINATION_RELATION
            or edge["annotation_policy_version"] != STYLE_COORDINATION_ANNOTATION_POLICY
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise GateEvidenceError(
                f"Style coordination graph edge {ordinal} is invalid"
            )
        previous_edge_id = edge_id
        seen_edge_ids.add(edge_id)

        anchor = _exact_graph_keys(
            edge["anchor"], anchor_keys, label=f"edge {ordinal} anchor"
        )
        candidate = _exact_graph_keys(
            edge["candidate"], candidate_keys, label=f"edge {ordinal} candidate"
        )
        anchor_path = anchor["image_path"]
        selection = by_destination.get(anchor_path)
        draft = selection.get("draft") if isinstance(selection, dict) else None
        if (
            not isinstance(anchor["asset_id"], str)
            or not anchor["asset_id"]
            or not isinstance(anchor_path, str)
            or not isinstance(draft, dict)
            or draft.get("source_dataset") != anchor["source_dataset"]
            or draft.get("source_record_id") != anchor["source_record_id"]
            or (
                isinstance(draft.get("product_id"), str)
                and draft["product_id"] != anchor["product_id"]
            )
            or candidate["category_l1"] not in STYLE_COORDINATION_CATEGORIES
            or candidate["category_l1"] == anchor["category_l1"]
            or candidate["audience"] not in {"women", "unisex"}
        ):
            raise GateEvidenceError(
                f"Style coordination graph edge {ordinal} identity is invalid"
            )
        if any(
            not isinstance(candidate.get(field), str) or not candidate[field]
            for field in (
                "asset_id",
                "product_id",
                "source_dataset",
                "source_record_id",
                "image_sha256",
                "image_path",
                "display_title",
            )
        ):
            raise GateEvidenceError(
                f"Style coordination graph edge {ordinal} candidate is incomplete"
            )
        raw_facets = edge["facets"]
        if not isinstance(raw_facets, list) or not raw_facets:
            raise GateEvidenceError(
                f"Style coordination graph edge {ordinal} facets are empty"
            )
        for facet_ordinal, raw_facet in enumerate(raw_facets, 1):
            facet = _exact_graph_keys(
                raw_facet,
                facet_keys,
                label=f"edge {ordinal} facet {facet_ordinal}",
            )
            facet_confidence = facet["confidence"]
            if (
                not isinstance(facet["facet"], str)
                or not facet["facet"]
                or not isinstance(facet["value"], str)
                or not facet["value"]
                or isinstance(facet_confidence, bool)
                or not isinstance(facet_confidence, (int, float))
                or not 0.0 <= float(facet_confidence) <= 1.0
            ):
                raise GateEvidenceError(
                    f"Style coordination graph edge {ordinal} facet is invalid"
                )
        indexed[str(anchor["asset_id"])].append(edge)

    input_record = {
        "logical_path": "core/style-coordination-graph.json",
        "bytes": len(content),
        "sha256": sha256_bytes(content),
        "graph_sha256": observed_graph_sha,
        "edge_count": len(raw_edges),
        "source_bindings": dict(bindings),
    }
    return content, dict(indexed), input_record


def _rank(query_id: str, *, stratum: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{POLICY_VERSION}|{stratum}|{query_id}".encode("utf-8")
    ).hexdigest()
    return digest, query_id


def _sample_unique_assets(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    stratum: str,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: _rank(str(item["query"]["query_id"]), stratum=stratum),
    )
    selected: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    for item in ranked:
        asset_id = item["query"].get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise GateEvidenceError("sample candidate lacks a non-empty asset_id")
        if asset_id in seen_assets:
            continue
        seen_assets.add(asset_id)
        selected.append(item)
        if len(selected) == count:
            return selected
    raise GateEvidenceError(
        f"stratum {stratum} has only {len(selected)} unique assets; needs {count}"
    )


def _load_fashioniq_edges(
    captions_dir: Path,
    images_dir: Path,
    *,
    selected_anchors: set[tuple[str, str]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    input_files: list[dict[str, Any]] = []
    for filename in FASHIONIQ_CAPTION_FILES:
        path = captions_dir / filename
        content = _read_small(path, label=f"FashionIQ captions {filename}")
        rows = _json_array(content, label=f"FashionIQ captions {filename}")
        category = filename.split(".")[1]
        input_files.append(
            {
                "logical_path": f"fashioniq/captions/{filename}",
                "bytes": len(content),
                "sha256": sha256_bytes(content),
                "row_count": len(rows),
            }
        )
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise GateEvidenceError(f"{filename} row {row_index} is not an object")
            anchor_id = row.get("candidate")
            if (category, anchor_id) not in selected_anchors:
                continue
            target_id = row.get("target")
            raw_captions = row.get("captions")
            captions = (
                tuple(
                    value.strip()
                    for value in raw_captions
                    if isinstance(value, str) and value.strip()
                )
                if isinstance(raw_captions, list)
                else ()
            )
            # FashionIQ test annotations deliberately omit targets.  They cannot
            # establish a non-speculative candidate and are excluded.
            if (
                not isinstance(target_id, str)
                or not target_id
                or target_id == anchor_id
                or not captions
            ):
                continue
            target_path = images_dir / category / f"{target_id}.jpg"
            if not target_path.is_file():
                continue
            edges[(category, str(anchor_id))].append(
                {
                    "category": category,
                    "anchor_id": anchor_id,
                    "target_id": target_id,
                    "captions": captions,
                    "source_file": filename,
                    "source_file_sha256": sha256_bytes(content),
                    "source_row_index": row_index,
                    "target_path": target_path,
                }
            )
    return dict(edges), input_files


def _select_edge(query_id: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
    def edge_rank(edge: dict[str, Any]) -> tuple[str, str]:
        payload = "|".join(
            (
                POLICY_VERSION,
                "style.edge",
                query_id,
                str(edge["category"]),
                str(edge["anchor_id"]),
                str(edge["target_id"]),
                str(edge["source_file"]),
                str(edge["source_row_index"]),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), str(
            edge["target_id"]
        )

    return min(edges, key=edge_rank)


def _matching_coordination_edges(
    *,
    query: dict[str, Any],
    selection: dict[str, Any],
    coordination_edges: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    text = str(query["text"])
    requested = _requested_style_coordination_categories(text)
    allowed = _style_coordination_allowed_categories(text)
    if not allowed:
        return []
    draft = selection.get("draft")
    if not isinstance(draft, dict):
        raise GateEvidenceError("Style selection lacks a draft binding")
    asset_id = str(query["asset_id"])
    by_category: dict[str, list[dict[str, Any]]] = {}
    for edge in coordination_edges.get(asset_id, ()):
        if (
            edge["anchor"]["asset_id"] != asset_id
            or edge["anchor"]["image_path"] != query["image_path"]
            or edge["anchor"]["source_dataset"] != draft.get("source_dataset")
            or edge["anchor"]["source_record_id"] != draft.get("source_record_id")
            or not _style_coordination_candidate_matches(text, edge["candidate"])
        ):
            continue
        category = str(edge["candidate"]["category_l1"])
        by_category.setdefault(category, []).append(edge)
    for rows in by_category.values():
        rows.sort(
            key=lambda edge: (
                -round(float(edge["confidence"]), 8),
                str(edge["candidate"]["product_id"]),
                str(edge["edge_id"]),
            )
        )
    # Runtime's coverage gate applies to every allowed family, including a
    # family introduced only by a constrained negative clause (for example,
    # "搭小包；鞋跟不要超过三厘米").  A ceiling that checked only the
    # positive request could incorrectly count a partial result as mapped.
    if requested and any(category not in by_category for category in allowed):
        return []

    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < 5:
        added = False
        for category in allowed:
            rows = by_category.get(category, ())
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == 5:
                    break
        if not added:
            break
        offset += 1
    return selected


def _style_samples(
    queries: list[dict[str, Any]],
    by_destination: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str], list[dict[str, Any]]],
    coordination_edges: dict[str, list[dict[str, Any]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    eligible_pool: list[dict[str, Any]] = []
    mapped_pool: list[dict[str, Any]] = []
    uncovered_pool: list[dict[str, Any]] = []
    runtime_coordination_prefilter: list[dict[str, Any]] = []
    style_queries = [
        query
        for query in queries
        if query.get("canonical_capability") == "product.style_recommendation"
        and query.get("split") in STYLE_CEILING_SPLITS
    ]
    for query in style_queries:
        selection = _bind_query_selection(query, by_destination)
        text = query.get("text")
        if not isinstance(text, str):
            raise GateEvidenceError(f"Style query {query.get('query_id')} lacks text")
        if _is_style_coordination_query(text):
            item = {
                "query": query,
                "selection": selection,
                "anchor": _fashioniq_anchor(selection),
                "requested_target_categories": (
                    _requested_style_coordination_categories(text)
                ),
                "negated_target_categories": tuple(
                    sorted(_negated_style_coordination_categories(text))
                ),
                "allowed_target_categories": (
                    _style_coordination_allowed_categories(text)
                ),
            }
            runtime_coordination_prefilter.append(item)
            item["coordination_edges"] = _matching_coordination_edges(
                query=query,
                selection=selection,
                coordination_edges=coordination_edges,
            )
            if item["coordination_edges"]:
                mapped_pool.append(item)
            else:
                uncovered_pool.append(item)
            continue
        anchor = _fashioniq_anchor(selection)
        if anchor is not None and edges.get(anchor):
            eligible_pool.append(
                {
                    "query": query,
                    "selection": selection,
                    "anchor": anchor,
                }
            )

    eligible = _sample_unique_assets(
        eligible_pool,
        count=STYLE_ELIGIBLE_SAMPLE_SIZE,
        stratum="style.same_category_eligible",
    )
    mapped = sorted(
        mapped_pool,
        key=lambda item: _rank(
            str(item["query"]["query_id"]),
            stratum="style.cross_category_coordination_mapped",
        ),
    )
    uncovered = sorted(
        uncovered_pool,
        key=lambda item: _rank(
            str(item["query"]["query_id"]),
            stratum="style.cross_category_coordination_uncovered",
        ),
    )
    if not mapped:
        raise GateEvidenceError(
            "cross-category mapped stratum has no exact graph-backed asset"
        )
    if len(runtime_coordination_prefilter) != len(mapped_pool) + len(uncovered_pool):
        raise GateEvidenceError(
            "runtime coordination prefilter does not partition into mapped/uncovered"
        )
    pool_counts = {
        "style_query_count_in_dev_mini_and_opt_pool": len(style_queries),
        "same_category_eligible_query_count": len(eligible_pool),
        "same_category_eligible_unique_asset_count": len(
            {str(item["query"]["asset_id"]) for item in eligible_pool}
        ),
        "runtime_coordination_prefilter_query_count": len(
            runtime_coordination_prefilter
        ),
        "runtime_coordination_prefilter_unique_asset_count": len(
            {str(item["query"]["asset_id"]) for item in runtime_coordination_prefilter}
        ),
        "cross_category_mapped_query_count": len(mapped_pool),
        "cross_category_mapped_unique_asset_count": len(
            {str(item["query"]["asset_id"]) for item in mapped_pool}
        ),
        "cross_category_uncovered_query_count": len(uncovered_pool),
        "cross_category_uncovered_unique_asset_count": len(
            {str(item["query"]["asset_id"]) for item in uncovered_pool}
        ),
        "runtime_coordination_unconfirmed_query_count": 0,
    }
    return eligible, mapped, uncovered, pool_counts


def _build_style_results(
    eligible: list[dict[str, Any]],
    mapped: list[dict[str, Any]],
    uncovered: list[dict[str, Any]],
    edges: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    coordination_graph_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eligible_results: list[dict[str, Any]] = []
    for item in eligible:
        query = item["query"]
        category, anchor_id = item["anchor"]
        edge = _select_edge(str(query["query_id"]), edges[(category, anchor_id)])
        target_content = _read_small(
            Path(edge["target_path"]),
            label=f"FashionIQ target image {edge['target_id']}",
        )
        hit = {
            "candidate_id": f"fashioniq:{edge['target_id']}",
            "category_l1": category,
            "facets": [
                {
                    "name": "relative_style_change",
                    "values": list(edge["captions"]),
                }
            ],
            "provenance": {
                "dataset": "fashioniq",
                "relation": "relative_caption",
                "source_record_id": f"{category}:{anchor_id}->{edge['target_id']}",
                "source_file": edge["source_file"],
                "source_file_sha256": edge["source_file_sha256"],
                "source_row_index": edge["source_row_index"],
                "target_image_logical_path": (
                    f"fashioniq/images/{category}/{edge['target_id']}.jpg"
                ),
                "target_image_bytes": len(target_content),
                "target_image_sha256": sha256_bytes(target_content),
            },
        }
        eligible_results.append(
            {
                "query_id": query["query_id"],
                "asset_id": query["asset_id"],
                "stratum": "same_category_eligible",
                "status": "supported",
                "anchor": {
                    "category_l1": category,
                    "source_record_id": f"{category}:{anchor_id}",
                },
                "hits": [hit],
                "counted_in_eligible_hit_rate": True,
            }
        )

    mapped_results: list[dict[str, Any]] = []
    for item in mapped:
        query = item["query"]
        first_edge = item["coordination_edges"][0]
        hits = []
        for edge in item["coordination_edges"]:
            candidate = edge["candidate"]
            hits.append(
                {
                    "candidate_id": candidate["product_id"],
                    "category_l1": candidate["category_l1"],
                    "facets": [dict(facet) for facet in edge["facets"]],
                    "provenance": {
                        "dataset": candidate["source_dataset"],
                        "relation": STYLE_COORDINATION_RELATION,
                        "edge_id": edge["edge_id"],
                        "graph_sha256": coordination_graph_sha256,
                        "candidate_source_record_id": candidate["source_record_id"],
                        "candidate_image_sha256": candidate["image_sha256"],
                    },
                }
            )
        mapped_results.append(
            {
                "query_id": query["query_id"],
                "asset_id": query["asset_id"],
                "stratum": "cross_category_coordination_mapped",
                "status": "supported",
                "anchor": {
                    "category_l1": first_edge["anchor"]["category_l1"],
                    "source_record_id": first_edge["anchor"]["source_record_id"],
                },
                "requested_target_categories": list(
                    item["requested_target_categories"]
                ),
                "allowed_target_categories": list(item["allowed_target_categories"]),
                "semantic_evidence": {
                    "runtime_classifier": True,
                    "runtime_policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                    "allowed_target_categories": list(
                        item["allowed_target_categories"]
                    ),
                    "negated_target_categories": list(
                        item["negated_target_categories"]
                    ),
                },
                "hits": hits,
                "counted_in_eligible_hit_rate": True,
            }
        )

    uncovered_results = [
        {
            "query_id": item["query"]["query_id"],
            "asset_id": item["query"]["asset_id"],
            "stratum": "cross_category_coordination_uncovered",
            "status": "uncovered",
            "unsupported_reason": "no_exact_anchor_to_requested_category_edge",
            "semantic_evidence": {
                "runtime_classifier": True,
                "runtime_policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                "anchor_dataset": item["selection"]["draft"].get("source_dataset"),
                "anchor_category_l1": (
                    item["anchor"][0] if item["anchor"] is not None else None
                ),
                "anchor_source_record_id": item["selection"]["draft"].get(
                    "source_record_id"
                ),
                "requested_target_categories": list(
                    item["requested_target_categories"]
                ),
                "allowed_target_categories": list(item["allowed_target_categories"]),
                "negated_target_categories": list(item["negated_target_categories"]),
            },
            "hits": [],
            "counted_in_eligible_hit_rate": False,
        }
        for item in uncovered
    ]
    return eligible_results, mapped_results, uncovered_results


def _load_rpc_annotations(
    archive_path: Path,
    adapter_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_content = _read_small(
        adapter_manifest_path,
        label="RPC adapter manifest",
    )
    manifest = _json_object(manifest_content, label="RPC adapter manifest")
    archives = manifest.get("archives")
    if not isinstance(archives, list) or len(archives) != 1:
        raise GateEvidenceError("RPC adapter manifest must bind exactly one archive")
    archive_record = archives[0]
    if not isinstance(archive_record, dict):
        raise GateEvidenceError("RPC archive binding must be an object")
    declared_size = archive_record.get("bytes")
    declared_sha256 = archive_record.get("sha256")
    if (
        not isinstance(declared_size, int)
        or declared_size <= 0
        or not isinstance(declared_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", declared_sha256)
    ):
        raise GateEvidenceError("RPC adapter archive binding is invalid")
    try:
        observed_size = archive_path.stat().st_size
    except OSError as exc:
        raise GateEvidenceError(
            f"unable to inspect RPC archive: {archive_path}"
        ) from exc
    if not archive_path.is_file() or observed_size != declared_size:
        raise GateEvidenceError("RPC archive size does not match its adapter manifest")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(RPC_ANNOTATION_MEMBER)
            if info.flag_bits & 0x1:
                raise GateEvidenceError("RPC annotation member must not be encrypted")
            member_content = archive.read(info)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise GateEvidenceError("unable to read the RPC annotation member") from exc
    annotation = _json_object(member_content, label="RPC annotation member")
    input_record = {
        "adapter_manifest": {
            "logical_path": "rpc-core-v1/adapter-run.json",
            "bytes": len(manifest_content),
            "sha256": sha256_bytes(manifest_content),
        },
        "archive": {
            "logical_path": archive_record.get("logical_path"),
            "bytes_observed": observed_size,
            "sha256_declared_by_adapter_manifest": declared_sha256,
            "full_archive_sha256_reverified": False,
        },
        "annotation_member": {
            "member": RPC_ANNOTATION_MEMBER,
            "bytes": len(member_content),
            "sha256": sha256_bytes(member_content),
            "crc32": f"{info.CRC:08x}",
        },
    }
    return annotation, input_record


def _rpc_image_id(selection: dict[str, Any]) -> int:
    draft = selection.get("draft")
    if not isinstance(draft, dict) or draft.get("source_dataset") != "rpc":
        raise GateEvidenceError("Multi sample must be bound to an RPC selection")
    source_record_id = draft.get("source_record_id")
    if not isinstance(source_record_id, str) or not source_record_id.startswith(
        "val2019:"
    ):
        raise GateEvidenceError("RPC selection has an invalid source_record_id")
    try:
        return int(source_record_id.split(":", 1)[1])
    except ValueError as exc:
        raise GateEvidenceError(
            "RPC source_record_id has a non-integer image id"
        ) from exc


def _multi_sample(
    queries: list[dict[str, Any]],
    by_destination: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pool: list[dict[str, Any]] = []
    for query in queries:
        if query.get("canonical_capability") != "product.multi_search":
            continue
        selection = _bind_query_selection(query, by_destination)
        image_id = _rpc_image_id(selection)
        pool.append({"query": query, "selection": selection, "image_id": image_id})
    selected = _sample_unique_assets(
        pool,
        count=MULTI_SAMPLE_SIZE,
        stratum="multi.annotation_oracle",
    )
    return selected, {
        "multi_query_count": len(pool),
        "multi_unique_asset_count": len(
            {str(item["query"]["asset_id"]) for item in pool}
        ),
    }


def _build_multi_results(
    sample: list[dict[str, Any]],
    annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    categories_raw = annotation.get("categories")
    images_raw = annotation.get("images")
    annotations_raw = annotation.get("annotations")
    if not all(
        isinstance(value, list)
        for value in (categories_raw, images_raw, annotations_raw)
    ):
        raise GateEvidenceError("RPC annotation member lacks COCO-shaped arrays")

    categories: dict[int, str] = {}
    for row in categories_raw:
        if not isinstance(row, dict):
            raise GateEvidenceError("RPC category must be an object")
        category_id = row.get("id")
        name = row.get("name")
        if (
            not isinstance(category_id, int)
            or category_id in categories
            or not isinstance(name, str)
            or not name.strip()
        ):
            raise GateEvidenceError("RPC category id/name binding is invalid")
        categories[category_id] = name.strip()

    images: set[int] = set()
    for row in images_raw:
        if not isinstance(row, dict) or not isinstance(row.get("id"), int):
            raise GateEvidenceError("RPC image must have an integer id")
        image_id = int(row["id"])
        if image_id in images:
            raise GateEvidenceError(f"duplicate RPC image id: {image_id}")
        images.add(image_id)

    sampled_image_ids = {int(item["image_id"]) for item in sample}
    missing_images = sampled_image_ids - images
    if missing_images:
        raise GateEvidenceError(
            f"sampled RPC image ids are absent: {sorted(missing_images)}"
        )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations_raw:
        if not isinstance(row, dict):
            raise GateEvidenceError("RPC annotation must be an object")
        image_id = row.get("image_id")
        if image_id in sampled_image_ids:
            grouped[int(image_id)].append(row)

    results: list[dict[str, Any]] = []
    for item in sample:
        query = item["query"]
        image_id = int(item["image_id"])
        source_annotations = sorted(
            grouped.get(image_id, []),
            key=lambda row: (
                row.get("id") if isinstance(row.get("id"), int) else 2**63,
                row.get("category_id")
                if isinstance(row.get("category_id"), int)
                else 2**63,
                json.dumps(row.get("bbox"), sort_keys=True, separators=(",", ":")),
            ),
        )
        if not source_annotations:
            raise GateEvidenceError(f"RPC image {image_id} has no annotations")

        objects: list[dict[str, Any]] = []
        item_refs_by_category: dict[int, list[str]] = defaultdict(list)
        for ordinal, source in enumerate(source_annotations, start=1):
            item_ref = f"item-{ordinal:03d}"
            category_id = source.get("category_id")
            matched = isinstance(category_id, int) and category_id in categories
            resolution_status = "matched" if matched else "unresolved"
            object_row: dict[str, Any] = {
                "item_ref": item_ref,
                "resolution_status": resolution_status,
                "annotation_id": source.get("id"),
                "bbox": source.get("bbox"),
                "category_id": category_id if isinstance(category_id, int) else None,
                "object_type": categories[category_id] if matched else None,
            }
            if not matched:
                object_row["unresolved_reason"] = "category_not_bound_in_rpc_taxonomy"
            else:
                item_refs_by_category[int(category_id)].append(item_ref)
            objects.append(object_row)

        candidates: list[dict[str, Any]] = []
        for category_id in sorted(item_refs_by_category):
            item_refs = item_refs_by_category[category_id]
            candidate_id = f"rpc-category:{category_id}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "category_id": category_id,
                    "category_name": categories[category_id],
                    "item_refs": item_refs,
                    "quantity": len(item_refs),
                    "provenance": {
                        "dataset": "rpc",
                        "annotation_member": RPC_ANNOTATION_MEMBER,
                        "image_id": image_id,
                        "category_id": category_id,
                    },
                    "card": {
                        "candidate_id": candidate_id,
                        "title": categories[category_id],
                        "item_refs": item_refs,
                        "quantity": len(item_refs),
                    },
                }
            )

        results.append(
            {
                "query_id": query["query_id"],
                "asset_id": query["asset_id"],
                "source_record_id": f"val2019:{image_id}",
                "status": "supported",
                "objects": objects,
                "candidates": candidates,
            }
        )
    return results


def _rate(flags: Iterable[bool]) -> float:
    values = list(flags)
    if not values:
        raise GateEvidenceError("a gate rate cannot have an empty denominator")
    return sum(values) / len(values)


def _style_metrics(
    eligible: list[dict[str, Any]],
    mapped: list[dict[str, Any]],
    uncovered: list[dict[str, Any]],
) -> dict[str, float]:
    provenance_fields = {
        "dataset",
        "relation",
        "source_record_id",
        "source_file",
        "source_file_sha256",
        "source_row_index",
        "target_image_logical_path",
        "target_image_bytes",
        "target_image_sha256",
    }
    hits = [hit for row in eligible for hit in row["hits"]]
    mapped_hits = [hit for row in mapped for hit in row["hits"]]
    mapped_provenance_fields = {
        "dataset",
        "relation",
        "edge_id",
        "graph_sha256",
        "candidate_source_record_id",
        "candidate_image_sha256",
    }
    semantic_rows = [*mapped, *uncovered]
    return {
        "style_same_category_nonempty_rate": _rate(
            bool(row["hits"]) for row in eligible
        ),
        "style_same_category_category_consistency_rate": _rate(
            hit.get("category_l1") == row["anchor"].get("category_l1")
            for row in eligible
            for hit in row["hits"]
        ),
        "style_same_category_facet_complete_rate": _rate(
            bool(hit.get("facets"))
            and all(
                isinstance(facet, dict)
                and isinstance(facet.get("name"), str)
                and bool(facet["name"])
                and isinstance(facet.get("values"), list)
                and bool(facet["values"])
                and all(isinstance(value, str) and value for value in facet["values"])
                for facet in hit.get("facets", [])
            )
            for hit in hits
        ),
        "style_same_category_provenance_complete_rate": _rate(
            isinstance(hit.get("provenance"), dict)
            and provenance_fields.issubset(hit["provenance"])
            and all(
                hit["provenance"][field] not in (None, "")
                for field in provenance_fields
            )
            for hit in hits
        ),
        "style_same_category_non_speculative_candidate_rate": _rate(
            hit.get("provenance", {}).get("dataset") == "fashioniq"
            and hit.get("provenance", {}).get("relation") == "relative_caption"
            and isinstance(hit.get("candidate_id"), str)
            and hit["candidate_id"].startswith("fashioniq:")
            for hit in hits
        ),
        "style_cross_category_mapped_nonempty_rate": _rate(
            bool(row.get("hits")) for row in mapped
        ),
        "style_cross_category_mapped_category_consistency_rate": _rate(
            hit.get("category_l1") in row.get("allowed_target_categories", [])
            and hit.get("category_l1") != row.get("anchor", {}).get("category_l1")
            for row in mapped
            for hit in row["hits"]
        ),
        "style_cross_category_mapped_requested_family_complete_rate": _rate(
            set(row.get("requested_target_categories", [])).issubset(
                {str(hit.get("category_l1")) for hit in row["hits"]}
            )
            for row in mapped
        ),
        "style_cross_category_mapped_facet_complete_rate": _rate(
            isinstance(hit.get("facets"), list)
            and bool(hit["facets"])
            and all(
                isinstance(facet, dict)
                and set(facet) == {"facet", "value", "confidence"}
                and isinstance(facet["facet"], str)
                and bool(facet["facet"])
                and isinstance(facet["value"], str)
                and bool(facet["value"])
                for facet in hit["facets"]
            )
            for hit in mapped_hits
        ),
        "style_cross_category_mapped_provenance_complete_rate": _rate(
            isinstance(hit.get("provenance"), dict)
            and mapped_provenance_fields.issubset(hit["provenance"])
            and all(
                hit["provenance"][field] not in (None, "")
                for field in mapped_provenance_fields
            )
            for hit in mapped_hits
        ),
        "style_cross_category_mapped_non_speculative_candidate_rate": _rate(
            hit.get("provenance", {}).get("relation") == STYLE_COORDINATION_RELATION
            and hit.get("provenance", {}).get("dataset") == "abo"
            and isinstance(hit.get("candidate_id"), str)
            and bool(hit["candidate_id"])
            for hit in mapped_hits
        ),
        "style_cross_category_uncovered_zero_hit_rate": (
            _rate(
                row.get("status") == "uncovered" and row.get("hits") == []
                for row in uncovered
            )
            if uncovered
            else 1.0
        ),
        "style_cross_category_uncovered_not_counted_rate": (
            _rate(row.get("counted_in_eligible_hit_rate") is False for row in uncovered)
            if uncovered
            else 1.0
        ),
        "style_cross_category_semantic_evidence_rate": _rate(
            isinstance(row.get("semantic_evidence"), dict)
            and row["semantic_evidence"].get("runtime_classifier") is True
            and row["semantic_evidence"].get("runtime_policy_version")
            == PORTFOLIO_TOOL_RUNTIME_POLICY
            and isinstance(
                row["semantic_evidence"].get("allowed_target_categories"), list
            )
            and isinstance(
                row["semantic_evidence"].get("negated_target_categories"), list
            )
            for row in semantic_rows
        ),
    }


def _multi_metrics(
    results: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    objects = [obj for row in results for obj in row["objects"]]
    candidates = [candidate for row in results for candidate in row["candidates"]]

    dedup_flags: list[bool] = []
    item_alignment_flags: list[bool] = []
    quantity_flags: list[bool] = []
    card_flags: list[bool] = []
    non_speculative_flags: list[bool] = []
    for row in results:
        row_candidates = row["candidates"]
        keys = [candidate["category_id"] for candidate in row_candidates]
        candidate_refs = [
            item_ref
            for candidate in row_candidates
            for item_ref in candidate["item_refs"]
        ]
        matched_refs = [
            obj["item_ref"]
            for obj in row["objects"]
            if obj["resolution_status"] == "matched"
        ]
        unresolved_refs = {
            obj["item_ref"]
            for obj in row["objects"]
            if obj["resolution_status"] == "unresolved"
        }
        dedup_flags.append(len(keys) == len(set(keys)))
        item_alignment_flags.append(
            sorted(candidate_refs) == sorted(matched_refs)
            and len(candidate_refs) == len(set(candidate_refs))
            and not unresolved_refs.intersection(candidate_refs)
        )
        quantity_flags.extend(
            candidate["quantity"] == len(candidate["item_refs"])
            for candidate in row_candidates
        )
        card_flags.extend(
            candidate.get("card", {}).get("candidate_id") == candidate["candidate_id"]
            and candidate.get("card", {}).get("quantity") == candidate["quantity"]
            and candidate.get("card", {}).get("item_refs") == candidate["item_refs"]
            for candidate in row_candidates
        )
        non_speculative_flags.extend(
            candidate.get("provenance", {}).get("dataset") == "rpc"
            and candidate.get("provenance", {}).get("image_id")
            == int(row["source_record_id"].split(":", 1)[1])
            and candidate.get("provenance", {}).get("category_id")
            == candidate.get("category_id")
            for candidate in row_candidates
        )

    metrics = {
        "multi_nonempty_scene_rate": _rate(bool(row["objects"]) for row in results),
        "multi_object_resolution_status_typed_rate": _rate(
            obj.get("resolution_status") in {"matched", "unresolved"} for obj in objects
        ),
        "multi_matched_object_type_complete_rate": _rate(
            obj.get("resolution_status") != "matched"
            or isinstance(obj.get("object_type"), str)
            and bool(obj["object_type"])
            for obj in objects
        ),
        "multi_candidate_dedup_rate": _rate(dedup_flags),
        "multi_item_refs_alignment_rate": _rate(item_alignment_flags),
        "multi_quantity_alignment_rate": _rate(quantity_flags),
        "multi_card_alignment_rate": _rate(card_flags),
        "multi_non_speculative_candidate_rate": _rate(non_speculative_flags),
    }
    counts = {
        "object_count": len(objects),
        "matched_object_count": sum(
            obj["resolution_status"] == "matched" for obj in objects
        ),
        "unresolved_object_count": sum(
            obj["resolution_status"] == "unresolved" for obj in objects
        ),
        "candidate_count_after_category_dedup": len(candidates),
    }
    return metrics, counts


def _checks(metrics: dict[str, float]) -> list[dict[str, Any]]:
    if set(metrics) != set(THRESHOLDS):
        raise GateEvidenceError("metric and threshold sets differ")
    return [
        {
            "metric": metric,
            "observed": metrics[metric],
            "operator": ">=",
            "threshold": THRESHOLDS[metric],
            "passed": metrics[metric] >= THRESHOLDS[metric],
        }
        for metric in sorted(metrics)
    ]


def build_receipt(
    *,
    queries_path: Path,
    selection_path: Path,
    style_coordination_graph_path: Path,
    fashioniq_captions_dir: Path,
    fashioniq_images_dir: Path,
    rpc_archive_path: Path,
    rpc_adapter_manifest_path: Path,
) -> dict[str, Any]:
    query_content, queries = _load_queries(queries_path)
    selection_content, selections = _load_selection(selection_path)
    by_destination = _selection_by_destination(selections)
    (
        _coordination_graph_content,
        coordination_edges,
        coordination_graph_input,
    ) = _load_style_coordination_graph(
        style_coordination_graph_path,
        selection_content=selection_content,
        by_destination=by_destination,
    )

    style_queries = [
        query
        for query in queries
        if query.get("canonical_capability") == "product.style_recommendation"
        and query.get("split") in STYLE_CEILING_SPLITS
    ]
    selected_anchors = {
        anchor
        for query in style_queries
        if (anchor := _fashioniq_anchor(_bind_query_selection(query, by_destination)))
        is not None
    }
    edges, caption_inputs = _load_fashioniq_edges(
        fashioniq_captions_dir,
        fashioniq_images_dir,
        selected_anchors=selected_anchors,
    )
    eligible, mapped, uncovered, style_pool_counts = _style_samples(
        queries,
        by_destination,
        edges,
        coordination_edges,
    )
    eligible_results, mapped_results, uncovered_results = _build_style_results(
        eligible,
        mapped,
        uncovered,
        edges,
        coordination_graph_sha256=coordination_graph_input["graph_sha256"],
    )

    multi_sample, multi_pool_counts = _multi_sample(queries, by_destination)
    rpc_annotations, rpc_input = _load_rpc_annotations(
        rpc_archive_path,
        rpc_adapter_manifest_path,
    )
    multi_results = _build_multi_results(multi_sample, rpc_annotations)

    style_metric_values = _style_metrics(
        eligible_results,
        mapped_results,
        uncovered_results,
    )
    multi_metric_values, multi_counts = _multi_metrics(multi_results)
    metrics = {**style_metric_values, **multi_metric_values}
    checks = _checks(metrics)

    target_images: dict[str, dict[str, Any]] = {}
    for row in eligible_results:
        for hit in row["hits"]:
            provenance = hit["provenance"]
            logical_path = provenance["target_image_logical_path"]
            target_images[logical_path] = {
                "logical_path": logical_path,
                "bytes": provenance["target_image_bytes"],
                "sha256": provenance["target_image_sha256"],
            }

    script_content = Path(__file__).read_bytes()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "core-style-multi-tool-ceiling-receipt",
        "policy_version": POLICY_VERSION,
        "evidence_kind": EVIDENCE_KIND,
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "gate0_passed": all(check["passed"] for check in checks),
        "claim_boundary": {
            "proves": (
                "local annotation and output-contract tool ceiling, including "
                "exact graph-backed cross-category evidence availability"
            ),
            "does_not_prove": [
                "detector_effectiveness",
                "retrieval_effectiveness",
                "assistant_effectiveness",
                "end_to_end_model_gain",
                "human_preference_or_outfit_quality",
            ],
            "formal_eligible": False,
            "model_effect_claimed": False,
        },
        "execution": {
            "mode": "local",
            "provider_calls_performed": 0,
            "dashscope_calls_performed": 0,
            "network_calls_performed": 0,
        },
        "sampling_policy": {
            "rank": "sha256(policy_version|stratum|query_id), then query_id",
            "style": {
                "eligible_splits": sorted(STYLE_CEILING_SPLITS),
                "test_frozen_text_used": False,
                "same_category_eligible": STYLE_ELIGIBLE_SAMPLE_SIZE,
                "same_category_unique_asset_required": True,
                "cross_category_coordination": (
                    "census of every runtime-classified cross query in eligible "
                    "splits, partitioned by exact constrained graph coverage"
                ),
                "runtime_classifier": {
                    "module": "skillchain.tools.portfolio_runtime",
                    "runtime_policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                    "classification_helper": "_is_style_coordination_query",
                    "requested_helper": ("_requested_style_coordination_categories"),
                    "negated_helper": "_negated_style_coordination_categories",
                    "allowed_helper": "_style_coordination_allowed_categories",
                    "candidate_helper": "_style_coordination_candidate_matches",
                },
                "uncovered_counted_as_hit": False,
                "design": "targeted_stratified_ceiling_not_population_estimate",
            },
            "multi": {
                "sample_size": MULTI_SAMPLE_SIZE,
                "stratum": "rpc_annotation_oracle",
                "design": "deterministic_unique_asset_ceiling_sample",
            },
        },
        "thresholds": dict(sorted(THRESHOLDS.items())),
        "inputs": {
            "core_queries": {
                "logical_path": "core/queries.jsonl",
                "bytes": len(query_content),
                "sha256": sha256_bytes(query_content),
                "row_count": len(queries),
            },
            "core_selection": {
                "logical_path": "core/selection-manifest.json",
                "bytes": len(selection_content),
                "sha256": sha256_bytes(selection_content),
                "selection_count": len(selections),
            },
            "style_coordination_graph": coordination_graph_input,
            "fashioniq_caption_files": sorted(
                caption_inputs,
                key=lambda item: item["logical_path"],
            ),
            "fashioniq_sampled_target_images": [
                target_images[key] for key in sorted(target_images)
            ],
            "rpc": rpc_input,
            "implementation": {
                "logical_path": "scripts/build_core_style_multi_tool_ceiling.py",
                "bytes": len(script_content),
                "sha256": sha256_bytes(script_content),
            },
        },
        "style": {
            "pool_counts": style_pool_counts,
            "sample_ids": {
                "same_category_eligible": [row["query_id"] for row in eligible_results],
                "cross_category_coordination_mapped": [
                    row["query_id"] for row in mapped_results
                ],
                "cross_category_coordination_uncovered": [
                    row["query_id"] for row in uncovered_results
                ],
            },
            "same_category_eligible": {
                "sample_count": len(eligible_results),
                "results": eligible_results,
                "metrics": {
                    key: value
                    for key, value in style_metric_values.items()
                    if key.startswith("style_same_category_")
                },
            },
            "cross_category_coordination_mapped": {
                "sample_count": len(mapped_results),
                "results": mapped_results,
                "metrics": {
                    key: value
                    for key, value in style_metric_values.items()
                    if key.startswith("style_cross_category_mapped_")
                },
            },
            "cross_category_coordination_uncovered": {
                "sample_count": len(uncovered_results),
                "results": uncovered_results,
                "uncovered_counted_as_hit": False,
                "eligible_hit_numerator_contribution": 0,
                "eligible_hit_denominator_contribution": 0,
                "metrics": {
                    key: value
                    for key, value in style_metric_values.items()
                    if key.startswith("style_cross_category_uncovered_")
                },
            },
            "cross_category_semantic_evidence_rate": style_metric_values[
                "style_cross_category_semantic_evidence_rate"
            ],
        },
        "multi": {
            "pool_counts": multi_pool_counts,
            "sample_ids": [row["query_id"] for row in multi_results],
            "sample_count": len(multi_results),
            "counts": multi_counts,
            "metrics": multi_metric_values,
            "results": multi_results,
        },
        "checks": checks,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--style-coordination-graph", required=True, type=Path)
    parser.add_argument("--fashioniq-captions-dir", required=True, type=Path)
    parser.add_argument("--fashioniq-images-dir", required=True, type=Path)
    parser.add_argument("--rpc-archive", required=True, type=Path)
    parser.add_argument("--rpc-adapter-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_receipt(
        queries_path=args.queries,
        selection_path=args.selection,
        style_coordination_graph_path=args.style_coordination_graph,
        fashioniq_captions_dir=args.fashioniq_captions_dir,
        fashioniq_images_dir=args.fashioniq_images_dir,
        rpc_archive_path=args.rpc_archive,
        rpc_adapter_manifest_path=args.rpc_adapter_manifest,
    )
    atomic_create_file(args.output, canonical_json_bytes(receipt))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": receipt["status"],
                "gate0_passed": receipt["gate0_passed"],
                "receipt_sha256": receipt["receipt_sha256"],
                "provider_calls_performed": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if receipt["gate0_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
