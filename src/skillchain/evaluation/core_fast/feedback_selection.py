from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Iterable, Mapping, Sequence

from skillchain.schemas import Query
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

from .models import AssistantObservation, CAPABILITIES, S1Settings


_COMPONENT_ORDER = (
    "route_acceptable",
    "no_hard_error",
    "tool_contract_pass",
    "evidence_grounded",
    "output_contract_pass",
)


def _hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _self_hashed(payload: dict[str, object], field: str) -> dict[str, object]:
    return {**payload, field: _hash(payload)}


def project_feedback_query(query: Query) -> dict[str, object]:
    """Return only the ordinary provider-visible query semantics."""

    return {
        "query_id": query.query_id,
        "text": query.text,
        "turns": [turn.model_dump(mode="json") for turn in query.turns],
        "canonical_capability": query.canonical_capability,
        "acceptable_capabilities": list(query.acceptable_capabilities),
        "is_boundary": query.is_boundary,
        "boundary_strategy": query.boundary_strategy,
        "requires_card": query.requires_card,
    }


def project_feedback_observation(
    observation: AssistantObservation,
) -> dict[str, object]:
    """Drop replay context, receipts, scorer payloads, and runtime paths."""

    return {
        "query_id": observation.query_id,
        "response_text": observation.response_text,
        "selected_capability": observation.selected_capability,
        "tool_trace": [item.model_dump(mode="json") for item in observation.tool_trace],
        "answer_mode": observation.answer_mode,
        "gcs_components": dict(observation.gcs_components),
        "gcs_reason_codes": list(observation.gcs_reason_codes),
        "gcs_score": observation.gcs_score,
        "hard_error": observation.hard_error,
        "card_violation": observation.card_violation,
        "evidence_violation": observation.evidence_violation,
        "tool_violation": observation.tool_violation,
        "repair": observation.repair,
    }


def _failure_cluster(
    query: Query, observation: AssistantObservation
) -> dict[str, object]:
    failed_components = [
        component
        for component in _COMPONENT_ORDER
        if not observation.gcs_components[component]
    ]
    reasons = list(observation.gcs_reason_codes)
    trace = list(observation.tool_trace)
    tool_sequence = [item.tool_name for item in trace]
    invalid_arguments = sum(item.error_code == "invalid_arguments" for item in trace)
    tool_runtime_errors = sum(
        item.status == "error" and item.error_code != "invalid_arguments"
        for item in trace
    )
    repair = observation.repair or "none"
    assistant_error_code = repair if repair not in {"none", "success"} else "none"
    key_payload = {
        "capability": query.canonical_capability,
        "primary_gcs_reason_code": reasons[0] if reasons else "success",
        "gcs_component_failures": failed_components,
        "route_acceptable": observation.gcs_components["route_acceptable"],
        "tool_called": bool(trace),
        "tool_name_or_sequence": tool_sequence,
        "assistant_error_code": assistant_error_code,
        "repair_status": repair,
        "source_or_boundary_subtype": (
            query.boundary_strategy or observation.source or "none"
        ),
    }
    return {
        **key_payload,
        "cluster_sha256": _hash(key_payload),
        "tool_invalid_arguments": invalid_arguments,
        "tool_runtime_errors": tool_runtime_errors,
    }


def build_discovery_feedback_population(
    *,
    queries: Sequence[Query],
    observations: Mapping[str, AssistantObservation],
    fold_roles: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    """Project the complete discovery600 into deterministic, non-provider rows."""

    rows: list[dict[str, object]] = []
    for ordinal, query in enumerate(queries):
        if query.split != "opt_pool" or fold_roles.get(query.query_id) != "discovery":
            continue
        observation = observations[query.query_id]
        if not observation.oracle_available:
            raise ValueError(
                f"discovery Feedback input lacks oracle coverage: {query.query_id}"
            )
        cluster = _failure_cluster(query, observation)
        trace = list(observation.tool_trace)
        failed = observation.hard_error or observation.gcs_score < 1.0
        route_ok = observation.gcs_components["route_acceptable"]
        tools_succeeded = bool(trace) and all(
            item.status == "success" for item in trace
        )
        body_fixable = failed and route_ok and tools_succeeded
        rows.append(
            {
                "query_ordinal": ordinal,
                "query_id": query.query_id,
                "asset_id": query.asset_id,
                "leakage_group_id": query.leakage_group_id,
                "capability": query.canonical_capability,
                "role": "failure" if failed else "anchor",
                "selection_class": (
                    "body_fixable_failure"
                    if body_fixable
                    else ("boundary_failure" if failed else "success_anchor")
                ),
                "hard_error": observation.hard_error,
                "gcs_components": dict(observation.gcs_components),
                "gcs_reason_codes": list(observation.gcs_reason_codes),
                "route_acceptable": route_ok,
                "tool_called": bool(trace),
                "tool_sequence": [item.tool_name for item in trace],
                "repeated_tool_call_count": max(
                    0, len(trace) - len({item.tool_name for item in trace})
                ),
                "tool_invalid_arguments": cluster["tool_invalid_arguments"],
                "tool_runtime_errors": cluster["tool_runtime_errors"],
                "response_repair_failure": bool(
                    observation.hard_error and (observation.repair or "none") != "none"
                ),
                "card_error": observation.card_violation,
                "evidence_error": observation.evidence_violation,
                "fallback_error": "fallback_contract_failed"
                in observation.gcs_reason_codes,
                "output_section_error": "output_section_invalid"
                in observation.gcs_reason_codes,
                "cluster": cluster,
            }
        )
    if len(rows) != 600 or len({row["query_id"] for row in rows}) != 600:
        raise ValueError("discovery Feedback population must cover 600 unique rows")
    return tuple(rows)


def build_discovery_failure_summary(
    population: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(population) != 600:
        raise ValueError("discovery summary requires exactly 600 rows")
    capabilities: dict[str, object] = {}
    for capability in CAPABILITIES:
        rows = [row for row in population if row["capability"] == capability]
        component_failures: Counter[str] = Counter()
        reason_codes: Counter[str] = Counter()
        reason_cooccurrence: Counter[str] = Counter()
        tool_sequences: Counter[str] = Counter()
        repeated_tool_calls: Counter[str] = Counter()
        assistant_errors: Counter[str] = Counter()
        repair_statuses: Counter[str] = Counter()
        clusters: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            components = row["gcs_components"]
            assert isinstance(components, dict)
            component_failures.update(
                key for key, passed in components.items() if not passed
            )
            reasons = tuple(str(item) for item in row["gcs_reason_codes"])
            reason_codes.update(reasons)
            reason_cooccurrence.update(
                "+".join((left, right))
                for index, left in enumerate(reasons)
                for right in reasons[index + 1 :]
            )
            sequence = row["tool_sequence"]
            assert isinstance(sequence, list)
            tool_sequences[" -> ".join(str(item) for item in sequence) or "<none>"] += 1
            cluster = row["cluster"]
            assert isinstance(cluster, dict)
            repeated_tool_calls[str(row["repeated_tool_call_count"])] += 1
            assistant_errors[str(cluster["assistant_error_code"])] += 1
            repair_statuses[str(cluster["repair_status"])] += 1
            clusters[str(cluster["cluster_sha256"])].append(str(row["query_id"]))
        capabilities[capability] = {
            "sample_count": len(rows),
            "success_count": sum(row["role"] == "anchor" for row in rows),
            "failure_count": sum(row["role"] == "failure" for row in rows),
            "hard_error_count": sum(bool(row["hard_error"]) for row in rows),
            "route_acceptable_count": sum(
                bool(row["route_acceptable"]) for row in rows
            ),
            "route_unacceptable_count": sum(
                not bool(row["route_acceptable"]) for row in rows
            ),
            "tool_called_count": sum(bool(row["tool_called"]) for row in rows),
            "component_failure_counts": dict(sorted(component_failures.items())),
            "reason_code_counts": dict(sorted(reason_codes.items())),
            "reason_code_cooccurrence_counts": dict(
                sorted(reason_cooccurrence.items())
            ),
            "tool_sequence_counts": dict(sorted(tool_sequences.items())),
            "repeated_tool_call_count_distribution": dict(
                sorted(repeated_tool_calls.items())
            ),
            "assistant_error_code_counts": dict(sorted(assistant_errors.items())),
            "repair_status_counts": dict(sorted(repair_statuses.items())),
            "tool_invalid_arguments_count": sum(
                int(row["tool_invalid_arguments"]) for row in rows
            ),
            "tool_runtime_error_count": sum(
                int(row["tool_runtime_errors"]) for row in rows
            ),
            "response_repair_failure_count": sum(
                bool(row["response_repair_failure"]) for row in rows
            ),
            "card_error_count": sum(bool(row["card_error"]) for row in rows),
            "evidence_error_count": sum(bool(row["evidence_error"]) for row in rows),
            "fallback_error_count": sum(bool(row["fallback_error"]) for row in rows),
            "output_section_error_count": sum(
                bool(row["output_section_error"]) for row in rows
            ),
            "success_anchor_count": sum(row["role"] == "anchor" for row in rows),
            "failure_clusters": [
                {
                    "cluster_sha256": cluster_sha,
                    "sample_count": len(query_ids),
                    "representative_query_ids": query_ids[:3],
                }
                for cluster_sha, query_ids in sorted(clusters.items())
                if any(
                    row["query_id"] in query_ids and row["role"] == "failure"
                    for row in rows
                )
            ],
        }
    unsigned = {
        "schema_version": 1,
        "kind": "core-fast-discovery-failure-summary",
        "population_count": len(population),
        "population_sha256": _hash(list(population)),
        "capabilities": capabilities,
    }
    return _self_hashed(unsigned, "summary_sha256")


def _round_robin(rows: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    buckets: dict[str, deque[Mapping[str, object]]] = defaultdict(deque)
    for row in rows:
        cluster = row["cluster"]
        assert isinstance(cluster, dict)
        buckets[str(cluster["cluster_sha256"])].append(row)
    ordered: list[Mapping[str, object]] = []
    while buckets:
        for key in sorted(tuple(buckets)):
            ordered.append(buckets[key].popleft())
            if not buckets[key]:
                del buckets[key]
    return ordered


def _unique_take(
    candidates: Iterable[Mapping[str, object]],
    count: int,
    *,
    selected: list[Mapping[str, object]],
) -> None:
    query_ids = {row["query_id"] for row in selected}
    asset_ids = {row["asset_id"] for row in selected}
    leakage_ids = {row["leakage_group_id"] for row in selected}
    for row in candidates:
        if len(selected) >= count:
            return
        if (
            row["query_id"] in query_ids
            or row["asset_id"] in asset_ids
            or row["leakage_group_id"] in leakage_ids
        ):
            continue
        selected.append(row)
        query_ids.add(row["query_id"])
        asset_ids.add(row["asset_id"])
        leakage_ids.add(row["leakage_group_id"])


def _select_capability_rows(
    rows: Sequence[Mapping[str, object]],
    count: int,
    *,
    target_focused: bool,
    contrastive: bool = False,
    excluded_query_ids: frozenset[object] = frozenset(),
    excluded_asset_ids: frozenset[object] = frozenset(),
    excluded_leakage_ids: frozenset[object] = frozenset(),
) -> list[Mapping[str, object]]:
    rows = [
        row
        for row in rows
        if row["query_id"] not in excluded_query_ids
        and row["asset_id"] not in excluded_asset_ids
        and row["leakage_group_id"] not in excluded_leakage_ids
    ]
    body = _round_robin(
        row for row in rows if row["selection_class"] == "body_fixable_failure"
    )
    boundary = _round_robin(
        row for row in rows if row["selection_class"] == "boundary_failure"
    )
    anchors = sorted(
        (row for row in rows if row["selection_class"] == "success_anchor"),
        key=lambda row: (int(row["query_ordinal"]), str(row["query_id"])),
    )
    selected: list[Mapping[str, object]] = []
    if contrastive:
        failure_goal = count // 2
        anchor_goal = count - failure_goal
        # Reserve protected successes first. Failure-dense selection can
        # otherwise consume their leakage groups and silently collapse the
        # contrast set back into another mostly-failure packet.
        _unique_take(anchors, anchor_goal, selected=selected)
        _unique_take(body, anchor_goal + failure_goal, selected=selected)
        _unique_take(boundary, anchor_goal + failure_goal, selected=selected)
        if len(selected) < count:
            _unique_take(anchors, count, selected=selected)
    elif target_focused:
        body_goal = round(count * 0.7)
        boundary_goal = round(count * 0.2)
        anchor_goal = count - body_goal - boundary_goal
        _unique_take(body, body_goal, selected=selected)
        _unique_take(boundary, body_goal + boundary_goal, selected=selected)
        _unique_take(
            anchors, body_goal + boundary_goal + anchor_goal, selected=selected
        )
    else:
        _unique_take([*body, *boundary], max(0, count - 1), selected=selected)
        _unique_take(anchors, count, selected=selected)
    fallback = sorted(
        rows,
        key=lambda row: (
            {"body_fixable_failure": 0, "boundary_failure": 1, "success_anchor": 2}[
                str(row["selection_class"])
            ],
            int(row["query_ordinal"]),
            str(row["query_id"]),
        ),
    )
    _unique_take(fallback, count, selected=selected)
    return selected


def select_feedback_samples(
    population: Sequence[Mapping[str, object]], settings: S1Settings
) -> tuple[dict[str, object], ...]:
    total = settings.feedback_total_count
    selected: list[Mapping[str, object]] = []
    if settings.feedback_allocation == "target-focused":
        target = settings.target_capabilities[0]
        selected = _select_capability_rows(
            [row for row in population if row["capability"] == target],
            total,
            target_focused=True,
            contrastive=(
                settings.feedback_selection_policy == "discovery-contrastive-v2"
            ),
        )
        if len(selected) < total:
            fallback = sorted(
                (row for row in population if row["capability"] != target),
                key=lambda row: (
                    CAPABILITIES.index(str(row["capability"])),
                    {
                        "body_fixable_failure": 0,
                        "boundary_failure": 1,
                        "success_anchor": 2,
                    }[str(row["selection_class"])],
                    int(row["query_ordinal"]),
                    str(row["query_id"]),
                ),
            )
            _unique_take(fallback, total, selected=selected)
    elif settings.feedback_selection_policy == "discovery-contrastive-v2":
        quotient, remainder = divmod(total, len(CAPABILITIES))
        quotas = {
            capability: quotient + int(index < remainder)
            for index, capability in enumerate(CAPABILITIES)
        }
        by_capability: dict[str, list[Mapping[str, object]]] = {
            capability: [] for capability in CAPABILITIES
        }
        selected_query_ids: set[object] = set()
        selected_asset_ids: set[object] = set()
        selected_leakage_ids: set[object] = set()

        def take(
            capability: str,
            candidates: Iterable[Mapping[str, object]],
            target_count: int,
        ) -> None:
            for row in candidates:
                if len(by_capability[capability]) >= target_count:
                    return
                if (
                    row["query_id"] in selected_query_ids
                    or row["asset_id"] in selected_asset_ids
                    or row["leakage_group_id"] in selected_leakage_ids
                ):
                    continue
                by_capability[capability].append(row)
                selected_query_ids.add(row["query_id"])
                selected_asset_ids.add(row["asset_id"])
                selected_leakage_ids.add(row["leakage_group_id"])

        populations = {
            capability: [row for row in population if row["capability"] == capability]
            for capability in CAPABILITIES
        }
        desired_failures = {
            capability: min(
                quotas[capability] // 2,
                sum(row["role"] == "failure" for row in populations[capability]),
            )
            for capability in CAPABILITIES
        }
        # Capabilities with few anchors choose first; this deterministic
        # rare-first pass prevents another capability in the same leakage
        # group from consuming every protected success.
        anchor_order = sorted(
            CAPABILITIES,
            key=lambda capability: (
                sum(row["role"] == "anchor" for row in populations[capability]),
                CAPABILITIES.index(capability),
            ),
        )
        for capability in anchor_order:
            anchor_goal = quotas[capability] - desired_failures[capability]
            anchors = sorted(
                (row for row in populations[capability] if row["role"] == "anchor"),
                key=lambda row: (int(row["query_ordinal"]), str(row["query_id"])),
            )
            take(capability, anchors, anchor_goal)
        for capability in CAPABILITIES:
            failure_goal = min(
                quotas[capability],
                len(by_capability[capability]) + desired_failures[capability],
            )
            body = _round_robin(
                row
                for row in populations[capability]
                if row["selection_class"] == "body_fixable_failure"
            )
            boundary = _round_robin(
                row
                for row in populations[capability]
                if row["selection_class"] == "boundary_failure"
            )
            take(capability, body, failure_goal)
            take(capability, boundary, failure_goal)
            fallback = sorted(
                populations[capability],
                key=lambda row: (
                    {
                        "body_fixable_failure": 0,
                        "boundary_failure": 1,
                        "success_anchor": 2,
                    }[str(row["selection_class"])],
                    int(row["query_ordinal"]),
                    str(row["query_id"]),
                ),
            )
            take(capability, fallback, quotas[capability])
        selected = [
            row for capability in CAPABILITIES for row in by_capability[capability]
        ]
    else:
        quotient, remainder = divmod(total, len(CAPABILITIES))
        for index, capability in enumerate(CAPABILITIES):
            quota = quotient + int(index < remainder)
            capability_rows = _select_capability_rows(
                [row for row in population if row["capability"] == capability],
                quota,
                target_focused=False,
                contrastive=(
                    settings.feedback_selection_policy == "discovery-contrastive-v2"
                ),
                excluded_query_ids=frozenset(row["query_id"] for row in selected),
                excluded_asset_ids=frozenset(row["asset_id"] for row in selected),
                excluded_leakage_ids=frozenset(
                    row["leakage_group_id"] for row in selected
                ),
            )
            selected.extend(capability_rows)
    if len(selected) != total:
        raise ValueError("discovery population cannot satisfy Feedback quota")
    if (
        len({row["query_id"] for row in selected}) != total
        or len({row["asset_id"] for row in selected}) != total
        or len({row["leakage_group_id"] for row in selected}) != total
    ):
        raise ValueError("Feedback selection uniqueness constraints are unsatisfied")
    return tuple(dict(row) for row in selected)


def build_feedback_selection_manifest(
    *,
    population: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
    settings: S1Settings,
    opt_static_sha256: str,
    opt_fold_mapping_sha256: str,
) -> dict[str, object]:
    role_counts = Counter(str(row["selection_class"]) for row in selected)
    capability_counts = Counter(str(row["capability"]) for row in selected)
    cluster_counts = Counter(
        str(row["cluster"]["cluster_sha256"])  # type: ignore[index]
        for row in selected
    )
    unsigned = {
        "schema_version": 1,
        "kind": "core-fast-feedback-selection-manifest",
        "round_id": settings.round_id,
        "discovery_population_count": len(population),
        "discovery_population_sha256": _hash(list(population)),
        "opt_fold_mapping_sha256": opt_fold_mapping_sha256,
        "opt_static_sha256": opt_static_sha256,
        "selection_policy": settings.feedback_selection_policy,
        "requested_count": settings.feedback_total_count,
        "effective_count": len(selected),
        "canary_count": settings.feedback_canary_count,
        "allocation": settings.feedback_allocation,
        "target_capabilities": list(settings.target_capabilities),
        "selected_query_ids": [row["query_id"] for row in selected],
        "selected_samples": [
            {
                "selection_ordinal": index,
                "query_id": row["query_id"],
                "capability": row["capability"],
                "role": row["role"],
                "selection_class": row["selection_class"],
                "cluster_sha256": row["cluster"]["cluster_sha256"],  # type: ignore[index]
                "failure_cluster": row["cluster"],
                "selection_reason": ("cluster_round_robin_then_frozen_quota_fallback"),
            }
            for index, row in enumerate(selected, start=1)
        ],
        "capability_quotas": dict(sorted(capability_counts.items())),
        "role_quotas": dict(sorted(role_counts.items())),
        "cluster_coverage": dict(sorted(cluster_counts.items())),
    }
    return _self_hashed(unsigned, "manifest_sha256")


__all__ = [
    "build_discovery_failure_summary",
    "build_discovery_feedback_population",
    "build_feedback_selection_manifest",
    "project_feedback_observation",
    "project_feedback_query",
    "select_feedback_samples",
]
