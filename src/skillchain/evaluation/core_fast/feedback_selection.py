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


class CounterfactualEvidenceError(ValueError):
    """The frozen parent cannot supply a complete 3/3/3 evidence packet."""


def _public_evidence_shape(observation: AssistantObservation) -> dict[str, object]:
    result = observation.replay_context.get("assistant_result")
    if not isinstance(result, dict):
        return {"visible_cards": 0, "tool_evidence": []}
    evidence = result.get("visible_tool_evidence")
    rows = evidence if isinstance(evidence, list) else []
    projected: list[dict[str, object]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        projected.append(
            {
                "tool_name": item.get("tool_name"),
                "status": item.get("status"),
                "cards": len(item.get("cards", []))
                if isinstance(item.get("cards"), list)
                else 0,
                "citations": len(item.get("citations", []))
                if isinstance(item.get("citations"), list)
                else 0,
                "detections": len(item.get("detections", []))
                if isinstance(item.get("detections"), list)
                else 0,
            }
        )
    cards = result.get("visible_cards")
    return {
        "visible_cards": len(cards) if isinstance(cards, list) else 0,
        "tool_evidence": projected,
    }


def _surface_success(
    observation: AssistantObservation,
    surface: str,
    capability: str,
) -> bool:
    if observation.selected_capability != capability:
        return False
    components = observation.gcs_components
    if surface == "action-policy":
        return bool(components["route_acceptable"] and components["tool_contract_pass"])
    if surface == "response-policy":
        return bool(
            components["route_acceptable"]
            and components["tool_contract_pass"]
            and components["no_hard_error"]
            and components["evidence_grounded"]
            and components["output_contract_pass"]
        )
    raise CounterfactualEvidenceError(f"unknown counterfactual surface: {surface}")


def _counterfactual_state(
    query: Query,
    observation: AssistantObservation,
    surface: str,
) -> dict[str, object]:
    trace = [
        {
            "tool_name": item.tool_name,
            "status": item.status,
            "error_code": item.error_code,
        }
        for item in observation.tool_trace
    ]
    common = {
        "selected_capability": observation.selected_capability,
        "source_or_boundary": query.boundary_strategy or observation.source or "none",
        "answer_mode": observation.answer_mode,
        "tool_trace": trace,
        "public_evidence_shape": _public_evidence_shape(observation),
    }
    if surface == "action-policy":
        return {
            **common,
            "failed_action_components": [
                name
                for name in ("route_acceptable", "tool_contract_pass")
                if not observation.gcs_components[name]
            ],
        }
    return {
        **common,
        "failed_response_components": [
            name
            for name in (
                "no_hard_error",
                "evidence_grounded",
                "output_contract_pass",
            )
            if not observation.gcs_components[name]
        ],
        "response_reason_codes": list(observation.gcs_reason_codes),
    }


def build_parent_counterfactual_population(
    *,
    queries: Sequence[Query],
    observations: Mapping[str, AssistantObservation],
    settings: S1Settings,
) -> tuple[dict[str, object], ...]:
    """Project all parent opt800 rows for one pre-frozen surface treatment."""

    if settings.target_surface is None or len(settings.target_capabilities) != 1:
        raise CounterfactualEvidenceError(
            "counterfactual population requires one target and one surface"
        )
    target = settings.target_capabilities[0]
    rows: list[dict[str, object]] = []
    for ordinal, query in enumerate(queries):
        if query.split != "opt_pool" or query.canonical_capability != target:
            continue
        observation = observations[query.query_id]
        if not observation.oracle_available:
            raise CounterfactualEvidenceError(
                f"parent counterfactual input lacks oracle coverage: {query.query_id}"
            )
        state = _counterfactual_state(query, observation, settings.target_surface)
        success = _surface_success(observation, settings.target_surface, target)
        rows.append(
            {
                "query_ordinal": ordinal,
                "query_id": query.query_id,
                "asset_id": query.asset_id,
                "leakage_group_id": query.leakage_group_id,
                "capability": target,
                "surface": settings.target_surface,
                "role": "parent_success" if success else "parent_failure",
                "surface_success": success,
                "state": state,
                "state_sha256": _hash(state),
                "cluster_sha256": _hash(
                    {
                        "surface": settings.target_surface,
                        "capability": target,
                        "state": state,
                    }
                ),
            }
        )
    if not rows:
        raise CounterfactualEvidenceError("counterfactual target population is empty")
    return tuple(rows)


def select_parent_counterfactual_samples(
    population: Sequence[Mapping[str, object]],
    settings: S1Settings,
) -> tuple[dict[str, object], ...]:
    """Select one cluster plus matched successes and historical regressions."""

    by_id = {str(row["query_id"]): row for row in population}
    seeds = [
        by_id[query_id]
        for query_id in settings.counterfactual_gain_seed_query_ids
        if query_id in by_id and not bool(by_id[query_id]["surface_success"])
    ]
    if not seeds:
        raise CounterfactualEvidenceError(
            "insufficient_counterfactual_evidence: no frozen gain seed still fails"
        )
    cluster_support = Counter(str(row["cluster_sha256"]) for row in seeds)
    current_failures = Counter(
        str(row["cluster_sha256"])
        for row in population
        if not bool(row["surface_success"])
    )
    selected_cluster = min(
        cluster_support,
        key=lambda value: (
            -cluster_support[value],
            -current_failures[value],
            value,
        ),
    )
    failures = sorted(
        (
            row
            for row in population
            if not bool(row["surface_success"])
            and row["cluster_sha256"] == selected_cluster
            and row["query_id"] not in settings.counterfactual_regression_query_ids
        ),
        key=lambda row: (
            0 if row["query_id"] in settings.counterfactual_gain_seed_query_ids else 1,
            int(row["query_ordinal"]),
            str(row["query_id"]),
        ),
    )
    selected: list[Mapping[str, object]] = []

    def take(rows: Iterable[Mapping[str, object]], count: int) -> None:
        used_query = {row["query_id"] for row in selected}
        used_asset = {row["asset_id"] for row in selected}
        used_leakage = {row["leakage_group_id"] for row in selected}
        for row in rows:
            if len(selected) >= count:
                return
            if (
                row["query_id"] in used_query
                or row["asset_id"] in used_asset
                or row["leakage_group_id"] in used_leakage
            ):
                continue
            selected.append(row)
            used_query.add(row["query_id"])
            used_asset.add(row["asset_id"])
            used_leakage.add(row["leakage_group_id"])

    take(failures, 3)
    if len(selected) != 3:
        raise CounterfactualEvidenceError(
            "insufficient_counterfactual_evidence: fewer than three cluster failures"
        )
    exemplar = selected[0]
    exemplar_state = exemplar["state"]
    assert isinstance(exemplar_state, dict)

    def success_rank(row: Mapping[str, object]) -> tuple[int, int, int, str]:
        state = row["state"]
        assert isinstance(state, dict)
        if settings.target_surface == "response-policy":
            exact_boundary = int(
                state.get("tool_trace") != exemplar_state.get("tool_trace")
                or state.get("public_evidence_shape")
                != exemplar_state.get("public_evidence_shape")
            )
        else:
            exact_boundary = int(
                state.get("source_or_boundary")
                != exemplar_state.get("source_or_boundary")
            )
        return (
            exact_boundary,
            int(state.get("tool_trace") != exemplar_state.get("tool_trace")),
            int(row["query_ordinal"]),
            str(row["query_id"]),
        )

    successes = sorted(
        (
            row
            for row in population
            if bool(row["surface_success"])
            and row["query_id"] not in settings.counterfactual_regression_query_ids
        ),
        key=success_rank,
    )
    if settings.target_surface == "response-policy":
        successes = [row for row in successes if success_rank(row)[0] == 0]
    take(successes, 6)
    if len(selected) != 6:
        raise CounterfactualEvidenceError(
            "insufficient_counterfactual_evidence: fewer than three matched parent successes"
        )
    regressions = [
        by_id[query_id]
        for query_id in settings.counterfactual_regression_query_ids
        if query_id in by_id
    ]
    take(regressions, 9)
    if len(selected) != 9:
        raise CounterfactualEvidenceError(
            "insufficient_counterfactual_evidence: fewer than three distinct regressions"
        )
    roles = (
        "cluster_failure",
        "cluster_failure",
        "cluster_failure",
        "parent_success",
        "parent_success",
        "parent_success",
        "historical_regression",
        "historical_regression",
        "historical_regression",
    )
    rows = [
        {**dict(row), "counterfactual_role": role} for row, role in zip(selected, roles)
    ]
    # Canary3 observes one sample from every evidence class.
    order = (0, 3, 6, 1, 2, 4, 5, 7, 8)
    return tuple(rows[index] for index in order)


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
    elif settings.feedback_selection_policy in {
        "discovery-contrastive-v2",
        "discovery-supported-clusters-v3",
        "discovery-attributed-v4",
        "discovery-dual-policy-v5",
    }:
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
        cluster_support = Counter(
            str(row["cluster"]["cluster_sha256"])  # type: ignore[index]
            for row in population
            if row["role"] == "failure"
        )
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
            if settings.feedback_selection_policy in {
                "discovery-supported-clusters-v3",
                "discovery-attributed-v4",
                "discovery-dual-policy-v5",
            }:

                def rank(row: Mapping[str, object]) -> tuple[int, int, str]:
                    return (
                        -cluster_support[
                            str(row["cluster"]["cluster_sha256"])  # type: ignore[index]
                        ],
                        int(row["query_ordinal"]),
                        str(row["query_id"]),
                    )

                body = sorted(
                    (
                        row
                        for row in populations[capability]
                        if row["selection_class"] == "body_fixable_failure"
                    ),
                    key=rank,
                )
                boundary = sorted(
                    (
                        row
                        for row in populations[capability]
                        if row["selection_class"] == "boundary_failure"
                    ),
                    key=rank,
                )
            else:
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
        selected = []
        # Interleave capabilities so the operational canary covers all six
        # branches instead of accidentally probing only the first capability.
        for ordinal in range(max(len(rows) for rows in by_capability.values())):
            for rows in by_capability.values():
                selected.extend(rows[ordinal : ordinal + 1])
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


def build_parent_counterfactual_manifest(
    *,
    population: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
    settings: S1Settings,
    parent_bank_sha256: str,
    parent_opt_sha256: str,
) -> dict[str, object]:
    if len(selected) != 9:
        raise CounterfactualEvidenceError("counterfactual manifest requires nine rows")
    role_counts = Counter(str(row["counterfactual_role"]) for row in selected)
    if role_counts != Counter(
        {"cluster_failure": 3, "parent_success": 3, "historical_regression": 3}
    ):
        raise CounterfactualEvidenceError(
            "counterfactual manifest role geometry differs"
        )
    unsigned = {
        "schema_version": 1,
        "kind": "core-fast-parent-counterfactual-selection-manifest",
        "cycle_id": settings.cycle_id,
        "round_id": settings.round_id,
        "parent_bank_sha256": parent_bank_sha256,
        "parent_opt_sha256": parent_opt_sha256,
        "target_capability": settings.target_capabilities[0],
        "target_surface": settings.target_surface,
        "population_count": len(population),
        "population_sha256": _hash(list(population)),
        "selection_policy": settings.feedback_selection_policy,
        "requested_count": 9,
        "effective_count": 9,
        "canary_count": 3,
        "selected_query_ids": [row["query_id"] for row in selected],
        "selected_samples": [
            {
                "selection_ordinal": index,
                "query_id": row["query_id"],
                "capability": row["capability"],
                "role": row["counterfactual_role"],
                "selection_class": row["counterfactual_role"],
                "cluster_sha256": row["cluster_sha256"],
                "failure_cluster": row["state"],
                "state_sha256": row["state_sha256"],
                "selection_reason": "frozen_counterfactual_3_failure_3_success_3_regression",
            }
            for index, row in enumerate(selected, start=1)
        ],
        "capability_quotas": {settings.target_capabilities[0]: 9},
        "role_quotas": dict(sorted(role_counts.items())),
        "cluster_coverage": dict(
            sorted(Counter(str(row["cluster_sha256"]) for row in selected).items())
        ),
    }
    return _self_hashed(unsigned, "manifest_sha256")


__all__ = [
    "CounterfactualEvidenceError",
    "build_discovery_failure_summary",
    "build_discovery_feedback_population",
    "build_feedback_selection_manifest",
    "build_parent_counterfactual_manifest",
    "build_parent_counterfactual_population",
    "project_feedback_observation",
    "project_feedback_query",
    "select_feedback_samples",
    "select_parent_counterfactual_samples",
]
