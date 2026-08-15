from __future__ import annotations

import json
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from skillchain.evaluation.core_fast import (
    feedback_selection as feedback_selection_module,
)
from skillchain.evaluation.core_fast.engine import CoreFastEngine, FastPathError
from skillchain.evaluation.core_fast.fake_provider import FakeCoreFastAdapter
from skillchain.evaluation.core_fast.feedback_selection import (
    CounterfactualEvidenceError,
    build_discovery_feedback_population,
    build_parent_counterfactual_manifest,
    build_parent_counterfactual_population,
    project_feedback_observation_for_surface,
    select_parent_counterfactual_samples,
    select_feedback_samples,
)
from skillchain.evaluation.core_fast.models import (
    CAPABILITIES,
    S1ParentBinding,
    S1Settings,
    ToolTraceItem,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

_fixture_spec = importlib.util.spec_from_file_location(
    "core_fast_fixture_module", Path(__file__).with_name("test_core_fast.py")
)
assert _fixture_spec is not None and _fixture_spec.loader is not None
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
fast_fixture = _fixture_module.fast_fixture


def _settings(
    *,
    count: int,
    allocation: str = "balanced-six-capability",
    policy: str = "discovery-stratified-v1",
) -> S1Settings:
    payload: dict[str, object] = {
        "feedback_mode": "fresh-per-round",
        "feedback_total_count": count,
        "feedback_canary_count": min(6, count),
        "feedback_selection_policy": policy,
        "feedback_allocation": allocation,
    }
    if allocation == "target-focused":
        payload.update(
            {
                "target_capabilities": ("utility.recipe_guidance",),
                "max_patched_capabilities": 1,
                "protected_capabilities": tuple(
                    item for item in CAPABILITIES if item != "utility.recipe_guidance"
                ),
            }
        )
    return S1Settings.model_validate(payload, strict=True)


def _population(fast_fixture):
    spec, spec_path, queries = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=spec_path.parent / "population",
        adapter=FakeCoreFastAdapter(),
    )
    return build_discovery_feedback_population(
        queries=queries,
        observations=engine.opt_static(),
        fold_roles=engine.opt_fold_roles(),
    )


def test_discovery_population_covers_600_and_excludes_replay(fast_fixture) -> None:
    population = _population(fast_fixture)
    query_ids = {row["query_id"] for row in population}
    replay_ids = {
        query_id
        for query_id, role in CoreFastEngine(
            spec=fast_fixture[0],
            spec_path=fast_fixture[1],
            output_root=fast_fixture[1].parent / "roles",
            adapter=FakeCoreFastAdapter(),
        )
        .opt_fold_roles()
        .items()
        if role == "replay"
    }
    assert len(population) == len(query_ids) == 600
    assert query_ids.isdisjoint(replay_ids)


def test_parent_counterfactual_selection_freezes_one_cluster_and_three_three_three(
    fast_fixture,
) -> None:
    spec, spec_path, queries = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=spec_path.parent / "counterfactual-population",
        adapter=FakeCoreFastAdapter(),
    )
    target_rows = [
        query
        for query in queries
        if query.split == "opt_pool"
        and query.canonical_capability == "utility.recipe_guidance"
    ]
    settings = S1Settings.model_validate(
        {
            "round_id": "r31",
            "feedback_mode": "fresh-per-round",
            "feedback_total_count": 9,
            "feedback_canary_count": 3,
            "feedback_format_retry_limit": 1,
            "feedback_selection_policy": "parent-counterfactual-v6",
            "feedback_allocation": "target-focused",
            "target_capabilities": ("utility.recipe_guidance",),
            "proposal_mode": "single-surface-counterfactual-fanout-v4",
            "max_patched_capabilities": 1,
            "protected_capabilities": tuple(
                item for item in CAPABILITIES if item != "utility.recipe_guidance"
            ),
            "cycle_id": "s1-counterfactual-v1",
            "target_surface": "action-policy",
            "counterfactual_gain_seed_query_ids": tuple(
                sorted((target_rows[0].query_id, target_rows[2].query_id))
            ),
            "counterfactual_regression_query_ids": tuple(
                sorted(
                    (
                        target_rows[3].query_id,
                        target_rows[4].query_id,
                        target_rows[5].query_id,
                    )
                )
            ),
            "parent_protection_query_ids": ("style-protection",),
        },
        strict=True,
    )
    chosen = target_rows[::2][:9]
    observations = dict(engine.opt_static())
    for index, query in enumerate(chosen):
        original = observations[query.query_id]
        components = dict(original.gcs_components)
        components.update(
            {
                "route_acceptable": True,
                "tool_contract_pass": index >= 3,
            }
        )
        observations[query.query_id] = original.model_copy(
            update={
                "selected_capability": "utility.recipe_guidance",
                "gcs_components": components,
            }
        )
    settings = settings.model_copy(
        update={
            "counterfactual_gain_seed_query_ids": tuple(
                sorted(row.query_id for row in chosen[:2])
            ),
            "counterfactual_regression_query_ids": tuple(
                sorted(row.query_id for row in chosen[6:9])
            ),
        }
    )
    population = build_parent_counterfactual_population(
        queries=queries,
        observations=observations,
        settings=settings,
    )
    selected = select_parent_counterfactual_samples(population, settings)
    assert len(selected) == 9
    assert Counter(row["counterfactual_role"] for row in selected) == Counter(
        {"cluster_failure": 3, "parent_success": 3, "historical_regression": 3}
    )
    failures = [
        row for row in selected if row["counterfactual_role"] == "cluster_failure"
    ]
    assert len({row["cluster_sha256"] for row in failures}) == 1
    assert tuple(row["counterfactual_role"] for row in selected[:3]) == (
        "cluster_failure",
        "parent_success",
        "historical_regression",
    )
    assert len({row["leakage_group_id"] for row in selected}) == 9
    manifest = build_parent_counterfactual_manifest(
        population=population,
        selected=selected,
        settings=settings,
        parent_bank_sha256="a" * 64,
        parent_opt_sha256="b" * 64,
    )
    assert manifest["role_quotas"] == {
        "cluster_failure": 3,
        "historical_regression": 3,
        "parent_success": 3,
    }
    with pytest.raises(CounterfactualEvidenceError, match="parent successes"):
        select_parent_counterfactual_samples(
            tuple(
                row
                for row in population
                if row["role"] != "parent_success"
                or row["query_id"] in {target_rows[1].query_id, target_rows[6].query_id}
            ),
            settings,
        )


def test_response_selector_qualifies_fixed_tool_outcomes_and_abstracts_cardinality(
    fast_fixture,
) -> None:
    spec, spec_path, queries = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=spec_path.parent / "response-counterfactual-population",
        adapter=FakeCoreFastAdapter(),
    )
    target_rows = [
        query
        for query in queries
        if query.split == "opt_pool"
        and query.canonical_capability == "product.multi_search"
    ]
    chosen = target_rows[::2][:9]
    observations = dict(engine.opt_static())
    successful_trace = (
        ToolTraceItem(
            tool_name="multi_product_search",
            arguments={},
            status="success",
            result_sha256="a" * 64,
        ),
    )
    for index, query in enumerate(chosen):
        original = observations[query.query_id]
        components = dict(original.gcs_components)
        components.update(
            {
                "route_acceptable": True,
                "tool_contract_pass": True,
                "no_hard_error": True,
                "evidence_grounded": index >= 3,
                "output_contract_pass": True,
            }
        )
        context = json.loads(json.dumps(original.replay_context))
        result = context["assistant_result"]
        card_count = 1 if index % 2 == 0 else 3
        result["visible_cards"] = [{} for _ in range(card_count)]
        result["visible_tool_evidence"] = [
            {
                "tool_name": "multi_product_search",
                "status": "success",
                "cards": [{} for _ in range(card_count)],
                "citations": [],
                "detections": [],
            }
        ]
        observations[query.query_id] = original.model_copy(
            update={
                "selected_capability": "product.multi_search",
                "tool_trace": successful_trace,
                "hard_error": False,
                "gcs_components": components,
                "replay_context": context,
            }
        )
    settings = S1Settings.model_validate(
        {
            "round_id": "r32",
            "feedback_total_count": 9,
            "feedback_canary_count": 3,
            "feedback_format_retry_limit": 1,
            "feedback_selection_policy": "parent-counterfactual-v6",
            "feedback_allocation": "target-focused",
            "target_capabilities": ("product.multi_search",),
            "proposal_mode": "single-surface-counterfactual-fanout-v4",
            "max_patched_capabilities": 1,
            "protected_capabilities": tuple(
                item for item in CAPABILITIES if item != "product.multi_search"
            ),
            "cycle_id": "s1-counterfactual-v1",
            "target_surface": "response-policy",
            "counterfactual_gain_seed_query_ids": tuple(
                sorted(row.query_id for row in chosen[:2])
            ),
            "counterfactual_regression_query_ids": tuple(
                sorted(row.query_id for row in chosen[6:9])
            ),
            "parent_protection_query_ids": ("style-protection",),
        },
        strict=True,
    )
    population = build_parent_counterfactual_population(
        queries=queries,
        observations=observations,
        settings=settings,
    )
    selected = select_parent_counterfactual_samples(population, settings)
    failures = [
        row for row in selected if row["counterfactual_role"] == "cluster_failure"
    ]
    successes = [
        row for row in selected if row["counterfactual_role"] == "parent_success"
    ]
    assert len({row["cluster_sha256"] for row in failures}) == 1
    assert len(successes) == 3
    assert {
        row["state"]["public_evidence_shape"]["visible_cards"]  # type: ignore[index]
        for row in (*failures, *successes)
    } == {1, 3}


def test_v8_action_selector_requires_a_provider_visible_failure_state(
    fast_fixture,
) -> None:
    spec, spec_path, queries = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=spec_path.parent / "v8-action-population",
        adapter=FakeCoreFastAdapter(),
    )
    target = "utility.recipe_guidance"
    target_rows = [
        query
        for query in queries
        if query.split == "opt_pool" and query.canonical_capability == target
    ]
    chosen = target_rows[::2][:10]
    observations = dict(engine.opt_static())
    for index, query in enumerate(chosen):
        original = observations[query.query_id]
        components = dict(original.gcs_components)
        failed = index < 3
        components.update(
            {
                "route_acceptable": True,
                "tool_contract_pass": not failed,
            }
        )
        observations[query.query_id] = original.model_copy(
            update={
                "selected_capability": target,
                "tool_trace": (
                    ToolTraceItem(
                        tool_name="recipe_lookup",
                        status="error",
                        error_code="invalid_arguments",
                    ),
                )
                if failed
                else (
                    ToolTraceItem(
                        tool_name="recipe_lookup",
                        status="success",
                        result_sha256="a" * 64,
                    ),
                ),
                "gcs_components": components,
            }
        )
    settings = S1Settings.model_validate(
        {
            "round_id": "r44",
            "feedback_total_count": 9,
            "feedback_canary_count": 3,
            "feedback_format_retry_limit": 1,
            "feedback_selection_policy": "parent-counterfactual-v8",
            "feedback_allocation": "target-focused",
            "target_capabilities": (target,),
            "proposal_mode": "single-surface-counterfactual-fanout-v5",
            "max_patched_capabilities": 1,
            "protected_capabilities": tuple(
                item for item in CAPABILITIES if item != target
            ),
            "cycle_id": "s1-r12-adaptive-v1-b04",
            "target_surface": "action-policy",
            "counterfactual_gain_seed_query_ids": tuple(
                sorted(row.query_id for row in chosen[:2])
            ),
            "counterfactual_regression_query_ids": tuple(
                sorted(row.query_id for row in chosen[7:10])
            ),
            "counterfactual_parent_success_exclude_query_ids": (chosen[3].query_id,),
            "parent_protection_query_ids": ("style-protection",),
            "cycle_preflight_path": "batch-preflight.json",
            "cycle_preflight_sha256": "b" * 64,
        },
        strict=True,
    )
    population = build_parent_counterfactual_population(
        queries=queries,
        observations=observations,
        settings=settings,
    )
    selected = select_parent_counterfactual_samples(population, settings)
    failures = [
        row for row in selected if row["counterfactual_role"] == "cluster_failure"
    ]
    successes = [
        row for row in selected if row["counterfactual_role"] == "parent_success"
    ]
    assert {row["action_treatment_separable"] for row in failures} == {True}
    assert {
        canonical_json_bytes(row["action_treatment_signature"]) for row in failures
    } == {
        canonical_json_bytes(
            {
                "phase": "after-tool",
                "prior_tool_name": "recipe_lookup",
                "prior_tool_status": "invalid-arguments",
                "public_evidence": "unknown",
            }
        )
    }
    assert chosen[3].query_id not in {row["query_id"] for row in successes}
    manifest = build_parent_counterfactual_manifest(
        population=population,
        selected=selected,
        settings=settings,
        parent_bank_sha256="c" * 64,
        parent_opt_sha256="d" * 64,
    )
    assert manifest["parent_success_exclude_query_ids"] == [chosen[3].query_id]
    assert (
        manifest["selected_samples"][0]["action_treatment_signature"]
        == (failures[0]["action_treatment_signature"])
    )

    no_tool = observations[chosen[0].query_id].model_copy(update={"tool_trace": ()})
    no_tool_population = build_parent_counterfactual_population(
        queries=queries,
        observations={**observations, chosen[0].query_id: no_tool},
        settings=settings,
    )
    assert (
        next(
            row for row in no_tool_population if row["query_id"] == chosen[0].query_id
        )["action_treatment_separable"]
        is False
    )


def test_v9_repeated_successful_terminal_tool_binds_stop_directive() -> None:
    signature = feedback_selection_module._action_failure_signature(
        {
            "tool_trace": [
                {
                    "tool_name": "recipe_lookup",
                    "status": "success",
                    "error_code": None,
                },
                {
                    "tool_name": "recipe_lookup",
                    "status": "success",
                    "error_code": None,
                },
            ]
        }
    )

    assert signature == {
        "phase": "after-tool",
        "prior_tool_name": "recipe_lookup",
        "prior_tool_status": "success",
        "public_evidence": "unknown",
    }
    assert feedback_selection_module._action_treatment_directive(signature) == {
        "operation": "stop-action-loop",
        "tool_name": None,
        "arguments_from": "none",
    }


def test_v8_response_selector_uses_one_failure_family_and_stable_controls(
    fast_fixture,
) -> None:
    spec, spec_path, queries = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=spec_path.parent / "v8-response-population",
        adapter=FakeCoreFastAdapter(),
    )
    target = "product.multi_search"
    target_rows = [
        query
        for query in queries
        if query.split == "opt_pool" and query.canonical_capability == target
    ]
    chosen = target_rows[::2][:13]
    observations = dict(engine.opt_static())
    trace = (
        ToolTraceItem(
            tool_name="multi_product_search",
            status="success",
            result_sha256="a" * 64,
        ),
    )
    for index, query in enumerate(chosen):
        original = observations[query.query_id]
        failed = index < 6
        components = dict(original.gcs_components)
        components.update(
            {
                "route_acceptable": True,
                "tool_contract_pass": True,
                "no_hard_error": True,
                "evidence_grounded": not failed,
                "output_contract_pass": True,
            }
        )
        context = json.loads(json.dumps(original.replay_context))
        context["assistant_result"]["visible_cards"] = [{}]
        context["assistant_result"]["visible_tool_evidence"] = [
            {
                "tool_name": "multi_product_search",
                "status": "success",
                "cards": [{}],
                "citations": [],
                "detections": [],
            }
        ]
        observations[query.query_id] = original.model_copy(
            update={
                "selected_capability": target,
                "tool_trace": trace,
                "hard_error": False,
                "gcs_components": components,
                "gcs_reason_codes": (
                    ("multi_mapping_invalid",)
                    if index < 3
                    else ("card_contract_failed",)
                    if failed
                    else ()
                ),
                "replay_context": context,
            }
        )
    settings = S1Settings.model_validate(
        {
            "round_id": "r45",
            "feedback_total_count": 9,
            "feedback_canary_count": 3,
            "feedback_format_retry_limit": 1,
            "feedback_selection_policy": "parent-counterfactual-v8",
            "feedback_allocation": "target-focused",
            "target_capabilities": (target,),
            "proposal_mode": "single-surface-counterfactual-fanout-v5",
            "max_patched_capabilities": 1,
            "protected_capabilities": tuple(
                item for item in CAPABILITIES if item != target
            ),
            "cycle_id": "s1-r12-adaptive-v1-b04",
            "target_surface": "response-policy",
            "counterfactual_gain_seed_query_ids": tuple(
                sorted(row.query_id for row in chosen[:2])
            ),
            "counterfactual_regression_query_ids": tuple(
                sorted(row.query_id for row in chosen[10:13])
            ),
            "counterfactual_parent_success_exclude_query_ids": (chosen[6].query_id,),
            "parent_protection_query_ids": ("style-protection",),
            "cycle_preflight_path": "batch-preflight.json",
            "cycle_preflight_sha256": "b" * 64,
        },
        strict=True,
    )
    population = build_parent_counterfactual_population(
        queries=queries,
        observations=observations,
        settings=settings,
    )
    selected = select_parent_counterfactual_samples(population, settings)
    failures = [
        row for row in selected if row["counterfactual_role"] == "cluster_failure"
    ]
    successes = [
        row for row in selected if row["counterfactual_role"] == "parent_success"
    ]
    assert {
        row["response_treatment_signature"]["response_failure_family"]
        for row in failures
    } == {"item-association"}
    assert chosen[6].query_id not in {row["query_id"] for row in successes}
    assert {row["response_treatment_signature"] for row in successes} == {None}


def test_v9_response_selector_binds_terminal_evidence_and_scored_behavior(
    fast_fixture,
) -> None:
    spec, spec_path, queries = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=spec_path.parent / "v9-response-population",
        adapter=FakeCoreFastAdapter(),
    )
    target = "utility.recipe_guidance"
    target_rows = [
        query
        for query in queries
        if query.split == "opt_pool" and query.canonical_capability == target
    ]
    chosen = target_rows[::2][:13]
    observations = dict(engine.opt_static())
    trace = (
        ToolTraceItem(
            tool_name="object_detect", status="success", result_sha256="a" * 64
        ),
        ToolTraceItem(
            tool_name="recipe_lookup", status="success", result_sha256="b" * 64
        ),
    )
    for index, query in enumerate(chosen):
        original = observations[query.query_id]
        failed = index < 6
        components = dict(original.gcs_components)
        components.update(
            {
                "route_acceptable": True,
                "tool_contract_pass": True,
                "no_hard_error": True,
                "evidence_grounded": not failed,
                "output_contract_pass": True,
            }
        )
        context = json.loads(json.dumps(original.replay_context))
        context["assistant_result"]["visible_tool_evidence"] = [
            {
                "tool_name": "object_detect",
                "status": "success",
                "cards": [],
                "citations": [],
                "detections": [{}],
                "visible_text": "one visible detection",
            },
            {
                "tool_name": "recipe_lookup",
                "status": "success",
                "cards": [],
                "citations": [],
                "detections": [],
                "visible_text": "[tool-call-2-source-1] public source text",
            },
        ]
        observations[query.query_id] = original.model_copy(
            update={
                "selected_capability": target,
                "tool_trace": trace,
                "hard_error": False,
                "gcs_components": components,
                "gcs_reason_codes": ("unsupported_claim",) if failed else (),
                "replay_context": context,
            }
        )
    settings = S1Settings.model_validate(
        {
            "round_id": "r49",
            "feedback_total_count": 9,
            "feedback_canary_count": 3,
            "feedback_format_retry_limit": 1,
            "feedback_selection_policy": "parent-counterfactual-v9",
            "feedback_allocation": "target-focused",
            "target_capabilities": (target,),
            "proposal_mode": "single-surface-counterfactual-fanout-v6",
            "max_patched_capabilities": 1,
            "protected_capabilities": tuple(
                item for item in CAPABILITIES if item != target
            ),
            "cycle_id": "s1-r12-adaptive-v1-b06",
            "target_surface": "response-policy",
            "counterfactual_gain_seed_query_ids": tuple(
                sorted(row.query_id for row in chosen[:2])
            ),
            "counterfactual_regression_query_ids": tuple(
                sorted(row.query_id for row in chosen[10:13])
            ),
            "parent_protection_query_ids": ("style-protection",),
            "cycle_preflight_path": "batch-preflight.json",
            "cycle_preflight_sha256": "b" * 64,
        },
        strict=True,
    )
    population = build_parent_counterfactual_population(
        queries=queries, observations=observations, settings=settings
    )
    selected = select_parent_counterfactual_samples(population, settings)
    failures = [
        row for row in selected if row["counterfactual_role"] == "cluster_failure"
    ]
    successes = [
        row for row in selected if row["counterfactual_role"] == "parent_success"
    ]
    signature = failures[0]["response_treatment_signature"]
    assert signature == {
        "terminal_evidence_class": {
            "tool_names": ["recipe_lookup"],
            "outcome": "nonempty",
            "evidence_kinds": ["source-text"],
        },
        "response_failure_family": "unsupported-claim",
        "predicted_reason_code": "unsupported_claim",
        "predicted_metric_component": "evidence_grounded",
    }
    assert {
        canonical_json_bytes(row["response_treatment_signature"]) for row in failures
    } == {canonical_json_bytes(signature)}
    assert {
        canonical_json_bytes(row["state"]["terminal_response_evidence_class"])
        for row in successes
    } == {canonical_json_bytes(signature["terminal_evidence_class"])}


def test_counterfactual_feedback_projection_is_surface_closed(fast_fixture) -> None:
    spec, spec_path, queries = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=spec_path.parent / "surface-projection",
        adapter=FakeCoreFastAdapter(),
    )
    query = next(item for item in queries if item.split == "opt_pool")
    observation = engine.opt_static()[query.query_id]
    action = project_feedback_observation_for_surface(observation, "action-policy")
    response = project_feedback_observation_for_surface(observation, "response-policy")
    assert "response_text" not in action
    assert "card_violation" not in action and "evidence_violation" not in action
    assert "response_text" in response
    assert "tool_violation" not in response
    assert set(action["gcs_components"]) == {
        "route_acceptable",
        "tool_contract_pass",
    }


def test_counterfactual_feedback_bundle_keeps_only_the_target_surface(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, queries = fast_fixture
    target = "utility.recipe_guidance"
    query = next(
        item
        for item in queries
        if item.split == "opt_pool" and item.canonical_capability == target
    )
    settings = S1Settings.model_validate(
        {
            "round_id": "r31",
            "feedback_total_count": 9,
            "feedback_canary_count": 3,
            "feedback_format_retry_limit": 1,
            "feedback_selection_policy": "parent-counterfactual-v6",
            "feedback_allocation": "target-focused",
            "target_capabilities": (target,),
            "proposal_mode": "single-surface-counterfactual-fanout-v4",
            "max_patched_capabilities": 1,
            "protected_capabilities": tuple(
                item for item in CAPABILITIES if item != target
            ),
            "cycle_id": "s1-counterfactual-v1",
            "target_surface": "action-policy",
            "counterfactual_gain_seed_query_ids": ("g1", "g2"),
            "counterfactual_regression_query_ids": ("r1", "r2", "r3"),
            "parent_protection_query_ids": ("style-protection",),
        },
        strict=True,
    )
    engine = CoreFastEngine(
        spec=spec.model_copy(update={"s1_settings": settings}),
        spec_path=spec_path,
        output_root=tmp_path / "surface-filter",
        adapter=FakeCoreFastAdapter(),
    )
    bundle = engine._feedback_evidence_bundle(  # noqa: SLF001
        parent=engine.static_bank(),
        prepared={
            "discovery_summary": {},
            "selection_manifest": {
                "selected_query_ids": [query.query_id],
                "manifest_sha256": "a" * 64,
                "population_sha256": "b" * 64,
            },
        },
        feedback={
            query.query_id: {
                "query_id": query.query_id,
                "capability": target,
                "sample_role": "cluster_failure",
                "selection_class": "cluster_failure",
                "feedback": {
                    "skill_suggestions": [
                        "[policy_compatible] [action-policy] retry one visible tool call",
                        "[policy_compatible] [response-policy] rewrite the answer",
                        "[policy_compatible] suggestion without a surface",
                    ]
                },
            }
        },
    )
    suggestions = bundle["policy_compatible_suggestions"]
    assert isinstance(suggestions, list) and len(suggestions) == 1
    assert suggestions[0]["surface"] == "action-policy"
    assert bundle["rejected_suggestion_counts"] == {
        "cross_surface_suggestion": 1,
        "missing_or_ambiguous_policy_surface": 1,
    }


def test_typed_cycle_preflight_requires_every_round_before_feedback(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _queries = fast_fixture
    preflight_path = tmp_path / "cycle-preflight.json"
    parent_binding = S1ParentBinding(
        source_round_id="r12",
        bank_path="parent.json",
        bank_file_sha256="a" * 64,
        bank_sha256="b" * 64,
        decision_path="decision.json",
        decision_file_sha256="c" * 64,
        manifest_path="manifest.json",
        manifest_file_sha256="d" * 64,
        opt_results_path="parent-opt.jsonl",
        opt_results_sha256="e" * 64,
        parent_protection_query_ids=("style-protection",),
        protected_skill_sha256={"product.style_recommendation": "f" * 64},
    )
    preflight = {
        "schema_version": 1,
        "kind": "core-fast-s1-counterfactual-cycle-preflight",
        "cycle_id": "s1-counterfactual-v2",
        "parent_bank_sha256": parent_binding.bank_sha256,
        "parent_opt_sha256": parent_binding.opt_results_sha256,
        "rounds": {
            "r31": {
                "target_capability": "utility.recipe_guidance",
                "target_surface": "action-policy",
                "evidence_feasible": True,
                "treatment_sensitive": True,
                "selection_manifest_sha256": "1" * 64,
            },
            "r32": {
                "target_capability": "product.multi_search",
                "target_surface": "response-policy",
                "evidence_feasible": True,
                "treatment_sensitive": True,
                "selection_manifest_sha256": "2" * 64,
            },
        },
        "passed": True,
    }
    preflight_path.write_bytes(canonical_json_bytes(preflight))
    settings = S1Settings.model_validate(
        {
            "round_id": "r31",
            "feedback_total_count": 9,
            "feedback_canary_count": 3,
            "feedback_format_retry_limit": 1,
            "feedback_selection_policy": "parent-counterfactual-v7",
            "feedback_allocation": "target-focused",
            "target_capabilities": ("utility.recipe_guidance",),
            "proposal_mode": "single-surface-counterfactual-fanout-v5",
            "max_patched_capabilities": 1,
            "protected_capabilities": tuple(
                item for item in CAPABILITIES if item != "utility.recipe_guidance"
            ),
            "cycle_id": "s1-counterfactual-v2",
            "target_surface": "action-policy",
            "counterfactual_gain_seed_query_ids": ("g1", "g2"),
            "counterfactual_regression_query_ids": ("r1", "r2", "r3"),
            "parent_protection_query_ids": ("style-protection",),
            "cycle_preflight_path": str(preflight_path),
            "cycle_preflight_sha256": sha256_bytes(preflight_path.read_bytes()),
        },
        strict=True,
    )
    engine = CoreFastEngine(
        spec=spec.model_copy(
            update={"s1_settings": settings, "s1_parent": parent_binding}
        ),
        spec_path=spec_path,
        output_root=tmp_path / "preflight-pass",
        adapter=FakeCoreFastAdapter(),
    )
    assert (
        engine._require_counterfactual_cycle_preflight(  # noqa: SLF001
            {"selection_manifest": {"manifest_sha256": "1" * 64}}
        )
        == preflight
    )

    preflight["rounds"]["r32"]["treatment_sensitive"] = False
    preflight["passed"] = False
    failed_path = tmp_path / "cycle-preflight-failed.json"
    failed_path.write_bytes(canonical_json_bytes(preflight))
    failed_settings = settings.model_copy(
        update={
            "cycle_preflight_path": str(failed_path),
            "cycle_preflight_sha256": sha256_bytes(failed_path.read_bytes()),
        }
    )
    failed_engine = CoreFastEngine(
        spec=spec.model_copy(
            update={"s1_settings": failed_settings, "s1_parent": parent_binding}
        ),
        spec_path=spec_path,
        output_root=tmp_path / "preflight-fail",
        adapter=FakeCoreFastAdapter(),
    )
    with pytest.raises(FastPathError, match="all-round preflight"):
        failed_engine._require_counterfactual_cycle_preflight()  # noqa: SLF001


@pytest.mark.parametrize("count", (12, 24, 48, 60))
def test_selection_counts_are_deterministic_and_unique(
    fast_fixture, count: int
) -> None:
    population = _population(fast_fixture)
    first = select_feedback_samples(population, _settings(count=count))
    second = select_feedback_samples(population, _settings(count=count))
    assert canonical_json_bytes(list(first)) == canonical_json_bytes(list(second))
    assert len(first) == count
    assert len({row["query_id"] for row in first}) == count
    assert len({row["asset_id"] for row in first}) == count
    assert len({row["leakage_group_id"] for row in first}) == count


def test_balanced_and_target_focused_allocations(fast_fixture) -> None:
    population = _population(fast_fixture)
    balanced = select_feedback_samples(population, _settings(count=48))
    assert {
        capability: sum(row["capability"] == capability for row in balanced)
        for capability in CAPABILITIES
    } == {capability: 8 for capability in CAPABILITIES}
    focused = select_feedback_samples(
        population, _settings(count=48, allocation="target-focused")
    )
    assert all(row["capability"] == "utility.recipe_guidance" for row in focused)
    classes = [row["selection_class"] for row in focused]
    assert classes.count("body_fixable_failure") == sum(
        row["capability"] == "utility.recipe_guidance"
        and row["selection_class"] == "body_fixable_failure"
        for row in population
    )
    assert classes.count("success_anchor") >= 1


def test_contrastive_selection_balances_failures_and_success_anchors(
    fast_fixture,
) -> None:
    population = _population(fast_fixture)
    selected = select_feedback_samples(
        population,
        _settings(count=60, policy="discovery-contrastive-v2"),
    )
    for capability in CAPABILITIES:
        rows = [row for row in selected if row["capability"] == capability]
        assert len(rows) == 10
        available_failures = sum(
            row["capability"] == capability and row["role"] == "failure"
            for row in population
        )
        available_anchors = sum(
            row["capability"] == capability and row["role"] == "anchor"
            for row in population
        )
        observed_anchors = sum(row["role"] == "anchor" for row in rows)
        observed_failures = sum(row["role"] == "failure" for row in rows)
        if available_anchors >= 5:
            assert observed_anchors >= 5
        else:
            assert observed_anchors <= available_anchors
        assert observed_failures + observed_anchors == 10
        assert observed_failures <= available_failures


def test_supported_cluster_selection_interleaves_canary_and_prioritizes_support(
    fast_fixture,
) -> None:
    population = _population(fast_fixture)
    selected = select_feedback_samples(
        population,
        _settings(count=60, policy="discovery-supported-clusters-v3"),
    )
    assert tuple(row["capability"] for row in selected[:6]) == CAPABILITIES
    for capability in CAPABILITIES:
        selected_failures = [
            row
            for row in selected
            if row["capability"] == capability and row["role"] == "failure"
        ]
        all_failures = [
            row
            for row in population
            if row["capability"] == capability and row["role"] == "failure"
        ]
        support = Counter(row["cluster"]["cluster_sha256"] for row in all_failures)
        observed = [
            support[row["cluster"]["cluster_sha256"]] for row in selected_failures
        ]
        assert observed == sorted(observed, reverse=True)


def test_attributed_selection_uses_the_same_supported_interleaved_population(
    fast_fixture,
) -> None:
    population = _population(fast_fixture)
    supported = select_feedback_samples(
        population,
        _settings(count=60, policy="discovery-supported-clusters-v3"),
    )
    attributed = select_feedback_samples(
        population,
        _settings(count=60, policy="discovery-attributed-v4"),
    )
    assert canonical_json_bytes(list(attributed)) == canonical_json_bytes(
        list(supported)
    )


def test_prepare_is_byte_exact_and_manifest_drift_fails_closed(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=tmp_path / "prepare",
        adapter=FakeCoreFastAdapter(),
    )
    first = engine.prepare_feedback_selection()
    manifest = engine.output_root / "inputs" / "feedback-selection-manifest.json"
    before = manifest.read_bytes()
    second = engine.prepare_feedback_selection()
    assert first == second
    assert manifest.read_bytes() == before
    value = json.loads(before)
    value["effective_count"] -= 1
    manifest.write_bytes(canonical_json_bytes(value))
    with pytest.raises(FastPathError, match="changed on resume"):
        engine.prepare_feedback_selection()


def test_same_round_resume_uses_exact_feedback_results(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    adapter = FakeCoreFastAdapter(reject_stages=frozenset({"s1"}))
    root = tmp_path / "resume"
    first = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=root,
        adapter=adapter,
    )
    first.run_s1()
    assert adapter.calls["feedback"] == spec.s1_settings.feedback_total_count
    decision = root / "decisions" / "s1.json"
    decision.unlink()
    second_adapter = FakeCoreFastAdapter(reject_stages=frozenset({"s1"}))
    second = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=root,
        adapter=second_adapter,
    )
    resumed = second.run_s1()
    assert second_adapter.calls["feedback"] == 0
    assert resumed.metrics["feedback_resume_cache_hits"] == (
        spec.s1_settings.feedback_total_count
    )
    assert resumed.metrics["feedback_provider_calls_this_round"] == 0


def test_creator_payload_contains_summary_but_no_hidden_population(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=tmp_path / "privacy",
        adapter=FakeCoreFastAdapter(reject_stages=frozenset({"s1"})),
    )
    engine.run_s1()
    intent = json.loads(
        (
            engine.output_root / "calls" / "creator" / "s1-creator-once.intent.json"
        ).read_text(encoding="utf-8")
    )
    encoded = canonical_json_bytes(intent["payload"])
    assert intent["payload"]["discovery_summary"]["population_count"] == 600
    assert intent["payload"]["selection_manifest_sha256"]
    for forbidden in (
        b'"split"',
        b"replay_context",
        b"runtime_paths",
        b"scorer_calls",
        b"test_frozen",
        b"body_gate",
    ):
        assert forbidden not in encoded


def test_feedback_count_changes_spec_identity(fast_fixture) -> None:
    spec = fast_fixture[0]
    count12 = spec.model_copy(
        update={
            "s1_settings": _settings(count=12),
            "limits": spec.limits.model_copy(update={"max_feedback_calls": 12}),
        }
    )
    count24 = spec.model_copy(
        update={
            "s1_settings": _settings(count=24),
            "limits": spec.limits.model_copy(update={"max_feedback_calls": 24}),
        }
    )
    assert canonical_json_bytes(
        count12.model_dump(mode="json")
    ) != canonical_json_bytes(count24.model_dump(mode="json"))
