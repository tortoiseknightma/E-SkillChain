from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillchain.evaluation.core_fast.engine import CoreFastEngine, FastPathError
from skillchain.evaluation.core_fast.fake_provider import FakeCoreFastAdapter
from skillchain.evaluation.core_fast.models import (
    AssistantObservation,
    CAPABILITIES,
    CONFIGS,
    CallIntent,
    CoreFastSpec,
)
from skillchain.schemas import Query
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


_CAPABILITY_META = {
    "knowledge.visual_encyclopedia": ("encyclopedia", False),
    "product.exact_match": ("exact_match", True),
    "product.multi_search": ("multi_product", True),
    "product.style_recommendation": ("divergent_rec", True),
    "utility.document_reading": ("utility", False),
    "utility.recipe_guidance": ("utility", False),
}


def _digest(payload: dict[str, object], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def _bank() -> StaticBankArtifact:
    skills = []
    for capability in CAPABILITIES:
        slug = capability.replace(".", "-").replace("_", "-")
        raw: dict[str, object] = {
            "slug": slug,
            "version": 1,
            "description": f"Route {capability}",
            "body": f"# {capability}\n\nFollow the contract.\n",
            "static_refs": [],
            "operators": ["encyclopedia_lookup"],
            "capability_id": capability,
            "parent_skill_sha256": None,
            "skill_sha256": "0" * 64,
        }
        raw["skill_sha256"] = _digest(raw, "skill_sha256")
        skills.append(raw)
    skills.sort(key=lambda item: str(item["slug"]))
    payload: dict[str, object] = {
        "schema_version": 2,
        "baseline_kind": "llm_static",
        "construction_identity_sha256": "c" * 64,
        "construction_identity_policy": "reviewed-draft-v1",
        "runtime_binding_policy": "registry-runtime-v1",
        "compiler": {
            "compiler_id": "skillchain.static-authoring",
            "compiler_version": "3.0.0",
        },
        "tool_registry_sha256": "a" * 64,
        "tool_registry_runtime_sha256": "b" * 64,
        "skills": skills,
        "capability_map": [
            {
                "capability_id": capability,
                "skill_slug": capability.replace(".", "-").replace("_", "-"),
            }
            for capability in CAPABILITIES
        ],
        "bank_sha256": "0" * 64,
    }
    payload["bank_sha256"] = _digest(payload, "bank_sha256")
    return StaticBankArtifact.model_validate_json(
        canonical_json_bytes(payload), strict=True
    )


def _queries() -> list[Query]:
    geometry = (
        ("dev_mini", "dm", 200),
        ("opt_pool", "op", 800),
        ("val", "va", 200),
        ("test_frozen", "te", 300),
    )
    rows: list[Query] = []
    for split, prefix, count in geometry:
        for index in range(count):
            capability = CAPABILITIES[index % len(CAPABILITIES)]
            intent, requires_card = _CAPABILITY_META[capability]
            query_id = f"{prefix}-{index:04d}"
            text = f"query {query_id}"
            rows.append(
                Query(
                    schema_version=2,
                    taxonomy_version="ecommerce-mvp-taxonomy-v0",
                    task_spec_version="ecommerce-task-spec-v1",
                    query_id=query_id,
                    asset_id=f"asset.{query_id}",
                    image_path=f"query_images/{capability}/{query_id}.jpg",
                    leakage_group_id=f"leak-{prefix}-{index // 2:04d}",
                    boundary_group_id=None,
                    template_family=f"template-{index % 5}",
                    generator_batch_id=f"batch-{prefix}-{index // 25:02d}",
                    text=text,
                    turns=[{"role": "user", "content": text}],
                    canonical_intent=intent,
                    canonical_capability=capability,
                    acceptable_capabilities=[capability],
                    is_boundary=False,
                    boundary_strategy=None,
                    requires_card=requires_card,
                    split=split,
                    label_status="auto",
                    label_provenance=[
                        {
                            "decision_type": "constructed",
                            "annotator_kind": "planner",
                            "annotator_id": "core-fast-test",
                            "canonical_intent": intent,
                            "canonical_capability": capability,
                            "acceptable_capabilities": [capability],
                        }
                    ],
                )
            )
    return rows


def _observation(query: Query) -> dict[str, object]:
    components = {
        "route_acceptable": True,
        "no_hard_error": True,
        "tool_contract_pass": True,
        "evidence_grounded": False,
        "output_contract_pass": False,
    }
    return AssistantObservation(
        query_id=query.query_id,
        response_text="static",
        selected_capability=query.canonical_capability,
        route_trace_key=f"route:{query.canonical_capability}",
        tool_trace_key=f"tool:{query.query_id}",
        gcs_components=components,
        gcs_score=0.0,
        hard_error=False,
        evidence_violation=True,
    ).model_dump(mode="json")


@pytest.fixture
def fast_fixture(tmp_path: Path):
    queries = _queries()
    bank = _bank()
    query_path = tmp_path / "queries.jsonl"
    query_path.write_bytes(
        b"".join(
            canonical_json_bytes(query.model_dump(mode="json")) for query in queries
        )
    )
    bank_path = tmp_path / "bank.json"
    bank_path.write_bytes(bank.canonical_bytes())
    opt = [query for query in queries if query.split == "opt_pool"]
    opt_path = tmp_path / "opt-static.jsonl"
    opt_path.write_bytes(
        b"".join(canonical_json_bytes(_observation(query)) for query in opt)
    )
    route_path = tmp_path / "route.jsonl"
    route_path.write_bytes(
        b"".join(
            canonical_json_bytes(
                {
                    "query_id": query.query_id,
                    "expected_capability": query.canonical_capability,
                    "local_capability": query.canonical_capability,
                }
            )
            for query in opt
        )
    )
    by_split_capability: dict[tuple[str, str], list[Query]] = {}
    for query in queries:
        by_split_capability.setdefault(
            (query.split, str(query.canonical_capability)), []
        ).append(query)
    canary = []
    smoke = []
    body = []
    for capability in CAPABILITIES:
        opt_rows = by_split_capability[("opt_pool", capability)]
        dev_rows = by_split_capability[("dev_mini", capability)]
        canary.extend(
            (
                {
                    "query_id": opt_rows[0].query_id,
                    "capability": capability,
                    "role": "failure",
                },
                {
                    "query_id": opt_rows[1].query_id,
                    "capability": capability,
                    "role": "anchor",
                },
            )
        )
        smoke.extend(
            {
                "query_id": query.query_id,
                "capability": capability,
                "role": "smoke",
            }
            for query in dev_rows[:4]
        )
        body.extend(
            {
                "query_id": query.query_id,
                "capability": capability,
                "role": "body_failure" if index < 6 else "body_anchor",
            }
            for index, query in enumerate(opt_rows[:8])
        )
    spec_payload = {
        "schema_version": 1,
        "kind": "core-experiment-fast-v1",
        "experiment_id": "test-fast",
        "paths": {
            "queries": str(query_path),
            "static_bank": str(bank_path),
            "opt_static_results": str(opt_path),
            "opt_route_attribution": str(route_path),
        },
        "split_counts": {
            "dev_mini": 200,
            "opt_pool": 800,
            "val": 200,
            "test_frozen": 300,
        },
        "capabilities": list(CAPABILITIES),
        "configs": list(CONFIGS),
        "fixed_samples": {
            "canary12": canary,
            "dev_smoke24": smoke,
            "body48": body,
        },
        "models": {
            role: {
                "provider": "fake",
                "requested_model": (
                    "qwen3.8-max"
                    if role == "feedback"
                    else "gpt-5.6-sol"
                    if role == "creator"
                    else "fake-model"
                ),
                "moving_alias": role == "feedback",
                "reasoning_effort": "high" if role == "creator" else None,
                "credential_env": None,
                "estimated_call_cost_cny": 0.001,
            }
            for role in ("feedback", "creator", "assistant", "judge", "route_only")
        },
        "runtime": {
            "adapter": "python",
            "python_factory": "skillchain.evaluation.core_fast.fake_provider:create_adapter",
            "commands": {},
        },
        "bootstrap_replicates": 100,
        "bootstrap_seed": 7,
        "disclosures": ["test disclosure"],
    }
    spec = CoreFastSpec.model_validate_json(
        canonical_json_bytes(spec_payload), strict=True
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(canonical_json_bytes(spec_payload))
    return spec, spec_path, queries


def _engine(
    fast_fixture,
    tmp_path: Path,
    adapter: FakeCoreFastAdapter,
) -> CoreFastEngine:
    spec, spec_path, _queries = fast_fixture
    return CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=tmp_path / "output",
        adapter=adapter,
    )


def test_validate_fixes_core_geometry_and_samples(fast_fixture, tmp_path: Path) -> None:
    engine = _engine(fast_fixture, tmp_path, FakeCoreFastAdapter())
    summary = engine.validate(require_runtime=True)
    assert summary.query_count == 1500
    assert summary.split_counts == {
        "dev_mini": 200,
        "opt_pool": 800,
        "val": 200,
        "test_frozen": 300,
    }


def test_partial_feedback_still_invokes_creator_exactly_once(
    fast_fixture, tmp_path: Path
) -> None:
    failed_id = fast_fixture[0].fixed_samples.canary12[0].query_id
    adapter = FakeCoreFastAdapter(feedback_fail_ids=frozenset({failed_id}))
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.initialize()
    decision = engine.run_s1()
    assert decision.accepted
    assert adapter.calls["feedback"] == 12
    assert adapter.calls["creator"] == 1
    assert decision.metrics["feedback_success_count"] == 11


@pytest.mark.parametrize(
    ("status", "expected_calls"),
    (
        ("schema_error", 2),
        ("empty_response", 2),
        ("invalid_output", 2),
        ("provider_error", 1),
    ),
)
def test_final_judge_retries_only_format_failures(
    fast_fixture, tmp_path: Path, status: str, expected_calls: int
) -> None:
    adapter = FakeCoreFastAdapter(judge_first_status=status)
    engine = _engine(fast_fixture, tmp_path, adapter)
    query = next(query for query in fast_fixture[2] if query.split == "val")
    observation = AssistantObservation.model_validate(_observation(query), strict=True)
    engine._judge_one(
        split="unit",
        config="candidate",
        query=query,
        observation=observation,
    )
    assert adapter.calls["judge"] == expected_calls


def test_intent_without_result_becomes_unknown_and_is_not_replayed(
    fast_fixture, tmp_path: Path
) -> None:
    adapter = FakeCoreFastAdapter()
    engine = _engine(fast_fixture, tmp_path, adapter)
    intent = CallIntent(
        call_id="crashed",
        role="feedback",
        purpose="crash test",
        requested_model="qwen3.8-max",
        payload={},
    )
    path = engine.output_root / "calls" / "feedback" / "crashed.intent.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(intent.model_dump(mode="json")))
    result = engine._call(
        role="feedback", call_id="crashed", purpose="crash test", payload={}
    )
    assert result.status == "interrupted_unknown"
    assert adapter.calls["feedback"] == 0


def test_fake_provider_end_to_end_aliases_without_extra_calls(
    fast_fixture, tmp_path: Path
) -> None:
    adapter = FakeCoreFastAdapter(reject_stages=frozenset({"s1", "s2", "full"}))
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.run(through="test")
    val = (
        (engine.output_root / "reports" / "val-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    test = (
        (engine.output_root / "reports" / "test-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(val) == 1000
    assert len(test) == 1500
    rows = [json.loads(line) for line in test]
    by_config_query = {(row["config"], row["query"]["query_id"]): row for row in rows}
    for query_id in {row["query"]["query_id"] for row in rows}:
        static = by_config_query[("llm_static", query_id)]
        for alias in ("s1", "s1s2", "full"):
            row = by_config_query[(alias, query_id)]
            assert row["assistant"] == static["assistant"]
            assert row["judge"] == static["judge"]
            assert row["physical_config"] == "llm_static"
    # Only NoSkill and Static are physical test configurations.
    test_assistant_intents = list(
        (engine.output_root / "calls" / "assistant").glob("test_frozen-*.intent.json")
    )
    assert len(test_assistant_intents) == 600
    summary = json.loads(
        (engine.output_root / "reports" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["geometry"] == {
        "val_logical_rows": 1000,
        "test_logical_rows": 1500,
    }
    assert adapter.calls["creator"] == 3
    assert adapter.calls["judge"] == 600

    # `report` is a pure rebuild: deleting only the derived matrices must not
    # cause Assistant/Judge resampling.
    before = adapter.calls.copy()
    (engine.output_root / "reports" / "val-results.jsonl").unlink()
    (engine.output_root / "reports" / "test-results.jsonl").unlink()
    engine.report()
    assert adapter.calls == before


def test_s2_and_s3_edit_boundaries_reject_forbidden_fields(
    fast_fixture, tmp_path: Path
) -> None:
    engine = _engine(fast_fixture, tmp_path, FakeCoreFastAdapter())
    parent = engine.static_bank()
    changed = parent.model_copy(
        update={
            "skills": tuple(
                skill.model_copy(update={"body": skill.body + "bad\n"})
                if index == 0
                else skill
                for index, skill in enumerate(parent.skills)
            )
        }
    )
    assert engine._boundary_changes(parent, changed, "s2")

    incomplete_s1 = {
        "skills": [
            {
                "capability_id": parent.skills[0].capability_id,
                "description": parent.skills[0].description,
                "body": parent.skills[0].body,
                "static_refs": list(parent.skills[0].static_refs),
                "operators": list(parent.skills[0].operators),
            }
        ]
    }
    assert (
        engine._compile_candidate_payload(incomplete_s1, stage="s1", parent=parent)
        is None
    )


def test_accepted_three_stage_path_enforces_common_trace(
    fast_fixture, tmp_path: Path
) -> None:
    adapter = FakeCoreFastAdapter()
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.run(through="full")
    assert engine._existing_decision("s1").accepted  # type: ignore[union-attr]
    assert engine._existing_decision("s2").accepted  # type: ignore[union-attr]
    full = engine._existing_decision("full")
    assert full is not None and full.accepted
    assert full.metrics["gate"]["common_trace_violation_query_ids"] == []  # type: ignore[index]
    assert adapter.calls["creator"] == 3
    val_path = engine.output_root / "reports" / "val-results.jsonl"
    assert len(val_path.read_text(encoding="utf-8").splitlines()) == 1000
    assert not list(
        (engine.output_root / "calls" / "assistant").glob("test_frozen-*.intent.json")
    )


def test_test_responses_are_sealed_until_all_banks_are_frozen(
    fast_fixture, tmp_path: Path
) -> None:
    engine = _engine(fast_fixture, tmp_path, FakeCoreFastAdapter())
    query = next(query for query in fast_fixture[2] if query.split == "test_frozen")
    with pytest.raises(FastPathError, match="sealed"):
        engine._assistant_many(
            split="test_frozen",
            config="llm_static",
            queries=[query],
            bank=engine.static_bank(),
        )
    assert not (engine.output_root / "calls").exists()


def test_assistant_provider_failure_is_kept_in_fixed_denominator(
    fast_fixture, tmp_path: Path
) -> None:
    query = next(query for query in fast_fixture[2] if query.split == "val")
    engine = _engine(
        fast_fixture,
        tmp_path,
        FakeCoreFastAdapter(assistant_fail_ids=frozenset({query.query_id})),
    )
    rows = engine._assistant_many(
        split="unit-fixed-denominator",
        config="llm_static",
        queries=[query],
        bank=engine.static_bank(),
    )
    failed = rows[query.query_id]
    assert failed.gcs_score == 0.0
    assert failed.hard_error


def test_route_report_reuses_input_opt800_static_results(
    fast_fixture, tmp_path: Path
) -> None:
    adapter = FakeCoreFastAdapter(reject_stages=frozenset({"s1", "s2", "full"}))
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.run(through="test")
    opt_static_route_call = (
        engine.output_root
        / "calls"
        / "route_only"
        / "all1500-llm_static-op-0000.intent.json"
    )
    assert not opt_static_route_call.exists()


def test_s2_runs_from_static_parent_after_s1_rollback(
    fast_fixture, tmp_path: Path
) -> None:
    adapter = FakeCoreFastAdapter(reject_stages=frozenset({"s1"}))
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.run(through="s2")
    s1 = engine._existing_decision("s1")
    s2 = engine._existing_decision("s2")
    assert s1 is not None and not s1.accepted
    assert s2 is not None and s2.accepted
    assert s2.parent_bank == engine.static_bank().bank_sha256
    creator_intent = json.loads(
        (
            engine.output_root
            / "calls"
            / "creator"
            / "s2-route-optimizer-once.intent.json"
        ).read_text(encoding="utf-8")
    )
    assert len(creator_intent["payload"]["boundary_examples"]) <= 48


def test_common_trace_violation_rolls_back_full_without_aborting(
    fast_fixture, tmp_path: Path
) -> None:
    engine = _engine(
        fast_fixture,
        tmp_path,
        FakeCoreFastAdapter(break_s3_common_trace=True),
    )
    engine.run(through="full")
    decision = engine._existing_decision("full")
    assert decision is not None and not decision.accepted
    assert "candidate changed route/tool trace in smoke24" in decision.reasons


def test_budget_cutoff_occurs_before_new_intent(fast_fixture, tmp_path: Path) -> None:
    spec, spec_path, _ = fast_fixture
    limited = spec.model_copy(
        update={"limits": spec.limits.model_copy(update={"external_cost_cny": 0.0001})}
    )
    engine = CoreFastEngine(
        spec=limited,
        spec_path=spec_path,
        output_root=tmp_path / "limited",
        adapter=FakeCoreFastAdapter(),
    )
    with pytest.raises(FastPathError, match="hard cap"):
        engine._call(role="feedback", call_id="over", purpose="budget", payload={})
    assert not (engine.output_root / "calls" / "feedback" / "over.intent.json").exists()
