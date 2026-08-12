from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from skillchain.evaluation.core_fast.engine import CoreFastEngine, FastPathError
from skillchain.evaluation.core_fast.fake_provider import FakeCoreFastAdapter
from skillchain.evaluation.core_fast.models import (
    CAPABILITIES,
    CONFIGS,
    AssistantObservation,
    CallIntent,
    CoreFastSpec,
    S1Settings,
)
from skillchain.evaluation.evaluator_outputs import VisualFeedbackOutput
from skillchain.evaluation.portfolio_gcs import (
    GCSQueryScoreV2,
    build_gcs_population_v2,
)
from skillchain.evaluation.portfolio_treatments import (
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
)
from skillchain.evolution import s1_gcs_gate as frozen_s1_gate
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


@lru_cache(maxsize=1)
def _bank() -> StaticBankArtifact:
    codex_input = Path("specs/authoring/authoring-packet-codex-high-v5.json")
    semantic_input = Path("specs/authoring/authoring-packet-primary-v5-candidate.json")
    draft = Path(
        "runs/formal-authoring/llm-static-codex-primary-20260724-high-v5/"
        "pre-review-draft.json"
    )
    rebind = load_verified_codex_draft_rebind(
        codex_input_path=codex_input,
        expected_codex_input_file_sha256=sha256_bytes(codex_input.read_bytes()),
        semantic_input_path=semantic_input,
        expected_semantic_input_file_sha256=sha256_bytes(semantic_input.read_bytes()),
        draft_path=draft,
        expected_draft_file_sha256=sha256_bytes(draft.read_bytes()),
    )
    return compile_verified_codex_llm_static_bank(
        rebind,
        tool_registry_runtime_sha256="b" * 64,
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


def _observation(
    query: Query,
    *,
    success: bool = False,
    assistant_model: str = "fake-model",
) -> dict[str, object]:
    components = {
        "route_acceptable": True,
        "no_hard_error": True,
        "tool_contract_pass": True,
        "evidence_grounded": success,
        "output_contract_pass": success,
    }
    return AssistantObservation(
        query_id=query.query_id,
        response_text="static",
        selected_capability=query.canonical_capability,
        route_trace_key=f"route:{query.canonical_capability}",
        tool_trace_key=f"tool:{query.query_id}",
        replay_context={
            "response": {"backbone_model": assistant_model},
            "receipt": {
                "model_calls": [
                    {
                        "requested_model": assistant_model,
                        "response_model": assistant_model,
                        "provider_request_id": f"fixture-{query.query_id}",
                    }
                ]
            },
            "assistant_result": {
                "bank_sha256": _bank().bank_sha256,
                "backbone_model": assistant_model,
            },
        },
        answer_mode="supported" if success else "unresolved",
        oracle_available=True,
        gcs_components=components,
        gcs_score=float(success),
        gcs_reason_codes=(),
        hard_error=False,
        evidence_violation=not success,
        assistant_contract="core-fast-deterministic-action-response-v1",
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
    opt_position_by_capability: dict[str, int] = {}
    opt_rows: list[bytes] = []
    for query in opt:
        capability = str(query.canonical_capability)
        position = opt_position_by_capability.get(capability, 0)
        opt_position_by_capability[capability] = position + 1
        success = capability != "utility.document_reading" and position in {1, 6, 7}
        opt_rows.append(canonical_json_bytes(_observation(query, success=success)))
    opt_path.write_bytes(b"".join(opt_rows))
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
    fold_path = tmp_path / "fold-mapping.jsonl"
    fold_path.write_bytes(
        b"".join(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "query_id": query.query_id,
                    "role": "replay" if index >= 600 else "discovery",
                }
            )
            for index, query in enumerate(opt)
        )
    )
    val_queries = [query for query in queries if query.split == "val"]
    gate_path = tmp_path / "val-gates.json"
    gate_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "audit": {
                    "query_id_to_gate": {
                        query.query_id: (
                            "body_gate"
                            if index < 75
                            else "route_gate"
                            if index < 150
                            else "shadow_val"
                        )
                        for index, query in enumerate(val_queries)
                    }
                },
            }
        )
    )
    authoring_path = tmp_path / "authoring-input.json"
    source_authoring = Path(
        "specs/authoring/authoring-packet-primary-v5-candidate.json"
    )
    authoring_path.write_bytes(source_authoring.read_bytes())
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
                    "role": (
                        "failure"
                        if capability == "utility.document_reading"
                        else "anchor"
                    ),
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
        body_rows = (
            opt_rows[:8]
            if capability == "utility.document_reading"
            else [
                opt_rows[0],
                opt_rows[2],
                opt_rows[3],
                opt_rows[4],
                opt_rows[5],
                opt_rows[8],
                opt_rows[6],
                opt_rows[7],
            ]
        )
        body.extend(
            {
                "query_id": query.query_id,
                "capability": capability,
                "role": (
                    "body_failure"
                    if index < 6 or capability == "utility.document_reading"
                    else "body_anchor"
                ),
            }
            for index, query in enumerate(body_rows)
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
            "opt_fold_mapping": str(fold_path),
            "val_gate_assignments": str(gate_path),
            "s1_authoring_input": str(authoring_path),
        },
        "opt_fold_mapping_sha256": sha256_bytes(fold_path.read_bytes()),
        "static_bank_file_sha256": sha256_bytes(bank_path.read_bytes()),
        "opt_static_results_sha256": sha256_bytes(opt_path.read_bytes()),
        "val_gate_assignments_sha256": sha256_bytes(gate_path.read_bytes()),
        "s1_authoring_input_file_sha256": sha256_bytes(authoring_path.read_bytes()),
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
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 2026080601,
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
    engine = CoreFastEngine(
        spec=spec,
        spec_path=spec_path,
        output_root=tmp_path / "output",
        adapter=adapter,
    )
    # Focused engine tests verify orchestration, not wall-clock pacing. A
    # dedicated fake-clock test covers the measured 8 req/s profile.
    engine._feedback_start_pacer.wait = lambda: None
    return engine


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


def test_opt_static_rejects_rows_from_another_assistant_model(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    opt_path = Path(spec.paths.opt_static_results)
    rows = [
        json.loads(line) for line in opt_path.read_text(encoding="utf-8").splitlines()
    ]
    replay_context = rows[0]["replay_context"]
    replay_context["response"]["backbone_model"] = "qwen3-vl-flash-2026-01-22"
    replay_context["assistant_result"]["backbone_model"] = "qwen3-vl-flash-2026-01-22"
    opt_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    drifted = spec.model_copy(
        update={"opt_static_results_sha256": sha256_bytes(opt_path.read_bytes())}
    )
    engine = CoreFastEngine(
        spec=drifted,
        spec_path=spec_path,
        output_root=tmp_path / "wrong-assistant-model",
        adapter=FakeCoreFastAdapter(),
    )

    with pytest.raises(
        FastPathError, match="Static opt800 Assistant model differs from spec"
    ):
        engine.opt_static()


def test_fresh_static_opt800_run_writes_model_bound_bootstrap(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    opt_path = tmp_path / "fresh-lineage" / "opt800-static-observations.jsonl"
    bootstrap_spec = spec.model_copy(
        update={
            "experiment_id": "test-fast-static-bootstrap",
            "paths": spec.paths.model_copy(
                update={"opt_static_results": str(opt_path)}
            ),
            # The producer path deliberately does not trust this placeholder;
            # its bootstrap receipt freezes the hash after the create-only run.
            "opt_static_results_sha256": "0" * 64,
        }
    )
    adapter = FakeCoreFastAdapter()
    engine = CoreFastEngine(
        spec=bootstrap_spec,
        spec_path=spec_path,
        output_root=tmp_path / "fresh-static-output",
        adapter=adapter,
    )

    engine.initialize_static_opt800()
    bootstrap = engine.run_static_opt800()

    assert adapter.calls["assistant"] == 800
    assert bootstrap["row_count"] == 800
    assert bootstrap["assistant_model"] == "fake-model"
    assert bootstrap["opt_static_results_sha256"] == sha256_bytes(opt_path.read_bytes())
    rows = [
        json.loads(line) for line in opt_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 800
    request_ids = []
    for row in rows:
        context = row["replay_context"]
        assert context["response"]["backbone_model"] == "fake-model"
        assert context["assistant_result"]["backbone_model"] == "fake-model"
        call = context["receipt"]["model_calls"][0]
        assert call["requested_model"] == "fake-model"
        assert call["response_model"] == "fake-model"
        request_ids.append(call["provider_request_id"])
    assert len(request_ids) == len(set(request_ids)) == 800
    assert len(bootstrap["fixed_samples"]["canary12"]) == 12
    assert len(bootstrap["fixed_samples"]["dev_smoke24"]) == 24
    assert len(bootstrap["fixed_samples"]["body48"]) == 48
    assert (engine.output_root / "static-opt800-manifest.json").is_file()
    assert (engine.output_root / "static-opt800-bootstrap.json").is_file()


@pytest.mark.parametrize(
    ("field", "loader", "message"),
    (
        (
            "opt_static_results_sha256",
            "opt_static",
            "Static opt800 observations SHA-256",
        ),
        ("opt_fold_mapping_sha256", "opt_fold_roles", "opt fold mapping SHA-256"),
        (
            "val_gate_assignments_sha256",
            "val_gate_roles",
            "validation gate assignment SHA-256",
        ),
        (
            "s1_authoring_input_file_sha256",
            "s1_authoring_input",
            "S1 AuthoringInput SHA-256",
        ),
    ),
)
def test_frozen_s1_input_hash_drift_fails_closed(
    fast_fixture,
    tmp_path: Path,
    field: str,
    loader: str,
    message: str,
) -> None:
    spec, spec_path, _ = fast_fixture
    drifted = spec.model_copy(update={field: "0" * 64})
    engine = CoreFastEngine(
        spec=drifted,
        spec_path=spec_path,
        output_root=tmp_path / field,
        adapter=FakeCoreFastAdapter(),
    )
    with pytest.raises(FastPathError, match=message):
        getattr(engine, loader)()


@pytest.mark.parametrize(
    ("phase", "candidate_macro", "ci_low", "accepted", "reason"),
    (
        ("replay200", 0.20, None, True, None),
        ("body_gate75", 0.22, 0.0, True, None),
        (
            "body_gate75",
            0.2199,
            0.0,
            False,
            "capability-macro GCS delta is below 2pp",
        ),
        (
            "body_gate75",
            0.22,
            -0.0001,
            False,
            "S1 component-bootstrap CI95 lower bound is below 0pp",
        ),
    ),
)
def test_s1_gate_keeps_frozen_macro_and_bootstrap_boundaries(
    fast_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    candidate_macro: float,
    ci_low: float | None,
    accepted: bool,
    reason: str | None,
) -> None:
    engine = _engine(fast_fixture, tmp_path, FakeCoreFastAdapter())
    queries = [query for query in fast_fixture[2] if query.split == "val"][:75]

    def summary(macro: float) -> dict[str, object]:
        return {
            "query_count": 75,
            "capability_macro_gcs": macro,
            "query_micro_gcs": macro,
            "capability_gcs": {capability: macro for capability in CAPABILITIES},
            "hard_errors": 0,
            "card_violations": 0,
            "evidence_violations": 0,
            "tool_violations": 0,
        }

    summaries = iter((summary(0.20), summary(candidate_macro)))
    monkeypatch.setattr(engine, "_summary", lambda *_args: next(summaries))
    if phase == "body_gate75":
        monkeypatch.setattr(
            engine,
            "_paired_component_bootstrap",
            lambda *_args, **_kwargs: {
                "component_count": 1,
                "replicates_requested": 10_000,
                "replicates_available": 10_000,
                "ci95_low_pp": ci_low,
                "ci95_high_pp": 3.0,
                "derived_seed": 1,
                "draw_stream_sha256": "0" * 64,
            },
        )

    passed, reasons, _metrics = engine._s1_gate({}, {}, queries, phase=phase)
    assert passed is accepted
    if reason is None:
        assert reasons == ()
    else:
        assert reason in reasons


def test_s1_replay_gate_rejects_incomplete_oracle_coverage(
    fast_fixture, tmp_path: Path
) -> None:
    engine = _engine(fast_fixture, tmp_path, FakeCoreFastAdapter())
    queries = tuple(query for query in fast_fixture[2] if query.split == "opt_pool")[
        :200
    ]
    parent = {
        query.query_id: AssistantObservation.model_validate(
            _observation(query, success=True), strict=True
        )
        for query in queries
    }
    candidate = dict(parent)
    failed_id = queries[0].query_id
    candidate[failed_id] = candidate[failed_id].model_copy(
        update={"oracle_available": False}
    )

    passed, reasons, metrics = engine._s1_gate(
        parent, candidate, queries, phase="replay200"
    )
    assert not passed
    assert "S1 oracle coverage is incomplete" in reasons
    assert metrics["oracle_coverage_complete"] is False
    assert metrics["oracle_coverage_failure_query_ids"] == [failed_id]


def test_fast_bootstrap_matches_frozen_s1_gate_byte_for_byte(
    fast_fixture, tmp_path: Path
) -> None:
    engine = _engine(fast_fixture, tmp_path, FakeCoreFastAdapter())
    queries = tuple(engine._queries_for_val_gate("body_gate"))
    population = build_gcs_population_v2(queries)
    component_by_query = {
        item.query_id: item.component_id for item in population.bindings
    }
    parent: dict[str, AssistantObservation] = {}
    candidate: dict[str, AssistantObservation] = {}
    parent_scores: list[GCSQueryScoreV2] = []
    candidate_scores: list[GCSQueryScoreV2] = []

    def score(
        query: Query,
        *,
        config: str,
        success: bool,
    ) -> GCSQueryScoreV2:
        return GCSQueryScoreV2(
            query_id=query.query_id,
            config=config,
            canonical_capability=query.canonical_capability,
            component_id=component_by_query[query.query_id],
            route_disposition="pass",
            answer_mode="supported",
            oracle_available=True,
            semantic_claim_support_resolved=True,
            route_acceptable=1,
            no_hard_error=1,
            tool_contract_pass=1,
            evidence_grounded=int(success),
            output_contract_pass=1,
            hard_error=0,
            gcs=int(success),
            reason_codes=() if success else ("material_claim_uncited",),
            evaluated_capability=query.canonical_capability,
            style_support_status=(
                "candidates"
                if query.canonical_capability == "product.style_recommendation"
                else None
            ),
        )

    for index, query in enumerate(queries):
        parent_success = index % 5 == 0
        candidate_success = parent_success or index % 7 == 0
        parent[query.query_id] = AssistantObservation.model_validate(
            _observation(query, success=parent_success), strict=True
        )
        candidate[query.query_id] = AssistantObservation.model_validate(
            _observation(query, success=candidate_success), strict=True
        )
        parent_scores.append(score(query, config="llm_static", success=parent_success))
        candidate_scores.append(score(query, config="s1", success=candidate_success))

    fast = engine._paired_component_bootstrap(
        parent,
        candidate,
        queries,
        scope=frozen_s1_gate.S1_BODY_GATE_SCOPE,
    )
    frozen = frozen_s1_gate._paired_gcs_v2_contrast(
        baseline=tuple(parent_scores),
        candidate=tuple(candidate_scores),
        queries=queries,
        scope=frozen_s1_gate.S1_BODY_GATE_SCOPE,
    )
    assert fast == {
        "component_count": frozen.component_count,
        "replicates_requested": frozen.replicates,
        "replicates_available": frozen.macro_available_replicates,
        "ci95_low_pp": frozen.macro_ci95_low_pp,
        "ci95_high_pp": frozen.macro_ci95_high_pp,
        "derived_seed": frozen.derived_seed,
        "draw_stream_sha256": frozen.draw_stream_sha256,
    }


class _MixedPolicyFeedbackAdapter(FakeCoreFastAdapter):
    def invoke(self, intent: CallIntent):
        result = super().invoke(intent)
        if intent.role != "feedback" or result.status != "success":
            return result
        feedback = VisualFeedbackOutput(
            schema_version=1,
            summary="mixed policy dispositions",
            rule_violations=(),
            ideal_response_gaps=(),
            skill_suggestions=(
                "[policy_compatible] preserve the frozen contract",
                "[requires_new_evidence] add an unsupported lookup",
                "[rejected] weaken the fallback boundary",
            ),
        )
        return result.model_copy(
            update={"output": {"feedback": feedback.model_dump(mode="json")}}
        )


class _S1ScreenAdapter(FakeCoreFastAdapter):
    def __init__(
        self,
        *,
        patch_capabilities: frozenset[str] = frozenset({"product.exact_match"}),
        smoke_hard_query_id: str | None = None,
        raw_contract_failure_capabilities: frozenset[str] = frozenset(),
        raw_force_success_capabilities: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__()
        self.patch_capabilities = patch_capabilities
        self.smoke_hard_query_id = smoke_hard_query_id
        self.raw_contract_failure_capabilities = raw_contract_failure_capabilities
        self.raw_force_success_capabilities = raw_force_success_capabilities

    def invoke(self, intent: CallIntent):
        result = super().invoke(intent)
        if (
            intent.role == "creator"
            and intent.payload.get("operation") == "s1_creator"
            and result.status == "success"
        ):
            parent = intent.payload["parent_bank"]
            templates = intent.payload["parent_authoring_content"]
            assert isinstance(parent, dict) and isinstance(templates, list)
            skills = parent["skills"]
            assert isinstance(skills, list)
            template_by_capability = {
                str(item["capability_id"]): item
                for item in templates
                if isinstance(item, dict)
            }
            generated = []
            for skill in sorted(skills, key=lambda item: str(item["capability_id"])):
                capability = str(skill["capability_id"])
                entry = {
                    "capability_id": capability,
                    "action": "inherit",
                    "parent_skill_sha256": skill["skill_sha256"],
                }
                if capability in self.patch_capabilities:
                    template = template_by_capability[capability]
                    entry = {
                        **entry,
                        "action": "patch",
                        "patch": {
                            "objective": template["objective"],
                            "steps": template["steps"],
                            "fallback_instruction": (
                                str(template["fallback_instruction"])
                                + " Make the supported-evidence boundary explicit."
                            ),
                            "citation_source_ids": template["citation_source_ids"],
                        },
                    }
                generated.append(entry)
            return result.model_copy(
                update={"output": {"schema_version": 1, "skills": generated}}
            )
        if (
            intent.role != "assistant"
            or result.status != "success"
            or not isinstance(result.output, dict)
            or not isinstance(result.output.get("observation"), dict)
        ):
            return result
        query = intent.payload["query"]
        assert isinstance(query, dict)
        query_id = str(query["query_id"])
        capability = str(query["canonical_capability"])
        is_smoke_failure = (
            intent.call_id.startswith("dev-smoke24-s1-candidate-")
            and query_id == self.smoke_hard_query_id
        )
        is_raw_replay = intent.call_id.startswith("opt-replay200-s1-candidate-")
        force_success = (
            is_raw_replay and capability in self.raw_force_success_capabilities
        )
        contract_failure = is_smoke_failure or (
            is_raw_replay and capability in self.raw_contract_failure_capabilities
        )
        if not force_success and not contract_failure:
            return result
        output = dict(result.output)
        observation = dict(output["observation"])
        components = dict(observation["gcs_components"])
        if force_success:
            components = {key: True for key in components}
            observation.update(
                {
                    "selected_capability": capability,
                    "route_trace_key": f"route:{capability}",
                    "gcs_components": components,
                    "gcs_score": 1.0,
                    "gcs_reason_codes": [],
                    "hard_error": False,
                    "answer_mode": "supported",
                    "repair": "none",
                    "card_violation": False,
                    "evidence_violation": False,
                    "tool_violation": False,
                }
            )
        if contract_failure:
            components["no_hard_error"] = False
            components["output_contract_pass"] = False
            observation.update(
                {
                    "gcs_components": components,
                    "gcs_score": 0.0,
                    "gcs_reason_codes": [
                        "assistant_hard_error",
                        "output_section_invalid",
                    ],
                    "hard_error": True,
                    "answer_mode": "unresolved",
                    "repair": "response_contract_error",
                }
            )
        output["observation"] = observation
        return result.model_copy(update={"output": output})


def test_creator_projection_keeps_only_policy_compatible_feedback(
    fast_fixture, tmp_path: Path
) -> None:
    engine = _engine(
        fast_fixture,
        tmp_path,
        _MixedPolicyFeedbackAdapter(reject_stages=frozenset({"s1"})),
    )
    engine.run_s1()
    intent = json.loads(
        (
            engine.output_root / "calls" / "creator" / "s1-creator-once.intent.json"
        ).read_text(encoding="utf-8")
    )
    guard = intent["payload"]["requirements"]["author_content_lexical_guard"]
    assert set(guard["forbidden_whole_words"]) >= {
        "label",
        "labels",
        "evaluation",
        "trajectory",
    }
    assert guard["required_detector_prediction_phrase"] == "predicted class name"
    assert "same path token" in guard["forbidden_path_reference_rule"]
    for rows in intent["payload"]["feedback_by_capability"].values():
        for row in rows:
            assert row["feedback"]["skill_suggestions"] == [
                "[policy_compatible] preserve the frozen contract"
            ]


def test_s1_round_focus_is_bound_and_missing_required_phrase_fails_closed(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    focused = S1Settings(
        round_id="r3",
        max_patched_capabilities=1,
        protected_capabilities=(
            "knowledge.visual_encyclopedia",
            "product.multi_search",
            "product.style_recommendation",
            "utility.document_reading",
            "utility.recipe_guidance",
        ),
        creator_directives=("Keep the literal fallback marker.",),
        required_patch_phrases={
            "product.exact_match": ("no supported match",),
        },
    )
    engine = CoreFastEngine(
        spec=spec.model_copy(update={"s1_settings": focused}),
        spec_path=spec_path,
        output_root=tmp_path / "focused",
        adapter=FakeCoreFastAdapter(),
    )

    decision = engine.run_s1()

    assert not decision.accepted
    assert decision.reasons == (
        "Creator failed or returned an invalid sparse S1 proposal",
    )
    assert (
        decision.metrics["creator_candidate_rejection_reason"]
        == "required_patch_phrase_missing"
    )
    assert not (engine.output_root / "banks" / "s1-candidate.json").exists()
    intent = json.loads(
        (
            engine.output_root / "calls" / "creator" / "s1-creator-once.intent.json"
        ).read_text(encoding="utf-8")
    )
    requirements = intent["payload"]["requirements"]
    assert requirements["max_patched_capabilities"] == 1
    assert requirements["creator_directives"] == ["Keep the literal fallback marker."]
    assert requirements["required_patch_phrases"] == {
        "product.exact_match": ["no supported match"]
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


def test_s1_smoke_hard_error_on_inherited_capability_does_not_block_replay(
    fast_fixture, tmp_path: Path
) -> None:
    smoke_query_id = next(
        item.query_id
        for item in fast_fixture[0].fixed_samples.dev_smoke24
        if item.capability == "utility.recipe_guidance"
    )
    adapter = _S1ScreenAdapter(smoke_hard_query_id=smoke_query_id)
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.initialize()

    decision = engine.run_s1()

    assert decision.metrics["smoke_hard_errors"] == 1
    assert decision.metrics["smoke_oracle_coverage_complete"] is True
    assert not any("dev smoke24" in reason for reason in decision.reasons)
    assert (engine.output_root / "banks" / "s1-development-screen.json").is_file()
    assistant_calls = engine.output_root / "calls" / "assistant"
    assert list(assistant_calls.glob("opt-replay200-s1-candidate-*.intent.json"))
    smoke_rows = {}
    for item in fast_fixture[0].fixed_samples.dev_smoke24:
        result = engine.calls.get(
            "assistant",
            engine._assistant_call_id("dev-smoke24", "s1-candidate", item.query_id),
        )
        assert result is not None and isinstance(result.output, dict)
        smoke_rows[item.query_id] = AssistantObservation.model_validate(
            result.output["observation"], strict=True
        )
    assert engine._s1_smoke_ok(smoke_rows)
    assert not engine._smoke_ok(smoke_rows)


def test_s1_smoke_oracle_failure_still_stops_before_raw_replay(
    fast_fixture, tmp_path: Path
) -> None:
    failed_id = fast_fixture[0].fixed_samples.dev_smoke24[0].query_id
    adapter = FakeCoreFastAdapter(assistant_fail_ids=frozenset({failed_id}))
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.initialize()

    decision = engine.run_s1()

    assert not decision.accepted
    assert decision.metrics["smoke_oracle_coverage_complete"] is False
    assert decision.metrics["smoke_oracle_coverage_failure_query_ids"] == [failed_id]
    assert decision.reasons == (
        "candidate failed fixed dev smoke24 operational/oracle coverage",
    )
    assistant_calls = engine.output_root / "calls" / "assistant"
    assert not list(assistant_calls.glob("opt-replay200-*.intent.json"))
    assert not (engine.output_root / "banks" / "s1-development-screen.json").exists()


def test_s1_raw_replay_oracle_failure_stops_before_development_screen(
    fast_fixture, tmp_path: Path
) -> None:
    adapter = FakeCoreFastAdapter()
    engine = _engine(fast_fixture, tmp_path, adapter)
    replay_roles = engine.opt_fold_roles()
    failed_id = next(
        query.query_id
        for query in fast_fixture[2]
        if query.split == "opt_pool" and replay_roles[query.query_id] == "replay"
    )
    adapter.assistant_fail_ids = frozenset({failed_id})
    engine.initialize()

    decision = engine.run_s1()

    assert not decision.accepted
    assert decision.metrics["raw_replay_oracle_coverage_complete"] is False
    assert decision.metrics["raw_replay_oracle_coverage_failure_query_ids"] == [
        failed_id
    ]
    assert "raw replay200 oracle coverage is incomplete" in decision.reasons
    assert not (engine.output_root / "banks" / "s1-development-screen.json").exists()
    assistant_calls = engine.output_root / "calls" / "assistant"
    assert not list(assistant_calls.glob("opt-replay200-composite-*.intent.json"))


def test_s1_replay_screen_reverts_only_document_and_reruns_composite(
    fast_fixture, tmp_path: Path
) -> None:
    retained = frozenset({"product.exact_match", "product.multi_search"})
    adapter = _S1ScreenAdapter(
        patch_capabilities=retained | {"utility.document_reading"},
        raw_contract_failure_capabilities=frozenset({"utility.document_reading"}),
        raw_force_success_capabilities=retained,
    )
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.initialize()

    decision = engine.run_s1()

    assert decision.metrics["retained_patch_capabilities"] == sorted(retained)
    assert decision.metrics["reverted_patch_capabilities"] == [
        "utility.document_reading"
    ]
    assert decision.metrics["composite_replay_rerun"] is True
    screen = json.loads(
        (engine.output_root / "banks" / "s1-development-screen.json").read_text(
            encoding="utf-8"
        )
    )
    screen_by_capability = {
        item["capability_id"]: item["decision"] for item in screen["decisions"]
    }
    assert screen_by_capability["utility.document_reading"] == "inherit_parent"
    assert all(screen_by_capability[item] == "retain_patch" for item in retained)

    parent = StaticBankArtifact.model_validate_json(
        (engine.output_root / "banks" / "llm_static.json").read_bytes(), strict=True
    )
    raw = StaticBankArtifact.model_validate_json(
        (engine.output_root / "banks" / "s1-candidate.json").read_bytes(), strict=True
    )
    screened = StaticBankArtifact.model_validate_json(
        (engine.output_root / "banks" / "s1-screened-candidate.json").read_bytes(),
        strict=True,
    )
    parent_by_capability = {item.capability_id: item for item in parent.skills}
    raw_by_capability = {item.capability_id: item for item in raw.skills}
    screened_by_capability = {item.capability_id: item for item in screened.skills}
    assert canonical_json_bytes(
        screened_by_capability["utility.document_reading"].model_dump(mode="json")
    ) == canonical_json_bytes(
        parent_by_capability["utility.document_reading"].model_dump(mode="json")
    )
    for capability in retained:
        assert canonical_json_bytes(
            screened_by_capability[capability].model_dump(mode="json")
        ) == canonical_json_bytes(raw_by_capability[capability].model_dump(mode="json"))
    assistant_calls = engine.output_root / "calls" / "assistant"
    assert list(
        assistant_calls.glob("opt-replay200-composite-s1-candidate-*.intent.json")
    )


def test_s1_replay_screen_stops_when_all_patches_revert(
    fast_fixture, tmp_path: Path
) -> None:
    adapter = _S1ScreenAdapter(
        raw_contract_failure_capabilities=frozenset({"product.exact_match"})
    )
    engine = _engine(fast_fixture, tmp_path, adapter)
    engine.initialize()

    decision = engine.run_s1()

    assert not decision.accepted
    assert decision.metrics["retained_patch_capabilities"] == []
    assert decision.metrics["reverted_patch_capabilities"] == ["product.exact_match"]
    assert "replay200 screen reverted all sparse patches" in decision.reasons
    assistant_calls = engine.output_root / "calls" / "assistant"
    assert not list(assistant_calls.glob("opt-replay200-composite-*.intent.json"))
    assert not list(assistant_calls.glob("body-gate75-*.intent.json"))


@pytest.mark.parametrize(
    "artifact_name",
    (
        "s1-development-screen.json",
        "s1-screened-candidate.json",
        "s1-screened-bank-receipt.json",
    ),
)
def test_s1_screened_artifact_drift_fails_closed_on_resume(
    fast_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    engine = _engine(fast_fixture, tmp_path, FakeCoreFastAdapter())
    engine.initialize()
    assistant_many = engine._assistant_many

    def interrupt_before_composite(*, split: str, **kwargs):
        if split == "opt-replay200-composite":
            raise FastPathError("injected interruption before composite replay")
        return assistant_many(split=split, **kwargs)

    monkeypatch.setattr(engine, "_assistant_many", interrupt_before_composite)
    with pytest.raises(FastPathError, match="injected interruption"):
        engine.run_s1()
    monkeypatch.setattr(engine, "_assistant_many", assistant_many)
    artifact = engine.output_root / "banks" / artifact_name
    artifact.write_bytes(artifact.read_bytes() + b" ")

    with pytest.raises(FastPathError, match="changed on resume"):
        engine.run_s1()


def test_feedback_reuse_mode_fails_closed_without_bound_cache(
    fast_fixture, tmp_path: Path
) -> None:
    spec, spec_path, _ = fast_fixture
    reused = spec.model_copy(
        update={
            "s1_settings": spec.s1_settings.model_copy(
                update={
                    "round_id": "r2",
                    "feedback_mode": "reuse-exact-call-results",
                    "feedback_reuse_manifest_sha256": "a" * 64,
                }
            )
        }
    )
    engine = CoreFastEngine(
        spec=reused,
        spec_path=spec_path,
        output_root=tmp_path / "missing-reuse",
        adapter=FakeCoreFastAdapter(),
    )
    with pytest.raises(FastPathError, match="source manifest"):
        engine.run_s1()
    assert not (engine.output_root / "calls").exists()


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
