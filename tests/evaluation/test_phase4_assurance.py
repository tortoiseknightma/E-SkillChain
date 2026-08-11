from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError

from skillchain import config as project_config
from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.evaluation.assistant_runs import (
    FORMAL_INELIGIBLE_REASON,
    MAIN_CONFIG_ORDER,
    AssistantBackendContractError,
    AssistantBackendResponse,
    AssistantQueryInput,
    AssistantRunError,
    JudgeAuditSelectionEntry,
    MatrixTreatment,
    RegistryRuntimeLockV2,
    VerifiedFiveConfigRuns,
    build_assistant_matrix_plan,
    build_assistant_query_input,
    create_assistant_matrix_plan_file,
    create_assistant_run_bundle,
    create_runner_owned_assistant_run_bundle,
    load_verified_assistant_matrix_plan,
    load_verified_assistant_run_bundle,
    load_verified_five_config_runs,
    load_verified_phase4_inputs,
    make_backbone_lock,
    make_inference_budget,
    make_phase4_input_selection_manifest,
)
from skillchain.evaluation.evaluator_isolation import (
    EvaluatorIsolationError,
    JudgeAuditDimension,
    JudgeAuditReferenceRow,
    build_bound_feedback_prompt,
    build_bound_final_prompt,
    build_judge_audit_dataset,
    build_verified_final_evaluation_packet,
    create_sealed_judge_audit,
    make_evaluator_isolation_lock,
    make_feedback_evaluator_identity,
    make_final_evaluator_identity,
    open_sealed_judge_audit,
    serialize_feedback_packet,
    serialize_final_evaluation_packet,
)
from skillchain.evaluation.packets import (
    AssistantToolTrace,
    EvaluationImage,
    FeedbackPacket,
    FinalEvaluationPacket,
    JudgeDimensionScore,
    JudgeOutcome,
    JudgeScores,
    RubricSnapshot,
    VisibleCard,
    VisibleToolEvidence,
    build_final_evaluation_packet,
    derive_blinded_evaluation_id,
    validate_paired_result_rows,
)
from skillchain.llm import LLMResponse, LLMToolCall, LLMUsage
from skillchain.runners.assistant import ProductionAssistantRunner
from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.static_authoring import (
    BankCapabilityBinding,
    CompilerIdentity,
    StaticBankArtifact,
    StrictSkillArtifact,
)
from skillchain.synthesis.splitting import FrozenSplitManifest
from skillchain.tools.model_artifacts import ModelRuntimeBinding
from skillchain.tools.registry import (
    MVP_TOOL_NAMES,
    MVP_TOOL_NAMES_V2,
    MVPToolServices,
    ToolRegistry,
    build_mvp_registry,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.taxonomy import TASK_SPEC_VERSION, TAXONOMY_VERSION


class _ProductService:
    artifact_binding = object()

    def __init__(self, runtime_seed: str) -> None:
        self.runtime_seed = runtime_seed

    @property
    def formal_runtime_binding_sha256(self) -> str:
        return hashlib.sha256(f"{self.runtime_seed}:product".encode()).hexdigest()

    def trace_image_product_search(self, *_args, **_kwargs):
        raise AssertionError("not used by Assistant runner contract tests")

    trace_text_product_search = trace_image_product_search
    trace_similar_styles = trace_image_product_search


class _KBService:
    def __init__(self, runtime_seed: str) -> None:
        self.runtime_seed = runtime_seed

    @property
    def formal_runtime_binding_sha256(self) -> str:
        return hashlib.sha256(f"{self.runtime_seed}:kb".encode()).hexdigest()

    def artifact_binding_for(self, kind):
        return ("fixture-binding", kind)

    def encyclopedia_lookup(self, *_args, **_kwargs):
        raise AssertionError("not used by Assistant runner contract tests")

    recipe_lookup = encyclopedia_lookup


@dataclass(frozen=True)
class _Artifact:
    runtime_binding: ModelRuntimeBinding


class _ModelService:
    def __init__(self, runtime_seed: str, artifact_kind: str) -> None:
        self.runtime_seed = runtime_seed
        self.artifact = _Artifact(
            ModelRuntimeBinding(
                artifact_kind=artifact_kind,
                model_id=f"fixture-{artifact_kind}",
                backend_name="fixture-backend",
                backend_version="1",
                manifest_sha256=hashlib.sha256(
                    f"{runtime_seed}:{artifact_kind}:manifest".encode()
                ).hexdigest(),
                artifacts=(),
            )
        )

    @property
    def formal_runtime_binding_sha256(self) -> str:
        return hashlib.sha256(
            f"{self.runtime_seed}:{self.artifact.runtime_binding.artifact_kind}".encode()
        ).hexdigest()

    def detect(self, *_args, **_kwargs):
        raise AssertionError("not used by Assistant runner contract tests")

    ocr = detect


class _SafetyResolver:
    def __init__(self, runtime_seed: str) -> None:
        self.runtime_seed = runtime_seed

    @property
    def formal_runtime_binding_sha256(self) -> str:
        return hashlib.sha256(f"{self.runtime_seed}:safety".encode()).hexdigest()

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("not used by Assistant runner contract tests")


def _registry(*, runtime_seed: str = "formal-runtime") -> ToolRegistry:
    registry = build_mvp_registry(
        MVPToolServices(
            product_search=_ProductService(runtime_seed),
            kb_lookup=_KBService(runtime_seed),
            object_detection=_ModelService(runtime_seed, "object_detector"),
            document_ocr=_ModelService(runtime_seed, "document_ocr"),
            safety_approval_for=_SafetyResolver(runtime_seed),
        )
    )
    return registry


def _phase4_query(index: int, *, split: str) -> Query:
    query_id = f"query-{index:03d}"
    text = f"Public query {index}"
    capability = "utility.document_reading"
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=TASK_SPEC_VERSION,
        query_id=query_id,
        asset_id=f"asset-public-{index}",
        image_path=f"images/public-{index}.png",
        leakage_group_id=f"leakage-{index}",
        boundary_group_id=None,
        template_family=f"template-{index}",
        generator_batch_id=f"generator-{index}",
        text=text,
        turns=[ConversationTurn(role="user", content=text)],
        canonical_intent="utility",
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=False,
        boundary_strategy=None,
        requires_card=False,
        split=split,
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="phase4-test",
                canonical_intent="utility",
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _phase4_inputs(
    tmp_path: Path,
    *,
    name: str,
    matrix_run_id: str,
    queries: tuple[Query, Query] | None = None,
):
    queries = queries or (
        _phase4_query(1, split="dev_mini"),
        _phase4_query(2, split="test_frozen"),
    )
    query_bytes = canonical_jsonl_bytes(
        tuple(item.model_dump(mode="json") for item in queries)
    )
    query_path = tmp_path / f"{name}-queries.jsonl"
    query_path.write_bytes(query_bytes)
    frozen_bytes = canonical_jsonl_bytes((queries[1].model_dump(mode="json"),))
    split_manifest = FrozenSplitManifest(
        profile="phase4-test",
        split_sizes={
            "dev_mini": 1,
            "opt_pool": 0,
            "val": 0,
            "test_frozen": 1,
        },
        count=1,
        sha256=sha256_bytes(frozen_bytes),
        assignment_sha256=sha256_bytes(query_bytes),
        seed=7,
        taxonomy_version=TAXONOMY_VERSION,
        full_plan_sha256="a" * 64,
        asset_catalog_sha256="b" * 64,
        leakage_policy_version="phase4-test-catalog-bound-v1",
        grouping_policy_version="query-connected-components-v1",
        group_fields=(
            "leakage_group_id",
            "boundary_group_id",
            "template_family",
            "generator_batch_id",
        ),
        component_count=2,
        leakage_violations=0,
        intent_counts={"utility": 1},
        capability_counts={"utility.document_reading": 1},
        boundary_count=0,
    )
    split_bytes = canonical_json_bytes(split_manifest.model_dump(mode="json"))
    split_path = tmp_path / f"{name}-split.json"
    split_path.write_bytes(split_bytes)
    rubric = _rubric()
    rubric_bytes = canonical_json_bytes(rubric.model_dump(mode="json"))
    rubric_path = tmp_path / f"{name}-rubric.json"
    rubric_path.write_bytes(rubric_bytes)
    evaluation_id = derive_blinded_evaluation_id(
        blinding_key=b"k" * 32,
        run_id=matrix_run_id,
        query_id=queries[0].query_id,
        config="full",
    )
    selection = make_phase4_input_selection_manifest(
        matrix_run_id=matrix_run_id,
        query_artifact_sha256=sha256_bytes(query_bytes),
        split_manifest_sha256=sha256_bytes(split_bytes),
        final_rubric_file_sha256=sha256_bytes(rubric_bytes),
        final_rubric_content_sha256=rubric.content_sha256,
        query_ids=tuple(item.query_id for item in queries),
        judge_audit_entries=(
            JudgeAuditSelectionEntry(
                query_id=queries[0].query_id,
                config="full",
                evaluation_id=evaluation_id,
            ),
        ),
    )
    selection_bytes = canonical_json_bytes(selection.model_dump(mode="json"))
    selection_path = tmp_path / f"{name}-selection.json"
    selection_path.write_bytes(selection_bytes)
    return load_verified_phase4_inputs(
        query_path=query_path,
        expected_query_sha256=sha256_bytes(query_bytes),
        split_manifest_path=split_path,
        expected_split_manifest_sha256=sha256_bytes(split_bytes),
        rubric_path=rubric_path,
        expected_rubric_file_sha256=sha256_bytes(rubric_bytes),
        selection_path=selection_path,
        expected_selection_file_sha256=sha256_bytes(selection_bytes),
    )


def _phase4_inputs_with_catalog(
    tmp_path: Path,
    *,
    name: str,
    matrix_run_id: str,
    cloud_upload_allowed: bool = False,
):
    asset_root = tmp_path / f"{name}-assets"
    drafts = []
    for index in (1, 2):
        relative = f"images/public-{index}.png"
        path = asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (32, 32))
        image.putdata(
            [
                (
                    (x * (17 + index) + y * 3) % 256,
                    (y * (29 + index) + x * 5) % 256,
                    ((x + y) * (11 + index)) % 256,
                )
                for y in range(32)
                for x in range(32)
            ]
        )
        image.save(path, format="PNG")
        drafts.append(
            DatasetAssetDraft(
                source_dataset="phase4-fixture",
                source_revision="phase4-fixture-v1",
                source_record_id=f"phase4:{index}",
                local_path=relative,
                license_id="CC0-1.0",
                source_url="https://example.org/phase4",
                attribution="phase4 test fixture",
                cloud_upload_allowed=cloud_upload_allowed,
                public_demo_allowed=False,
            )
        )
    assets = tuple(inventory_dataset_asset(item, asset_root) for item in drafts)
    catalog_root = tmp_path / f"{name}-catalog"
    publish_asset_catalog(assets, catalog_root, asset_root, coverage_roots=("images",))
    catalog = load_asset_catalog(catalog_root, asset_root, verify_files=True)
    queries = tuple(
        _phase4_query(index, split=split).model_copy(
            update={
                "asset_id": asset.asset_id,
                "image_path": asset.local_path,
                "leakage_group_id": catalog.component_for_asset(asset.asset_id),
            }
        )
        for index, split, asset in zip(
            (1, 2), ("dev_mini", "test_frozen"), assets, strict=True
        )
    )
    inputs = _phase4_inputs(
        tmp_path,
        name=name,
        matrix_run_id=matrix_run_id,
        queries=queries,
    )
    return inputs, catalog


def _verified_plan(
    tmp_path: Path,
    registry: ToolRegistry,
    *,
    name: str = "matrix",
    inputs=None,
    system_prompt_sha256: str = "a" * 64,
    endpoint: str = "https://example.org/qwen/v1",
    bank_sha256_by_config=None,
):
    matrix_run_id = f"{name}-run"
    backbone = make_backbone_lock(
        provider="qwen",
        model="qwen-plus-frozen",
        endpoint=endpoint,
        temperature=0.0,
        top_p=1.0,
        seed=7,
        system_prompt_sha256=system_prompt_sha256,
    )
    budget = make_inference_budget(
        max_input_tokens=1_000,
        max_output_tokens=200,
        max_tool_calls=3,
        max_turns=4,
        timeout_ms=30_000,
    )
    inputs = inputs or _phase4_inputs(tmp_path, name=name, matrix_run_id=matrix_run_id)
    plan = build_assistant_matrix_plan(
        inputs=inputs,
        backbone=backbone,
        budget=budget,
        registry=registry,
        bank_sha256_by_config=bank_sha256_by_config
        or {
            "llm_static": "1" * 64,
            "s1": "2" * 64,
            "s1s2": "3" * 64,
            "full": "4" * 64,
        },
        spec_baseline_bank_sha256="5" * 64,
    )
    created = create_assistant_matrix_plan_file(tmp_path / f"{name}.json", plan)
    verified = load_verified_assistant_matrix_plan(
        created.path,
        expected_file_sha256=created.file_sha256,
        registry=registry,
    )
    return verified


class _Backend:
    def __init__(
        self,
        *,
        fail_query: str | None = None,
        reported_model: str | None = None,
        include_trace: bool = False,
        trace_runtime_override: str | None = None,
        public_response: bool = False,
    ) -> None:
        self.fail_query = fail_query
        self.reported_model = reported_model
        self.include_trace = include_trace
        self.trace_runtime_override = trace_runtime_override
        self.public_response = public_response
        self.calls: list[str] = []

    def execute(self, request):
        self.calls.append(request.query.query_id)
        if request.query.query_id == self.fail_query:
            raise RuntimeError("private exception details must not enter evidence")
        skilled = request.config != "noskill"
        runtime_by_tool = {
            item.tool_name: item.runtime_binding_sha256
            for item in request.registry.tools
        }
        trace = (
            (
                AssistantToolTrace(
                    call_index=1,
                    tool_name="text_product_search",
                    status="success",
                    arguments_sha256="7" * 64,
                    result_sha256="8" * 64,
                    runtime_binding_sha256=(
                        self.trace_runtime_override
                        or runtime_by_tool["text_product_search"]
                    ),
                    latency_ms=3,
                ),
            )
            if self.include_trace
            else ()
        )
        return AssistantBackendResponse(
            request_sha256=request.request_sha256,
            backbone_provider=request.backbone.provider,
            backbone_model=self.reported_model or request.backbone.model,
            backbone_endpoint=request.backbone.endpoint,
            backbone_identity_sha256=request.backbone.identity_sha256,
            registry_sha256=request.registry.registry_sha256,
            registry_runtime_sha256=request.registry.registry_runtime_sha256,
            budget_sha256=request.budget.budget_sha256,
            response_text=(
                "A public answer."
                if self.public_response
                else f"public answer for {request.query.query_id}"
            ),
            tool_trace=trace,
            selected_capability="product.exact_match" if skilled else None,
            skill_slug=f"{request.config}-skill" if skilled else None,
            route_trace_sha256="6" * 64 if skilled else None,
            backbone_request_id=f"provider-{request.query_ordinal}",
            usage=LLMUsage(input_tokens=20, output_tokens=10),
            turn_count=2,
            latency_ms=25,
        )


def _runner_bank(
    registry: ToolRegistry,
    config_name: str,
    *,
    operators: tuple[str, ...] = (),
    description: str | None = None,
    body: str = "# Objective\n\nUse the frozen procedure.\n",
) -> StaticBankArtifact:
    if description is None:
        description = (
            "Frozen Stage-2 route description"
            if config_name in {"s1s2", "full"}
            else f"Frozen {config_name} skill"
        )
    skill_payload = {
        "slug": f"{config_name}-skill".replace("_", "-"),
        "version": 1,
        "description": description,
        "body": body,
        "static_refs": [],
        "operators": list(operators),
        "capability_id": "utility.document_reading",
        "parent_skill_sha256": None,
    }
    skill = StrictSkillArtifact.model_validate(
        {
            **skill_payload,
            "skill_sha256": sha256_bytes(canonical_json_bytes(skill_payload)),
        },
        strict=True,
    )
    payload = {
        "schema_version": 2,
        "baseline_kind": "llm_static",
        "construction_identity_sha256": hashlib.sha256(
            f"{config_name}:construction".encode()
        ).hexdigest(),
        "construction_identity_policy": "reviewed-draft-v1",
        "runtime_binding_policy": "registry-runtime-v1",
        "compiler": CompilerIdentity().model_dump(mode="json"),
        "tool_registry_sha256": registry.registry_sha256,
        "tool_registry_runtime_sha256": registry.registry_runtime_sha256,
        "skills": [skill.model_dump(mode="json")],
        "capability_map": [
            BankCapabilityBinding(
                capability_id=skill.capability_id, skill_slug=skill.slug
            ).model_dump(mode="json")
        ],
    }
    return StaticBankArtifact.model_validate(
        {**payload, "bank_sha256": sha256_bytes(canonical_json_bytes(payload))},
        strict=True,
    )


def test_production_runner_owns_model_usage_latency_and_receipt(
    tmp_path: Path, monkeypatch, canonical_registry_factory
) -> None:
    registry = canonical_registry_factory(name="assistant-runner-owned").registry
    banks = {
        name: _runner_bank(registry, name)
        for name in ("llm_static", "s1", "s1s2", "full")
    }
    system_prompt = "Frozen production Assistant prompt."
    inputs, catalog = _phase4_inputs_with_catalog(
        tmp_path,
        name="runner-owned",
        matrix_run_id="runner-owned-run",
        cloud_upload_allowed=True,
    )
    plan = _verified_plan(
        tmp_path,
        registry,
        name="runner-owned",
        system_prompt_sha256=sha256_bytes(system_prompt.encode()),
        endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
        bank_sha256_by_config={name: bank.bank_sha256 for name, bank in banks.items()},
        inputs=inputs,
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        prompt = messages[0]["content"]
        tool_calls = ()
        finish_reason = "stop"
        if "Route using descriptions only" in prompt:
            text = '{"selected_capability":"utility.document_reading"}'
        else:
            text = "Runner-owned public answer."
        return LLMResponse(
            provider="qwen",
            endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
            requested_model="qwen-plus-frozen",
            response_model="qwen-plus-frozen",
            request_id=f"provider-runner-{len(calls)}",
            text=text,
            tool_calls=tool_calls,
            usage=LLMUsage(input_tokens=31, output_tokens=11),
            finish_reason=finish_reason,
            latency_ms=7,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    runner = ProductionAssistantRunner(
        registry=registry,
        system_prompt=system_prompt,
        banks=banks,
        asset_catalog=catalog,
    )
    created = create_runner_owned_assistant_run_bundle(
        plan,
        config="noskill",
        runner=runner,
        registry=registry,
        output_dir=tmp_path / "runner-owned-run",
    )
    verified = load_verified_assistant_run_bundle(
        created.root,
        plan,
        expected_manifest_sha256=created.external_manifest_sha256,
        registry=registry,
    )
    assert verified.manifest.execution_provenance == "runner-owned-v1"
    assert len(calls) == len(plan.plan.queries)
    assert all(call[2]["images"] for call in calls)
    forbidden_public_markers = (
        "image_path",
        "query_images",
        "abo",
        "rpc",
        "fashioniq",
        "inaturalist",
        "isia",
        "wikimedia",
        "/",
        "\\",
    )
    for _provider, messages, _kwargs in calls:
        public = messages[1]["content"]
        assert all(marker not in public.casefold() for marker in forbidden_public_markers)
    for row in verified.rows:
        assert row.execution_provenance == "runner-owned-v1"
        assert row.execution_receipt is not None
        assert row.execution_receipt.aggregate_usage == LLMUsage(
            input_tokens=31, output_tokens=11
        )
        assert row.execution_receipt.model_calls[0].provider_request_id.startswith(
            "provider-runner-"
        )
        assert row.result.usage == row.execution_receipt.aggregate_usage
        assert row.result.latency_ms == row.execution_receipt.runner_latency_ms

    request = verified.requests[0]
    assert request.query.asset_binding is not None
    tampered_binding = request.query.asset_binding.model_copy(
        update={"image_path": "images/not-authoritative.png"}
    )
    tampered_request = request.model_copy(
        update={
            "query": request.query.model_copy(
                update={"asset_binding": tampered_binding}
            )
        }
    )
    with pytest.raises(AssistantBackendContractError, match="integrity validation"):
        runner.execute(tampered_request)

    noskill_call_count = len(calls)
    skilled_created = create_runner_owned_assistant_run_bundle(
        plan,
        config="s1",
        runner=runner,
        registry=registry,
        output_dir=tmp_path / "runner-owned-s1-run",
    )
    skilled = load_verified_assistant_run_bundle(
        skilled_created.root,
        plan,
        expected_manifest_sha256=skilled_created.external_manifest_sha256,
        registry=registry,
    )
    skilled_calls = calls[noskill_call_count:]
    assert len(skilled_calls) == 2 * len(plan.plan.queries)
    for route_call, action_call in zip(
        skilled_calls[::2], skilled_calls[1::2], strict=True
    ):
        assert "Use the frozen procedure" not in route_call[1][0]["content"]
        assert "Use the frozen procedure" in action_call[1][0]["content"]
    for row in skilled.rows:
        assert row.result.selected_capability == "utility.document_reading"
        assert row.result.skill_slug == "s1-skill"
        assert row.result.route_trace_sha256 is not None
        assert row.execution_receipt is not None
        assert len(row.execution_receipt.model_calls) == 2
        assert row.execution_receipt.aggregate_usage == LLMUsage(
            input_tokens=62, output_tokens=22
        )


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("s2_body", "S2 may only change Description"),
        ("s3_description", "S3 may only change Body"),
    ],
)
def test_production_runner_rejects_stage_boundary_drift(
    tmp_path: Path,
    canonical_registry_factory,
    mutation: str,
    expected_message: str,
) -> None:
    registry = canonical_registry_factory(name=f"stage-boundary-{mutation}").registry
    banks = {
        name: _runner_bank(registry, name)
        for name in ("llm_static", "s1", "s1s2", "full")
    }
    if mutation == "s2_body":
        changed_body = "# Objective\n\nS2 illegally changed the execution body.\n"
        banks["s1s2"] = _runner_bank(
            registry,
            "s1s2",
            body=changed_body,
        )
        banks["full"] = _runner_bank(
            registry,
            "full",
            body=changed_body,
        )
    else:
        banks["full"] = _runner_bank(
            registry,
            "full",
            description="S3 illegally changed the route description",
        )
    _, catalog = _phase4_inputs_with_catalog(
        tmp_path,
        name=f"stage-boundary-{mutation}",
        matrix_run_id=f"stage-boundary-{mutation}",
        cloud_upload_allowed=True,
    )

    with pytest.raises(ValueError, match=expected_message):
        ProductionAssistantRunner(
            registry=registry,
            system_prompt="Frozen stage-boundary prompt.",
            banks=banks,
            asset_catalog=catalog,
        )


def test_v1_matrix_registry_lock_keeps_legacy_wire_shape(tmp_path: Path) -> None:
    registry = _registry(runtime_seed="assistant-v1-wire")
    plan = _verified_plan(tmp_path, registry, name="assistant-v1-wire")
    dumped = plan.plan.registry.model_dump(mode="json")
    tools = [
        {
            "tool_name": name,
            "tool_spec_sha256": next(
                spec.spec_sha256 for spec in registry.specs() if spec.name == name
            ),
            "runtime_binding_sha256": registry.runtime_binding_sha256(name),
        }
        for name in sorted(MVP_TOOL_NAMES)
    ]
    unsigned = {
        "registry_sha256": registry.registry_sha256,
        "registry_runtime_sha256": registry.registry_runtime_sha256,
        "tools": tools,
    }

    assert dumped == {
        **unsigned,
        "lock_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }


def test_v2_production_runner_locks_and_executes_composite_tool(
    tmp_path: Path, monkeypatch, canonical_registry_factory
) -> None:
    formal = canonical_registry_factory(
        name="assistant-v2-composite-registry",
        include_multi_product=True,
    )
    registry = formal.registry
    banks = {
        name: _runner_bank(
            registry,
            name,
            operators=("multi_product_search",),
        )
        for name in ("llm_static", "s1", "s1s2", "full")
    }
    system_prompt = "Frozen composite-tool Assistant prompt."
    inputs, catalog = _phase4_inputs_with_catalog(
        tmp_path,
        name="assistant-v2-composite",
        matrix_run_id="assistant-v2-composite-run",
        cloud_upload_allowed=True,
    )
    plan = _verified_plan(
        tmp_path,
        registry,
        name="assistant-v2-composite",
        system_prompt_sha256=sha256_bytes(system_prompt.encode()),
        endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
        bank_sha256_by_config={name: bank.bank_sha256 for name, bank in banks.items()},
        inputs=inputs,
    )

    assert isinstance(plan.plan.registry, RegistryRuntimeLockV2)
    assert plan.plan.registry.schema_version == 2
    assert plan.plan.registry.policy_version == "assistant-registry-runtime-v2"
    assert tuple(item.tool_name for item in plan.plan.registry.tools) == tuple(
        sorted(MVP_TOOL_NAMES_V2)
    )
    locked_composite = next(
        item
        for item in plan.plan.registry.tools
        if item.tool_name == "multi_product_search"
    )
    live_composite = next(
        item for item in registry.specs() if item.name == "multi_product_search"
    )
    assert locked_composite.tool_spec_sha256 == live_composite.spec_sha256
    assert locked_composite.runtime_binding_sha256 == (
        registry.runtime_binding_sha256("multi_product_search")
    )
    incomplete_lock = plan.plan.registry.model_dump(mode="python")
    incomplete_lock["tools"] = tuple(
        item
        for item in incomplete_lock["tools"]
        if item["tool_name"] != "multi_product_search"
    )
    incomplete_unsigned = plan.plan.registry.model_dump(
        mode="json", exclude={"lock_sha256"}
    )
    incomplete_unsigned["tools"] = [
        item
        for item in incomplete_unsigned["tools"]
        if item["tool_name"] != "multi_product_search"
    ]
    incomplete_lock["lock_sha256"] = sha256_bytes(
        canonical_json_bytes(incomplete_unsigned)
    )
    with pytest.raises(ValidationError, match="exactly eight sorted tools"):
        RegistryRuntimeLockV2.model_validate(incomplete_lock, strict=True)

    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        prompt = messages[0]["content"]
        tool_calls = ()
        finish_reason = "stop"
        if "Route using descriptions only" in prompt:
            text = '{"selected_capability":"utility.document_reading"}'
        elif len(messages) == 2:
            public = json.loads(messages[1]["content"])
            text = ""
            tool_calls = (
                LLMToolCall(
                    call_id=f"composite-{len(calls)}",
                    name="multi_product_search",
                    arguments_json=json.dumps(
                        {"asset_id": public["asset_id"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
            finish_reason = "tool_calls"
        else:
            text = (
                "answer:\nComposite product result presented.\n"
                "item_mapping:\nnone\n"
                "product_cards:\nnone\n"
                "uncertainty:\nNo supported item was returned."
            )
        return LLMResponse(
            provider="qwen",
            endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
            requested_model="qwen-plus-frozen",
            response_model="qwen-plus-frozen",
            request_id=f"provider-composite-{len(calls)}",
            text=text,
            tool_calls=tool_calls,
            usage=LLMUsage(input_tokens=23, output_tokens=11),
            finish_reason=finish_reason,
            latency_ms=7,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    runner = ProductionAssistantRunner(
        registry=registry,
        system_prompt=system_prompt,
        banks=banks,
        asset_catalog=catalog,
    )
    created = create_runner_owned_assistant_run_bundle(
        plan,
        config="s1",
        runner=runner,
        registry=registry,
        output_dir=tmp_path / "assistant-v2-composite-run",
    )
    verified = load_verified_assistant_run_bundle(
        created.root,
        plan,
        expected_manifest_sha256=created.external_manifest_sha256,
        registry=registry,
    )

    assert len(calls) == 3 * len(plan.plan.queries)
    for request, row in zip(verified.requests, verified.rows, strict=True):
        assert isinstance(request.registry, RegistryRuntimeLockV2)
        assert row.result.error_code is None
        assert len(row.result.tool_trace) == 1
        trace = row.result.tool_trace[0]
        assert trace.tool_name == "multi_product_search"
        assert trace.status == "success"
        assert trace.runtime_binding_sha256 == locked_composite.runtime_binding_sha256
        assert row.execution_receipt is not None
        assert row.execution_receipt.tool_trace == row.result.tool_trace

    visible = VisibleToolEvidence(
        tool_name="multi_product_search",
        status="success",
        visible_text="Composite product result presented.",
    )
    public_result_payload = verified.rows[0].result.model_dump(mode="python")
    public_result_payload["visible_tool_evidence"] = (visible,)
    public_result = type(verified.rows[0].result).model_validate(
        public_result_payload, strict=True
    )
    packet = build_final_evaluation_packet(
        inputs.selected_queries[0],
        public_result,
        asset_catalog=catalog,
        rubric=inputs.rubric,
        blinding_key=b"k" * 32,
    )
    assert packet.tool_evidence == (visible,)


def test_production_runner_enforces_selected_skill_operators_but_not_noskill(
    tmp_path: Path, monkeypatch, canonical_registry_factory
) -> None:
    registry = canonical_registry_factory(name="assistant-operator-gate").registry
    banks = {
        name: _runner_bank(
            registry,
            name,
            operators=("object_detect",),
        )
        for name in ("llm_static", "s1", "s1s2", "full")
    }
    system_prompt = "Frozen operator-gate prompt."
    inputs, catalog = _phase4_inputs_with_catalog(
        tmp_path,
        name="operator-gate",
        matrix_run_id="operator-gate-run",
        cloud_upload_allowed=True,
    )
    plan = _verified_plan(
        tmp_path,
        registry,
        name="operator-gate",
        system_prompt_sha256=sha256_bytes(system_prompt.encode()),
        endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
        bank_sha256_by_config={name: bank.bank_sha256 for name, bank in banks.items()},
        inputs=inputs,
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        prompt = messages[0]["content"]
        tool_calls = ()
        finish_reason = "stop"
        if "Route using descriptions only" in prompt:
            text = '{"selected_capability":"utility.document_reading"}'
        elif len(messages) == 2:
            text = ""
            tool_calls = (
                LLMToolCall(
                    call_id=f"operator-gate-{len(calls)}",
                    name="encyclopedia_lookup",
                    arguments_json='{"entity":"dress"}',
                ),
            )
            finish_reason = "tool_calls"
        else:
            text = (
                "answer:\nnot enough evidence\n"
                "evidence:\nnot enough evidence\n"
                "uncertainty:\nOperator-gate answer."
            )
        return LLMResponse(
            provider="qwen",
            endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
            requested_model="qwen-plus-frozen",
            response_model="qwen-plus-frozen",
            request_id=f"provider-operator-gate-{len(calls)}",
            text=text,
            tool_calls=tool_calls,
            usage=LLMUsage(input_tokens=17, output_tokens=9),
            finish_reason=finish_reason,
            latency_ms=5,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    runner = ProductionAssistantRunner(
        registry=registry,
        system_prompt=system_prompt,
        banks=banks,
        asset_catalog=catalog,
    )

    noskill_created = create_runner_owned_assistant_run_bundle(
        plan,
        config="noskill",
        runner=runner,
        registry=registry,
        output_dir=tmp_path / "operator-gate-noskill",
    )
    noskill = load_verified_assistant_run_bundle(
        noskill_created.root,
        plan,
        expected_manifest_sha256=noskill_created.external_manifest_sha256,
        registry=registry,
    )
    for row in noskill.rows:
        assert row.result.error_code is None
        assert len(row.result.tool_trace) == 1
        assert row.result.tool_trace[0].tool_name == "encyclopedia_lookup"
        assert row.result.tool_trace[0].status == "success"

    skilled_created = create_runner_owned_assistant_run_bundle(
        plan,
        config="s1",
        runner=runner,
        registry=registry,
        output_dir=tmp_path / "operator-gate-skilled",
    )
    skilled = load_verified_assistant_run_bundle(
        skilled_created.root,
        plan,
        expected_manifest_sha256=skilled_created.external_manifest_sha256,
        registry=registry,
    )
    for row in skilled.rows:
        assert row.result.error_code == "response_contract_error"
        assert len(row.result.tool_trace) == 1
        assert row.result.tool_trace[0].tool_name == "encyclopedia_lookup"
        assert row.result.tool_trace[0].status == "error"
        assert row.result.tool_trace[0].error_code == "permission_denied"

def test_runner_owned_entry_rejects_arbitrary_backend(tmp_path: Path) -> None:
    registry = _registry(runtime_seed="runner-type")
    plan = _verified_plan(tmp_path, registry, name="runner-type")
    with pytest.raises(TypeError, match="exactly ProductionAssistantRunner"):
        create_runner_owned_assistant_run_bundle(
            plan,
            config="noskill",
            runner=_Backend(),
            registry=registry,
            output_dir=tmp_path / "must-not-exist",
        )


def test_matrix_has_exact_five_configs_and_no_per_config_runtime_fields(
    tmp_path: Path,
) -> None:
    registry = _registry()
    verified = _verified_plan(tmp_path, registry)
    plan = verified.plan

    assert tuple(item.config for item in plan.treatments) == MAIN_CONFIG_ORDER
    assert plan.diagnostics[0].name == "spec_baseline"
    assert plan.diagnostics[0].formal_eligible is False
    assert plan.formal_eligible is False
    assert plan.formal_ineligible_reason == FORMAL_INELIGIBLE_REASON
    for treatment in plan.treatments:
        assert not {
            "backbone",
            "budget",
            "registry",
            "queries",
            "query_order",
        } & set(type(treatment).model_fields)

    contaminated = plan.treatments[0].model_dump(mode="json")
    contaminated["backbone"] = {"model": "different"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MatrixTreatment.model_validate(contaminated)

    output = tmp_path / "diagnostic-must-not-run"
    with pytest.raises(AssistantRunError, match="diagnostic-only"):
        create_assistant_run_bundle(
            verified,
            config="spec_baseline",
            backend=_Backend(),
            registry=registry,
            output_dir=output,
        )
    assert not output.exists()


def test_matrix_rejects_raw_inputs_and_uses_fixed_public_projection(
    tmp_path: Path,
) -> None:
    registry = _registry()
    verified = _verified_plan(tmp_path, registry, name="public-projection")
    for query in verified.plan.queries:
        public = json.loads(query.public_input_json)
        assert set(public) == {"asset_id", "text", "turns"}
        assert public["asset_id"].startswith("asset-token-")
        assert query.asset_binding is not None
        assert public["asset_id"] == query.asset_binding.asset_token
        assert "image_path" not in query.public_input_json
        assert all(
            marker not in query.public_input_json.casefold()
            for marker in (
                "query_images",
                "abo",
                "rpc",
                "fashioniq",
                "inaturalist",
                "isia",
                "wikimedia",
                "/",
                "\\",
            )
        )
        assert not {
            "canonical_capability",
            "ground_truth",
            "label_provenance",
            "split",
        } & set(public)

    original = _phase4_query(99, split="dev_mini")
    relocated = original.model_copy(update={"image_path": "relocated/asset.png"})
    original_public = build_assistant_query_input(original)
    relocated_public = build_assistant_query_input(relocated)
    assert original_public.public_input_sha256 == relocated_public.public_input_sha256
    assert original_public.public_input_json == relocated_public.public_input_json
    assert original_public.asset_binding is not None
    assert relocated_public.asset_binding is not None
    assert original_public.asset_binding.image_path != relocated_public.asset_binding.image_path

    legacy_public = canonical_json_bytes(
        {
            "asset_id": "legacy-catalog-asset",
            "image_path": "images/legacy.png",
            "text": "Legacy fixture request",
            "turns": [{"role": "user", "content": "Legacy fixture request"}],
        }
    ).decode()
    legacy_unsigned = {
        "query_id": "legacy-query",
        "public_input_json": legacy_public,
        "public_input_sha256": sha256_bytes(legacy_public.encode()),
    }
    legacy = AssistantQueryInput.model_validate(
        {
            **legacy_unsigned,
            "query_sha256": sha256_bytes(canonical_json_bytes(legacy_unsigned)),
        },
        strict=True,
    )
    assert "asset_binding" not in legacy.model_dump(mode="json")

    with pytest.raises(TypeError, match="external-digest typed loader"):
        build_assistant_matrix_plan(
            inputs=object(),  # type: ignore[arg-type]
            backbone=make_backbone_lock(
                provider="qwen",
                model="qwen-plus",
                endpoint="https://example.org/v1",
                temperature=0.0,
                top_p=1.0,
                seed=1,
                system_prompt_sha256="c" * 64,
            ),
            budget=make_inference_budget(
                max_input_tokens=10,
                max_output_tokens=10,
                max_tool_calls=0,
                max_turns=1,
                timeout_ms=100,
            ),
            registry=registry,
            bank_sha256_by_config={
                "llm_static": "1" * 64,
                "s1": "2" * 64,
                "s1s2": "3" * 64,
                "full": "4" * 64,
            },
            spec_baseline_bank_sha256="5" * 64,
        )


def test_plan_loader_requires_external_digest_and_same_live_registry(
    tmp_path: Path,
) -> None:
    registry = _registry()
    verified = _verified_plan(tmp_path, registry, name="plan-lock")

    with pytest.raises(AssistantRunError, match="external digest"):
        load_verified_assistant_matrix_plan(
            verified.path,
            expected_file_sha256="0" * 64,
            registry=registry,
        )
    with pytest.raises(AssistantRunError, match="live tool registry"):
        load_verified_assistant_matrix_plan(
            verified.path,
            expected_file_sha256=verified.expected_file_sha256,
            registry=_registry(runtime_seed="different-runtime"),
        )


def test_phase4_loader_rejects_coordinated_selection_rehash(
    tmp_path: Path,
) -> None:
    matrix_run_id = "selection-attack-run"
    inputs = _phase4_inputs(
        tmp_path,
        name="selection-attack",
        matrix_run_id=matrix_run_id,
    )
    payload = json.loads(inputs.selection_path.read_text("utf-8"))
    payload["final_rubric_content_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in payload.items() if key != "selection_sha256"
    }
    payload["selection_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    attacker_bytes = canonical_json_bytes(payload)
    inputs.selection_path.write_bytes(attacker_bytes)

    with pytest.raises(AssistantRunError, match="does not bind.*rubric"):
        load_verified_phase4_inputs(
            query_path=inputs.query_path,
            expected_query_sha256=inputs.expected_query_sha256,
            split_manifest_path=inputs.split_manifest_path,
            expected_split_manifest_sha256=inputs.expected_split_manifest_sha256,
            rubric_path=inputs.rubric_path,
            expected_rubric_file_sha256=inputs.expected_rubric_file_sha256,
            selection_path=inputs.selection_path,
            expected_selection_file_sha256=sha256_bytes(attacker_bytes),
        )


def test_phase4_handle_rejects_disk_toctou_before_plan_build(tmp_path: Path) -> None:
    registry = _registry()
    inputs = _phase4_inputs(
        tmp_path,
        name="input-toctou",
        matrix_run_id="input-toctou-run",
    )
    inputs.rubric_path.write_bytes(inputs.rubric_path.read_bytes() + b" ")

    with pytest.raises(AssistantRunError, match="changed on disk"):
        build_assistant_matrix_plan(
            inputs=inputs,
            backbone=make_backbone_lock(
                provider="qwen",
                model="qwen-plus",
                endpoint="https://example.org/v1",
                temperature=0.0,
                top_p=1.0,
                seed=1,
                system_prompt_sha256="c" * 64,
            ),
            budget=make_inference_budget(
                max_input_tokens=10,
                max_output_tokens=10,
                max_tool_calls=0,
                max_turns=1,
                timeout_ms=100,
            ),
            registry=registry,
            bank_sha256_by_config={
                "llm_static": "1" * 64,
                "s1": "2" * 64,
                "s1s2": "3" * 64,
                "full": "4" * 64,
            },
            spec_baseline_bank_sha256="5" * 64,
        )


def test_safe_final_packet_factory_owns_query_result_and_rubric_join(
    tmp_path: Path,
) -> None:
    registry = _registry()
    name = "safe-packet"
    inputs, catalog = _phase4_inputs_with_catalog(
        tmp_path,
        name=name,
        matrix_run_id=f"{name}-run",
        cloud_upload_allowed=True,
    )
    plan = _verified_plan(tmp_path, registry, name=name, inputs=inputs)
    five_runs = _five_runs(
        tmp_path,
        registry,
        plan,
        public_response=True,
    )
    packet = build_verified_final_evaluation_packet(
        inputs,
        plan,
        five_runs,
        config="noskill",
        query_ordinal=0,
        registry=registry,
        asset_catalog=catalog,
        blinding_key=b"k" * 32,
    )
    assert packet.response_text == "A public answer."
    assert packet.turns == tuple(inputs.selected_queries[0].turns)
    assert packet.rubric == inputs.rubric

    with pytest.raises(TypeError, match="external-digest typed loader"):
        build_verified_final_evaluation_packet(
            inputs.selected_queries[0],  # type: ignore[arg-type]
            plan,
            five_runs,
            config="noskill",
            query_ordinal=0,
            registry=registry,
            asset_catalog=catalog,
            blinding_key=b"k" * 32,
        )


def test_bundle_retains_error_rows_and_requires_external_digest(tmp_path: Path) -> None:
    registry = _registry()
    plan = _verified_plan(tmp_path, registry)
    backend = _Backend(fail_query="query-002", include_trace=True)
    output = tmp_path / "s1-run"
    created = create_assistant_run_bundle(
        plan,
        config="s1",
        backend=backend,
        registry=registry,
        output_dir=output,
    )

    assert backend.calls == ["query-001", "query-002"]
    with pytest.raises(AssistantRunError, match="external manifest digest"):
        load_verified_assistant_run_bundle(
            output,
            plan,
            expected_manifest_sha256="0" * 64,
            registry=registry,
        )
    verified = load_verified_assistant_run_bundle(
        output,
        plan,
        expected_manifest_sha256=created.external_manifest_sha256,
        registry=registry,
    )
    assert verified.manifest.query_count == 2
    assert (verified.manifest.success_count, verified.manifest.error_count) == (1, 1)
    assert [row.result.query_id for row in verified.rows] == [
        "query-001",
        "query-002",
    ]
    assert verified.rows[1].result.error_code == "runtime_error"
    assert verified.rows[1].result.response_text == ""
    assert verified.manifest.formal_eligible is False

    calls_before = list(backend.calls)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        create_assistant_run_bundle(
            plan,
            config="s1",
            backend=backend,
            registry=registry,
            output_dir=output,
        )
    assert backend.calls == calls_before


def test_backend_cannot_vary_backbone_between_configs(tmp_path: Path) -> None:
    registry = _registry()
    plan = _verified_plan(tmp_path, registry)
    output = tmp_path / "identity-violation"

    with pytest.raises(AssistantBackendContractError, match="runtime identity"):
        create_assistant_run_bundle(
            plan,
            config="full",
            backend=_Backend(reported_model="attacker-model"),
            registry=registry,
            output_dir=output,
        )
    assert not output.exists()


def test_backend_tool_trace_must_match_live_registry_runtime(tmp_path: Path) -> None:
    registry = _registry()
    plan = _verified_plan(tmp_path, registry, name="trace-runtime")
    output = tmp_path / "trace-runtime-violation"

    with pytest.raises(AssistantBackendContractError, match="tool trace runtime"):
        create_assistant_run_bundle(
            plan,
            config="noskill",
            backend=_Backend(include_trace=True, trace_runtime_override="0" * 64),
            registry=registry,
            output_dir=output,
        )
    assert not output.exists()


def _coordinated_rehash(
    root: Path,
    *,
    duplicate_second_query: bool = False,
    drop_last_row: bool = False,
) -> str:
    result_path = root / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text("utf-8").splitlines()]
    if duplicate_second_query:
        rows[1]["result"]["query_id"] = rows[0]["result"]["query_id"]
        unsigned_row = {
            key: value for key, value in rows[1].items() if key != "row_sha256"
        }
        rows[1]["row_sha256"] = sha256_bytes(canonical_json_bytes(unsigned_row))
    if drop_last_row:
        rows.pop()
    results_bytes = canonical_jsonl_bytes(tuple(rows))
    result_path.write_bytes(results_bytes)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["artifacts"]["results.jsonl"].update(
        {
            "bytes": len(results_bytes),
            "rows": len(rows),
            "sha256": sha256_bytes(results_bytes),
        }
    )
    if drop_last_row:
        manifest["query_count"] = len(rows)
        manifest["success_count"] = sum(
            row["result"]["error_code"] is None for row in rows
        )
        manifest["error_count"] = len(rows) - manifest["success_count"]
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(unsigned_manifest))
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return sha256_bytes(manifest_bytes)


@pytest.mark.parametrize(
    ("duplicate", "missing", "message"),
    [
        (True, False, "identity differs|coverage/order"),
        (False, True, "differs from frozen matrix plan|exactly one"),
    ],
)
def test_coordinated_rehash_cannot_change_query_coverage(
    tmp_path: Path,
    duplicate: bool,
    missing: bool,
    message: str,
) -> None:
    registry = _registry()
    plan = _verified_plan(tmp_path, registry, name=f"rehash-{duplicate}-{missing}")
    output = tmp_path / f"bundle-{duplicate}-{missing}"
    create_assistant_run_bundle(
        plan,
        config="noskill",
        backend=_Backend(),
        registry=registry,
        output_dir=output,
    )
    attacker_supplied_digest = _coordinated_rehash(
        output,
        duplicate_second_query=duplicate,
        drop_last_row=missing,
    )

    with pytest.raises(AssistantRunError, match=message):
        load_verified_assistant_run_bundle(
            output,
            plan,
            expected_manifest_sha256=attacker_supplied_digest,
            registry=registry,
        )


def _packet_hash(payload: dict) -> str:
    def jsonable(value):
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    return sha256_bytes(canonical_json_bytes(jsonable(payload)))


def _rubric() -> RubricSnapshot:
    content = "Frozen task completion and answer quality rubric."
    return RubricSnapshot(
        rubric_id="rubric-v1",
        rubric_version="1",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def _final_packet() -> FinalEvaluationPacket:
    image_bytes = b"public-image"
    payload = {
        "schema_version": 2,
        "packet_kind": "final",
        "cache_namespace": "final-evaluator-v2",
        "evaluation_id": "d" * 64,
        "turns": (ConversationTurn(role="user", content="What is shown?"),),
        "image": EvaluationImage(
            mime_type="image/jpeg",
            sha256=hashlib.sha256(image_bytes).hexdigest(),
        ),
        "response_text": "A public answer.",
        "cards": (
            VisibleCard(
                title="Visible product card",
                body="Public card body, not a Skill Body.",
                fields=(("color", "blue"),),
            ),
        ),
        "tool_evidence": (),
        "card_requirement": "not_applicable",
        "rubric": _rubric(),
    }
    return FinalEvaluationPacket.model_validate(
        {**payload, "packet_sha256": _packet_hash(payload)}, strict=True
    )


def _feedback_packet() -> FeedbackPacket:
    image_bytes = b"public-image"
    payload = {
        "schema_version": 2,
        "packet_kind": "feedback",
        "cache_namespace": "feedback-evaluator-v2",
        "query_id": "query-001",
        "turns": (ConversationTurn(role="user", content="What is shown?"),),
        "image": EvaluationImage(
            mime_type="image/jpeg",
            sha256=hashlib.sha256(image_bytes).hexdigest(),
        ),
        "canonical_capability": "product.exact_match",
        "acceptable_capabilities": ("product.exact_match",),
        "response_text": "A public answer.",
        "cards": (),
        "tool_evidence": (),
        "tool_trace": (),
        "rubric": _rubric(),
    }
    return FeedbackPacket.model_validate(
        {**payload, "packet_sha256": _packet_hash(payload)}, strict=True
    )


def _evaluator_lock():
    feedback = make_feedback_evaluator_identity(
        provider="gemini",
        model="gemini-3.6-flash",
        model_family="gemini",
        endpoint="https://example.org/aifast/v1",
    )
    final = make_final_evaluator_identity(
        provider="kimi",
        model="kimi-k2.6",
        model_family="kimi",
        endpoint="https://example.org/kimi/v1",
    )
    return feedback, final, make_evaluator_isolation_lock(feedback, final)


def test_final_and_feedback_serializers_and_prompts_are_structurally_separate() -> None:
    final_packet = _final_packet()
    feedback_packet = _feedback_packet()
    _, _, lock = _evaluator_lock()

    final_bytes = serialize_final_evaluation_packet(final_packet)
    feedback_bytes = serialize_feedback_packet(feedback_packet)
    assert b'"packet_kind":"final"' in final_bytes
    assert b'"packet_kind":"feedback"' in feedback_bytes
    for secret in (
        b'"config"',
        b'"bank_sha256"',
        b'"skill_slug"',
        b'"ground_truth"',
        b'"split"',
        b'"judge_output"',
    ):
        assert secret not in final_bytes

    contaminated = final_packet.model_dump(mode="json")
    contaminated["config"] = "full"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FinalEvaluationPacket.model_validate(contaminated)
    with pytest.raises(TypeError, match="FinalEvaluationPacket"):
        serialize_final_evaluation_packet(feedback_packet)  # type: ignore[arg-type]

    feedback_prompt = build_bound_feedback_prompt(feedback_packet, lock)
    final_prompt = build_bound_final_prompt(final_packet, lock)
    assert feedback_prompt.cache_namespace != final_prompt.cache_namespace
    assert feedback_prompt.evaluator.role == "development-feedback"
    assert final_prompt.evaluator.role == "blinded-final-judge"
    assert feedback_prompt.evaluator.identity_sha256 != (
        final_prompt.evaluator.identity_sha256
    )
    assert feedback_prompt.evaluator_isolation_sha256 == lock.lock_sha256
    assert final_prompt.evaluator_isolation_sha256 == lock.lock_sha256
    assert feedback_prompt.formal_eligible is False
    assert final_prompt.formal_eligible is False
    assert lock.formal_eligible is False
    assert lock.shared_model_runtime is False
    assert (
        lock.independence_limitation
        == "cross-provider-isolation-with-third-party-gateway-identity-risk"
    )
    final_visible = final_prompt.messages[1].content.encode()
    for secret in (
        b'"config"',
        b'"bank_sha256"',
        b'"canonical_capability"',
        b'"split"',
        b'"judge_output"',
    ):
        assert secret not in final_visible


def test_packet_and_evaluator_model_copy_cannot_bypass_self_hashes() -> None:
    packet = _final_packet()
    stale_packet = packet.model_copy(update={"response_text": "config=full"})
    with pytest.raises(ValidationError, match="self hash"):
        serialize_final_evaluation_packet(stale_packet)

    _, _, lock = _evaluator_lock()
    stale_final = lock.final.model_copy(update={"model": "changed-model"})
    stale_lock = lock.model_copy(update={"final": stale_final})
    with pytest.raises(ValidationError, match="identity hash|lock self hash"):
        build_bound_final_prompt(packet, stale_lock)


def test_validly_rehashed_final_packet_rejects_secret_text_and_card_field() -> None:
    packet = _final_packet()
    contaminated = packet.model_copy(update={"response_text": "config=full"})
    contaminated = contaminated.model_copy(
        update={
            "packet_sha256": _packet_hash(
                contaminated.model_dump(mode="json", exclude={"packet_sha256"})
            )
        }
    )
    with pytest.raises(EvaluatorIsolationError, match="secret text"):
        serialize_final_evaluation_packet(contaminated)

    contaminated = packet.model_copy(
        update={
            "cards": (
                VisibleCard(
                    title="Visible",
                    body="Public",
                    fields=(("config", "full"),),
                ),
            )
        }
    )
    contaminated = contaminated.model_copy(
        update={
            "packet_sha256": _packet_hash(
                contaminated.model_dump(mode="json", exclude={"packet_sha256"})
            )
        }
    )
    with pytest.raises(EvaluatorIsolationError, match="secret text"):
        serialize_final_evaluation_packet(contaminated)


def test_validly_rehashed_final_packet_allows_treatment_words_in_product_title() -> None:
    packet = _final_packet()
    product_title = "find. EYE MASK FULL TREATMENT"
    public = packet.model_copy(
        update={
            "response_text": f"I found {product_title}.",
            "cards": (
                VisibleCard(
                    title=product_title,
                    body="Public catalog title.",
                    fields=(("color", "blue"),),
                ),
            ),
        }
    )
    public = public.model_copy(
        update={
            "packet_sha256": _packet_hash(
                public.model_dump(mode="json", exclude={"packet_sha256"})
            )
        }
    )

    serialized = serialize_final_evaluation_packet(public)
    _, _, lock = _evaluator_lock()
    prompt = build_bound_final_prompt(public, lock)

    assert product_title.encode("utf-8") in serialized
    assert product_title in prompt.messages[1].content


def test_blinded_ids_are_unique_across_config_and_failed_judges_are_zero() -> None:
    ids = {
        derive_blinded_evaluation_id(
            blinding_key=b"k" * 32,
            run_id="run-1",
            query_id="query-1",
            config=config,
        )
        for config in MAIN_CONFIG_ORDER
    }
    assert len(ids) == len(MAIN_CONFIG_ORDER)

    evaluation_id = next(iter(ids))
    scores = JudgeScores(
        evaluation_id=evaluation_id,
        requires_card=False,
        dimensions=(
            JudgeDimensionScore(dimension="CA", score=8, tier="Good"),
            JudgeDimensionScore(dimension="CQ", score=16, tier="Good"),
            JudgeDimensionScore(dimension="TCR", score=8, tier="Good"),
        ),
        j_project=80.0,
    )
    with pytest.raises(ValidationError, match="conservative zero"):
        JudgeOutcome(
            evaluation_id=evaluation_id,
            status="provider_error",
            attempts=1,
            max_attempts=1,
            scores=scores,
            judge_provider="provider",
            judge_model="model",
            prompt_sha256="a" * 64,
            error_code="provider_error",
        )

    with pytest.raises(ValueError, match="at least one query"):
        validate_paired_result_rows({config: [] for config in MAIN_CONFIG_ORDER})


def test_evaluator_lock_requires_different_provider_and_model_family() -> None:
    feedback = make_feedback_evaluator_identity(
        provider="gemini",
        model="gemini-3.6-flash",
        model_family="gemini",
        endpoint="https://example.org/gemini",
    )
    final = make_final_evaluator_identity(
        provider="kimi",
        model="kimi-k2.6",
        model_family="kimi",
        endpoint="https://example.org/a",
    )
    lock = make_evaluator_isolation_lock(feedback, final)
    assert lock.shared_model_runtime is False
    drifted = make_final_evaluator_identity(
        provider="gemini",
        model="gemini-3.6-flash",
        model_family="gemini",
        endpoint="https://example.org/gemini",
    )
    with pytest.raises(ValidationError, match="different provider runtimes"):
        make_evaluator_isolation_lock(feedback, drifted)


def _five_runs(
    tmp_path: Path,
    registry: ToolRegistry,
    plan,
    *,
    public_response: bool = False,
):
    locks = {}
    for config in MAIN_CONFIG_ORDER:
        created = create_assistant_run_bundle(
            plan,
            config=config,
            backend=_Backend(public_response=public_response),
            registry=registry,
            output_dir=tmp_path / f"run-{config}",
        )
        locks[config] = (created.root, created.external_manifest_sha256)
    return load_verified_five_config_runs(
        plan,
        bundle_locks=locks,
        registry=registry,
    )


class _ReverseDecryptor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, ciphertext: bytes, *, seal_scheme: str, audit_id: str):
        self.calls += 1
        assert seal_scheme == "test-external-reverse-v1"
        assert audit_id == "audit-001"
        return ciphertext[::-1]


def test_judge_audit_is_unreadable_until_all_five_bundles_are_verified(
    tmp_path: Path,
) -> None:
    registry = _registry()
    plan = _verified_plan(tmp_path, registry, name="audit-matrix")
    _, _, evaluator_lock = _evaluator_lock()
    row = JudgeAuditReferenceRow(
        evaluation_id=plan.plan.judge_audit_evaluation_ids[0],
        reviewer_id_sha256="f" * 64,
        requires_card=False,
        dimensions=(
            JudgeAuditDimension(dimension="CA", human_score=7),
            JudgeAuditDimension(dimension="CQ", human_score=15),
            JudgeAuditDimension(dimension="TCR", human_score=8),
        ),
    )
    dataset = build_judge_audit_dataset(
        audit_id="audit-001",
        rubric_sha256=plan.plan.final_rubric_sha256,
        rows=(row,),
    )
    plaintext = canonical_json_bytes(dataset.model_dump(mode="json"))
    ciphertext = plaintext[::-1]
    sealed = create_sealed_judge_audit(
        tmp_path / "judge-audit.sealed.json",
        audit_id="audit-001",
        seal_scheme="test-external-reverse-v1",
        ciphertext=ciphertext,
        plaintext_sha256=sha256_bytes(plaintext),
        query_count=1,
        matrix_plan_sha256=plan.plan.plan_sha256,
        selection_manifest_sha256=(plan.plan.judge_audit_selection_manifest_sha256),
        evaluation_ids_sha256=plan.plan.judge_audit_evaluation_ids_sha256,
        final_rubric_sha256=plan.plan.final_rubric_sha256,
        evaluator_isolation=evaluator_lock,
    )
    sealed_bytes = sealed.path.read_bytes()
    assert plaintext not in sealed_bytes
    assert b'"evaluation_id"' not in sealed_bytes

    incomplete = VerifiedFiveConfigRuns(
        plan_sha256=plan.plan.plan_sha256,
        matrix_run_id=plan.plan.matrix_run_id,
        query_artifact_sha256=plan.plan.query_artifact_sha256,
        split_manifest_sha256=plan.plan.split_manifest_sha256,
        query_order_sha256=plan.plan.query_order_sha256,
        query_set_sha256=plan.plan.query_set_sha256,
        runs=(),
        judge_audit_selection_manifest_sha256=(
            plan.plan.judge_audit_selection_manifest_sha256
        ),
        judge_audit_evaluation_ids=plan.plan.judge_audit_evaluation_ids,
        judge_audit_evaluation_ids_sha256=(plan.plan.judge_audit_evaluation_ids_sha256),
        final_rubric_sha256=plan.plan.final_rubric_sha256,
    )
    decryptor = _ReverseDecryptor()
    with pytest.raises(TypeError, match="five externally verified"):
        open_sealed_judge_audit(
            sealed.path,
            incomplete,
            expected_file_sha256=sealed.external_file_sha256,
            evaluator_isolation=evaluator_lock,
            decryptor=decryptor,
        )
    assert decryptor.calls == 0

    five_runs = _five_runs(tmp_path, registry, plan)
    opened = open_sealed_judge_audit(
        sealed.path,
        five_runs,
        expected_file_sha256=sealed.external_file_sha256,
        evaluator_isolation=evaluator_lock,
        decryptor=decryptor,
    )
    assert decryptor.calls == 1
    assert opened.dataset == dataset
    assert opened.formal_eligible is False
    assert opened.formal_ineligible_reason == FORMAL_INELIGIBLE_REASON


def test_sealed_audit_rejects_external_digest_mismatch(tmp_path: Path) -> None:
    registry = _registry()
    plan = _verified_plan(tmp_path, registry, name="audit-digest")
    five_runs = _five_runs(tmp_path, registry, plan)
    _, _, evaluator_lock = _evaluator_lock()
    dataset = build_judge_audit_dataset(
        audit_id="audit-001",
        rubric_sha256=plan.plan.final_rubric_sha256,
        rows=(
            JudgeAuditReferenceRow(
                evaluation_id=plan.plan.judge_audit_evaluation_ids[0],
                reviewer_id_sha256="f" * 64,
                requires_card=False,
                dimensions=(
                    JudgeAuditDimension(dimension="CA", human_score=5),
                    JudgeAuditDimension(dimension="CQ", human_score=10),
                    JudgeAuditDimension(dimension="TCR", human_score=5),
                ),
            ),
        ),
    )
    plaintext = canonical_json_bytes(dataset.model_dump(mode="json"))
    sealed = create_sealed_judge_audit(
        tmp_path / "audit-wrong-digest.json",
        audit_id="audit-001",
        seal_scheme="test-external-reverse-v1",
        ciphertext=plaintext[::-1],
        plaintext_sha256=sha256_bytes(plaintext),
        query_count=1,
        matrix_plan_sha256=plan.plan.plan_sha256,
        selection_manifest_sha256=(plan.plan.judge_audit_selection_manifest_sha256),
        evaluation_ids_sha256=plan.plan.judge_audit_evaluation_ids_sha256,
        final_rubric_sha256=plan.plan.final_rubric_sha256,
        evaluator_isolation=evaluator_lock,
    )
    decryptor = _ReverseDecryptor()
    with pytest.raises(EvaluatorIsolationError, match="external digest"):
        open_sealed_judge_audit(
            sealed.path,
            five_runs,
            expected_file_sha256="0" * 64,
            evaluator_isolation=evaluator_lock,
            decryptor=decryptor,
        )
    assert decryptor.calls == 0
