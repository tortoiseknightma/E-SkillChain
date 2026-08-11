from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from skillchain.evaluation.assistant_runs import (
    ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION,
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    AssistantModelCallReceipt,
    AssistantRequestSnapshot,
    AssistantSharedRouteReference,
    MatrixTreatment,
    RegistryRuntimeLockV2,
    ToolRuntimeBinding,
    build_legacy_assistant_query_input,
    make_backbone_lock,
    make_assistant_route_attempt,
    make_inference_budget,
)
from skillchain.evaluation.packets import (
    AssistantResult,
    AssistantRunConfig,
    AssistantToolTrace,
    VisibleCard,
    VisibleToolEvidence,
)
from skillchain.evaluation.portfolio_gcs import (
    GCS_BOOTSTRAP_REPLICATES,
    GCS_CAPABILITY_ORDER,
    GCS_POLICY_SHA256,
    GCS_POLICY_VERSION,
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
    ExpectedMultiItemV1,
    GCSBootstrapInterval,
    GCSCapabilityBootstrapDelta,
    GCSConfigSummary,
    GCSFiveConfigScoresV2,
    GCSFiveConfigSidecarMapV2,
    GCSQueryScore,
    GCSQueryScoreV2,
    PairedGCSBootstrap,
    PortfolioGCSError,
    PortfolioGCSIntegrityError,
    PublicScorerEvidenceV1,
    build_gcs_five_config_scores_v2,
    build_gcs_five_config_sidecar_map_v2,
    build_gcs_population,
    build_gcs_population_v2,
    evaluate_gcs_system_gain,
    evaluate_gcs_system_gain_five_config_v2,
    gcs_five_config_logical_scores_v2,
    gcs_policy_payload,
    gcs_v2_alias_mapping_policy_payload,
    gcs_v2_policy_payload,
    make_gcs_full_alias_binding_v2,
    make_gcs_physical_sidecar_binding_v2,
    make_gcs_unavailable_physical_sidecar_binding_v2,
    make_public_scorer_call_evidence,
    make_public_scorer_evidence,
    paired_component_bootstrap,
    paired_component_bootstrap_five_config_v2,
    portfolio_gcs_oracles_v1,
    portfolio_gcs_oracles_v2,
    score_portfolio_gcs,
    score_portfolio_gcs_mapped_v2,
    score_portfolio_gcs_v2,
    summarize_gcs,
    summarize_gcs_five_config_v2,
    summarize_gcs_v2,
)
from skillchain.evaluation.portfolio_launch import PortfolioLaunchInstance
from skillchain.evaluation.portfolio_gcs_evidence import (
    PublicScorerCallEvidenceV2,
    PublicScorerEvidenceV2,
    build_public_scorer_call_v2,
    expected_multi_items_from_call_v2,
    make_public_scorer_evidence_v2,
    project_document_ocr_lines_v2,
)
from skillchain.llm import LLMUsage
from skillchain.schemas import LabelDecision, Query
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.synthesis.splitting import GROUP_FIELDS
from skillchain.taxonomy import TAXONOMY_VERSION
from skillchain.tools.contracts import (
    JSONValue,
    ProductSearchTrace,
    RetrievalArtifactBinding,
    StyleFacetEvidence,
    StyleHit,
    validate_json_value,
)
from skillchain.tools.registry import MVP_TOOL_NAMES_V2
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes
from skillchain.schemas import Product
from skillchain.tools.document_ocr import (
    DocumentOCRResult,
    OCRField,
    OCRInputBinding,
    OCRLine,
)
from skillchain.tools.model_artifacts import ModelRuntimeBinding


_TASK_SPEC = load_mvp_task_specification_v1()
_ORACLES = portfolio_gcs_oracles_v1()
_ASSET_TOOLS = {
    "image_product_search",
    "multi_product_search",
    "object_detect",
    "document_ocr",
    "style_similar_search",
}
_CAPABILITY_META = {
    "knowledge.visual_encyclopedia": ("encyclopedia", False),
    "product.exact_match": ("exact_match", True),
    "product.multi_search": ("multi_product", True),
    "product.style_recommendation": ("divergent_rec", True),
    "utility.document_reading": ("utility", False),
    "utility.recipe_guidance": ("utility", False),
}


def _jsonable(value: object) -> JSONValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value))


def _hash_json(value: object) -> str:
    payload = _jsonable(value)
    validate_json_value(payload)
    return sha256_bytes(canonical_json_bytes(payload))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _query(
    capability: str,
    *,
    query_id: str | None = None,
    acceptable_capabilities: Sequence[str] | None = None,
) -> Query:
    intent, requires_card = _CAPABILITY_META[capability]
    query_id = query_id or f"gcs-{capability.replace('.', '-')}"
    acceptable = list(acceptable_capabilities or (capability,))
    text = f"Evaluate fixture {query_id}"
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version="ecommerce-task-spec-v1",
        query_id=query_id,
        asset_id=f"asset.fixture.{query_id}",
        image_path=f"fixtures/{query_id}.jpg",
        leakage_group_id=f"leakage-{query_id}",
        boundary_group_id=None,
        template_family=f"template-{query_id}",
        generator_batch_id=f"batch-{query_id}",
        text=text,
        turns=[{"role": "user", "content": text}],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=acceptable,
        is_boundary=False,
        boundary_strategy=None,
        requires_card=requires_card,
        split="dev_mini",
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="gcs-test",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=acceptable,
            )
        ],
    )


def _argument_projection(query: Query, tool_name: str) -> dict[str, JSONValue]:
    if tool_name in _ASSET_TOOLS:
        return {"asset_handle": "query_asset"}
    if tool_name == "text_product_search":
        return {"query": query.text}
    if tool_name == "encyclopedia_lookup":
        return {"entity": "oak tree"}
    if tool_name == "recipe_lookup":
        return {"dish": "bread"}
    raise AssertionError(tool_name)


def _argument_hash(query: Query, tool_name: str) -> str:
    if tool_name in _ASSET_TOOLS:
        return _hash_json({"asset_id": query.asset_id})
    projection = _argument_projection(query, tool_name)
    return _hash_json(projection)


def _receipt(
    query: Query,
    traces: Sequence[AssistantToolTrace],
    *,
    config: AssistantRunConfig,
    selected_capability: str | None = None,
) -> AssistantExecutionReceipt:
    skilled = config != "noskill"
    selected_capability = selected_capability or query.canonical_capability
    route_trace_sha256 = _digest(f"route:{query.query_id}:{selected_capability}")
    route_attempt = make_assistant_route_attempt(
        status="selected" if skilled else "not_applicable",
        selected_capability=selected_capability if skilled else None,
        skill_slug="fixture-skill" if skilled else None,
        route_trace_sha256=route_trace_sha256 if skilled else None,
    )
    model_call = AssistantModelCallReceipt(
        call_index=1,
        provider="qwen",
        endpoint="https://fixture.invalid/v1",
        requested_model="fixture-model",
        response_model="fixture-model",
        provider_request_id=f"request-{query.query_id}",
        input_tokens=11,
        output_tokens=7,
        finish_reason="stop",
        latency_ms=3,
        response_sha256=_digest(f"model-response:{query.query_id}:{config}"),
    )
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "policy_version": ASSISTANT_EXECUTION_RECEIPT_POLICY_VERSION,
        "request_sha256": _digest(f"request:{query.query_id}:{config}"),
        "asset_catalog_sha256": _digest("asset-catalog"),
        "query_asset_id": query.asset_id,
        "query_asset_sha256": _digest(f"asset-bytes:{query.query_id}"),
        "model_calls": (model_call,),
        "tool_trace": tuple(traces),
        "route_attempt": route_attempt,
        "aggregate_usage": LLMUsage(input_tokens=11, output_tokens=7),
        "runner_latency_ms": 9,
        "outcome": "success",
        "response_sha256": _digest(f"assistant-response:{query.query_id}:{config}"),
    }
    if skilled:
        unsigned["shared_route_reference"] = AssistantSharedRouteReference(
            artifact_sha256=_digest("shared-route-artifact"),
            route_call_response_sha256=_digest("shared-route-response"),
            reserved_usage=LLMUsage(input_tokens=1, output_tokens=1),
            reserved_turns=1,
        )
    return AssistantExecutionReceipt.model_validate(
        {**unsigned, "receipt_sha256": _hash_json(unsigned)}, strict=True
    )


def _execute(
    query: Query,
    *,
    response_text: str,
    calls: Sequence[dict[str, Any]],
    expected_multi_items: Sequence[ExpectedMultiItemV1] | None = None,
    config: AssistantRunConfig = "llm_static",
    selected_capability: str | None = None,
) -> tuple[AssistantResult, AssistantExecutionReceipt, PublicScorerEvidenceV1]:
    traces: list[AssistantToolTrace] = []
    evidence: list[VisibleToolEvidence] = []
    scorer_calls = []
    cards: list[VisibleCard] = []
    for call_index, spec in enumerate(calls, start=1):
        tool_name = spec["tool_name"]
        result_sha256 = _digest(
            f"tool-result:{query.query_id}:{call_index}:{tool_name}"
        )
        trace = AssistantToolTrace(
            call_index=call_index,
            tool_name=tool_name,
            status="success",
            arguments_sha256=_argument_hash(query, tool_name),
            result_sha256=result_sha256,
            runtime_binding_sha256=_digest(f"runtime:{tool_name}"),
            latency_ms=2,
        )
        call_cards = tuple(spec.get("cards", ()))
        traces.append(trace)
        evidence.append(
            VisibleToolEvidence(
                tool_name=tool_name,
                status="success",
                visible_text=spec.get("visible_text"),
                cards=call_cards,
            )
        )
        for card in call_cards:
            if card not in cards:
                cards.append(card)
        scorer_calls.append(
            make_public_scorer_call_evidence(
                call_index=call_index,
                tool_name=tool_name,
                arguments_sha256=trace.arguments_sha256,
                result_sha256=result_sha256,
                argument_projection=_argument_projection(query, tool_name),
                payload_kind=spec["payload_kind"],
                payload=spec["payload"],
            )
        )

    receipt = _receipt(
        query,
        traces,
        config=config,
        selected_capability=selected_capability,
    )
    skilled = config != "noskill"
    routed_capability = selected_capability or query.canonical_capability
    result = AssistantResult(
        run_id="gcs-fixture-run",
        query_id=query.query_id,
        config=config,
        response_text=response_text,
        visible_cards=tuple(cards),
        visible_tool_evidence=tuple(evidence),
        tool_trace=tuple(traces),
        selected_capability=routed_capability if skilled else None,
        skill_slug="fixture-skill" if skilled else None,
        bank_sha256=_digest("fixture-bank") if skilled else None,
        route_trace_sha256=(
            _digest(f"route:{query.query_id}:{routed_capability}") if skilled else None
        ),
        query_artifact_sha256=_digest("query-artifact"),
        split_manifest_sha256=_digest("split-manifest"),
        registry_sha256=_digest("registry"),
        registry_runtime_sha256=_digest("registry-runtime"),
        backbone_provider="qwen",
        backbone_model="fixture-model",
        backbone_request_id=f"answer-{query.query_id}",
        usage=LLMUsage(input_tokens=11, output_tokens=7),
        latency_ms=9,
    )
    sidecar = make_public_scorer_evidence(
        query_id=query.query_id,
        config=config,
        query_artifact_sha256=result.query_artifact_sha256,
        assistant_receipt_sha256=receipt.receipt_sha256,
        calls=scorer_calls,
        expected_multi_items=expected_multi_items,
    )
    return result, receipt, sidecar


def _score(
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    sidecar: PublicScorerEvidenceV1 | None,
):
    return score_portfolio_gcs(
        query,
        result,
        receipt,
        sidecar,
        _TASK_SPEC,
        _ORACLES,
        population=build_gcs_population((query,)),
    )


def _manual_scores(
    queries: Sequence[Query],
    *,
    config: AssistantRunConfig,
    successful_query_ids: set[str],
    unavailable_query_ids: set[str] | None = None,
    hard_error_query_ids: set[str] | None = None,
) -> tuple[GCSQueryScore, ...]:
    population = build_gcs_population(queries)
    bindings = {item.query_id: item for item in population.bindings}
    unavailable = unavailable_query_ids or set()
    hard_errors = hard_error_query_ids or set()
    rows: list[GCSQueryScore] = []
    for query in queries:
        success = query.query_id in successful_query_ids
        oracle_available = query.query_id not in unavailable
        semantic_resolved = oracle_available
        hard_error = query.query_id in hard_errors
        if success and (not oracle_available or hard_error):
            raise AssertionError("a successful fixture row must be fully resolved")
        rows.append(
            GCSQueryScore(
                query_id=query.query_id,
                config=config,
                canonical_capability=query.canonical_capability,
                component_id=bindings[query.query_id].component_id,
                route_disposition="pass",
                answer_mode="supported" if semantic_resolved else "unresolved",
                oracle_available=oracle_available,
                semantic_claim_support_resolved=semantic_resolved,
                route_acceptable=1,
                no_hard_error=0 if hard_error else 1,
                tool_contract_pass=1 if semantic_resolved else 0,
                evidence_grounded=1 if success else 0,
                output_contract_pass=1,
                hard_error=1 if hard_error else 0,
                gcs=1 if success else 0,
                reason_codes=() if success else ("semantic_claim_support_unresolved",),
            )
        )
    return tuple(rows)


def _component_queries(
    component_count: int,
    *,
    capabilities_per_component: Sequence[str] = GCS_CAPABILITY_ORDER,
    prefix: str = "component",
) -> tuple[Query, ...]:
    queries: list[Query] = []
    for component_index in range(component_count):
        for capability_index, capability in enumerate(capabilities_per_component):
            query_id = f"{prefix}-{component_index:03d}-{capability_index:03d}"
            queries.append(
                _query(capability, query_id=query_id).model_copy(
                    update={
                        "leakage_group_id": (f"{prefix}-leakage-{component_index:03d}"),
                        "template_family": f"{prefix}-template-{query_id}",
                        "generator_batch_id": f"{prefix}-batch-{query_id}",
                    }
                )
            )
    return tuple(queries)


def _product_candidate(*, title: str = "Blue Shoe") -> dict[str, JSONValue]:
    return {
        "candidate_ordinal": 1,
        "evidence_reference": "tool-call-1-evidence-1",
        "product_id": "tool-call-1-product-1",
        "title": title,
        "eligible": True,
        "public_attributes": [["color", "blue"]],
    }


def _product_card(*, title: str = "Blue Shoe") -> VisibleCard:
    return VisibleCard(
        title=title,
        body="Public candidate evidence.",
        fields=(
            ("evidence_reference", "tool-call-1-evidence-1"),
            ("product_id", "tool-call-1-product-1"),
        ),
    )


_GCS_V2_PHYSICAL_CONFIGS: tuple[AssistantRunConfig, ...] = (
    "noskill",
    "llm_static",
    "s1",
    "s1s2",
)
_GCS_V2_LOGICAL_CONFIGS: tuple[AssistantRunConfig, ...] = (
    *_GCS_V2_PHYSICAL_CONFIGS,
    "full",
)
_GCS_V2_MATRIX_RUN_ID = "gcs-v2-fixture-matrix"
_GCS_V2_BATCH_ID = "gcs-v2-fixture-batch"
_GCS_V2_RUNTIME_SHA256 = _digest("gcs-v2-runtime")
_GCS_V2_LAUNCH_SHA256 = _digest("gcs-v2-launch")
_GCS_V2_QUERY_ARTIFACT_SHA256 = _digest("gcs-v2-query-artifact-file")
_GCS_V2_SPLIT_MANIFEST_SHA256 = _digest("gcs-v2-split-manifest")
_GCS_V2_BANK_SHA256 = _digest("gcs-v2-s1s2-bank")
_GCS_V2_BANK_FILE_SHA256 = _digest("gcs-v2-s1s2-bank-file")


def _gcs_v2_shard_id(config: AssistantRunConfig) -> str:
    return f"gcs-v2-fixture-{config}-shard"


def _gcs_v2_launch_instance(
    query: Query,
    config: AssistantRunConfig,
    *,
    query_ordinal: int,
) -> PortfolioLaunchInstance:
    config_ordinal = _GCS_V2_LOGICAL_CONFIGS.index(config)
    shard_id = _gcs_v2_shard_id(config)
    query_input = build_legacy_assistant_query_input(query)
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "kind": "portfolio-launch-instance",
        "matrix_run_id": _GCS_V2_MATRIX_RUN_ID,
        "instance_ordinal": config_ordinal * 100 + query_ordinal,
        "shard_id": shard_id,
        "shard_ordinal": config_ordinal,
        "query_ordinal": query_ordinal,
        "config_ordinal": config_ordinal,
        "config": config,
        "accepted_batch_id": _GCS_V2_BATCH_ID,
        "query_id": query.query_id,
        "query_sha256": query_input.query_sha256,
        "public_input_sha256": query_input.public_input_sha256,
        "asset_id": query.asset_id,
        "image_path": query.image_path,
        "image_sha256": _digest(f"asset-bytes:{query.query_id}"),
        "assistant_output_relpath": (
            f"shards/{shard_id}/assistant/{query.query_id}.json"
        ),
        "final_output_relpath": f"shards/{shard_id}/final/{query.query_id}.json",
    }
    return PortfolioLaunchInstance.model_validate(
        {**unsigned, "instance_sha256": _hash_json(unsigned)}, strict=True
    )


def _gcs_v2_registry_lock() -> RegistryRuntimeLockV2:
    tools = tuple(
        ToolRuntimeBinding(
            tool_name=name,
            tool_spec_sha256=_digest(f"gcs-v2-tool-spec:{name}"),
            runtime_binding_sha256=_digest(f"gcs-v2-runtime-binding:{name}"),
        )
        for name in sorted(MVP_TOOL_NAMES_V2)
    )
    unsigned = {
        "schema_version": 2,
        "policy_version": "assistant-registry-runtime-v2",
        "registry_sha256": _digest("registry"),
        "registry_runtime_sha256": _digest("registry-runtime"),
        "tools": tools,
    }
    return RegistryRuntimeLockV2.model_validate(
        {**unsigned, "lock_sha256": _hash_json(unsigned)}, strict=True
    )


def _gcs_v2_treatment(config: AssistantRunConfig) -> MatrixTreatment:
    stages = {
        "noskill": (None, "disabled", "disabled"),
        "llm_static": (_digest("fixture-bank"), "llm_static", "llm_static"),
        "s1": (_digest("fixture-bank"), "s1", "s1"),
        "s1s2": (_GCS_V2_BANK_SHA256, "s2", "s1"),
        "full": (_GCS_V2_BANK_SHA256, "s2", "s3"),
    }
    bank, router, body = stages[config]
    return MatrixTreatment(
        config=config,
        bank_sha256=bank,
        router_stage=router,
        body_stage=body,
    )


def _gcs_v2_execution_bundle(
    query: Query,
    config: AssistantRunConfig,
    *,
    query_ordinal: int,
) -> tuple[
    PortfolioLaunchInstance,
    AssistantResult,
    AssistantExecutionReceipt,
    PublicScorerEvidenceV2,
    bytes,
]:
    result, receipt, sidecar = _execute(
        query,
        config=config,
        response_text=(
            "answer: Supported match\n"
            "product_cards: Blue Shoe tool-call-1-evidence-1 "
            "tool-call-1-product-1\n"
            "uncertainty: Candidate-only result."
        ),
        calls=[
            {
                "tool_name": "image_product_search",
                "payload_kind": "product_candidates_v1",
                "payload": {"candidates": [_product_candidate()]},
                "cards": (_product_card(),),
            }
        ],
    )
    launch_instance = _gcs_v2_launch_instance(
        query, config, query_ordinal=query_ordinal
    )
    treatment = _gcs_v2_treatment(config)
    backbone = make_backbone_lock(
        provider="qwen",
        model="fixture-model",
        endpoint="https://fixture.invalid/v1",
        temperature=0.0,
        top_p=1.0,
        seed=None,
        system_prompt_sha256=_digest("gcs-v2-system-prompt"),
    )
    budget = make_inference_budget(
        max_input_tokens=4096,
        max_output_tokens=512,
        max_tool_calls=3,
        max_turns=5,
        timeout_ms=30_000,
    )
    registry = _gcs_v2_registry_lock()
    request_unsigned = {
        "schema_version": 1,
        "matrix_run_id": _GCS_V2_MATRIX_RUN_ID,
        "config": config,
        "query_ordinal": query_ordinal,
        "query": build_legacy_assistant_query_input(query),
        "treatment": treatment,
        "backbone": backbone,
        "budget": budget,
        "registry": registry,
    }
    request = AssistantRequestSnapshot.model_validate(
        {**request_unsigned, "request_sha256": _hash_json(request_unsigned)},
        strict=True,
    )
    bound_result = result.model_copy(
        update={
            "run_id": _GCS_V2_MATRIX_RUN_ID,
            "bank_sha256": treatment.bank_sha256,
            "query_artifact_sha256": _GCS_V2_QUERY_ARTIFACT_SHA256,
            "split_manifest_sha256": _GCS_V2_SPLIT_MANIFEST_SHA256,
        }
    )
    response = AssistantBackendResponse(
        schema_version=2,
        request_sha256=request.request_sha256,
        backbone_provider=backbone.provider,
        backbone_model=backbone.model,
        backbone_endpoint=backbone.endpoint,
        backbone_identity_sha256=backbone.identity_sha256,
        registry_sha256=registry.registry_sha256,
        registry_runtime_sha256=registry.registry_runtime_sha256,
        budget_sha256=budget.budget_sha256,
        response_text=bound_result.response_text,
        visible_cards=bound_result.visible_cards,
        visible_tool_evidence=bound_result.visible_tool_evidence,
        tool_trace=bound_result.tool_trace,
        selected_capability=bound_result.selected_capability,
        skill_slug=bound_result.skill_slug,
        route_trace_sha256=bound_result.route_trace_sha256,
        backbone_request_id=bound_result.backbone_request_id,
        usage=bound_result.usage,
        turn_count=2,
        latency_ms=bound_result.latency_ms,
        error_code=bound_result.error_code,
    )
    receipt_unsigned = receipt.model_dump(mode="python", exclude={"receipt_sha256"})
    if receipt_unsigned.get("shared_route_reference") is None:
        receipt_unsigned.pop("shared_route_reference", None)
    receipt_unsigned.update(
        {
            "request_sha256": request.request_sha256,
            "query_asset_sha256": launch_instance.image_sha256,
            "response_sha256": _hash_json(response.model_dump(mode="json")),
        }
    )
    bound_receipt = AssistantExecutionReceipt.model_validate(
        {
            **receipt_unsigned,
            "receipt_sha256": _hash_json(receipt_unsigned),
        },
        strict=True,
    )
    v2_calls = tuple(
        PublicScorerCallEvidenceV2.model_validate(
            call.model_dump(mode="python"), strict=True
        )
        for call in sidecar.calls
    )
    bound_sidecar = make_public_scorer_evidence_v2(
        matrix_run_id=_GCS_V2_MATRIX_RUN_ID,
        instance_id=launch_instance.instance_sha256,
        request_sha256=request.request_sha256,
        query_id=query.query_id,
        config=config,
        query_artifact_sha256=_GCS_V2_QUERY_ARTIFACT_SHA256,
        assistant_result_sha256=_hash_json(bound_result),
        assistant_receipt_sha256=bound_receipt.receipt_sha256,
        calls=v2_calls,
    )
    row_unsigned = {
        "schema_version": 2,
        "kind": "portfolio-assistant-checkpoint",
        "instance_sha256": launch_instance.instance_sha256,
        "query_ordinal": launch_instance.query_ordinal,
        "request": request,
        "response": response,
        "receipt": bound_receipt,
        "public_scorer_evidence": bound_sidecar,
    }
    checkpoint_bytes = canonical_json_bytes(
        {
            **_jsonable(row_unsigned),
            "row_sha256": _hash_json(row_unsigned),
        }
    )
    return (
        launch_instance,
        bound_result,
        bound_receipt,
        bound_sidecar,
        checkpoint_bytes,
    )


def _gcs_v2_treatment_alias() -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "portfolio-execution-artifact-alias",
        "policy_version": "portfolio-execution-artifact-alias-v1",
        "target_config": "full",
        "source_config": "s1s2",
        "stage_decision": "rolled_back",
        "source_bank_sha256": _GCS_V2_BANK_SHA256,
        "target_bank_sha256": _GCS_V2_BANK_SHA256,
        "source_bank_file_sha256": _GCS_V2_BANK_FILE_SHA256,
        "target_bank_file_sha256": _GCS_V2_BANK_FILE_SHA256,
        "reuse_scope": "assistant_and_evaluator_query_artifacts",
        "provider_model_call_count": 0,
        "rejected_candidate_use": "diagnostic_only",
    }
    return {**unsigned, "alias_sha256": _hash_json(unsigned)}


def _gcs_v2_alias_receipt(
    query_ids: Sequence[str],
    treatment_alias: Mapping[str, object],
    *,
    updates: Mapping[str, object] | None = None,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "kind": "portfolio-shard-artifact-alias",
        "policy_version": "portfolio-execution-artifact-alias-v1",
        "matrix_run_id": _GCS_V2_MATRIX_RUN_ID,
        "accepted_batch_id": _GCS_V2_BATCH_ID,
        "target_shard_id": _gcs_v2_shard_id("full"),
        "target_config": "full",
        "source_shard_id": _gcs_v2_shard_id("s1s2"),
        "source_config": "s1s2",
        "query_ids": tuple(query_ids),
        "treatment_alias_sha256": treatment_alias["alias_sha256"],
        "source_bank_sha256": _GCS_V2_BANK_SHA256,
        "target_bank_sha256": _GCS_V2_BANK_SHA256,
        "source_shard_summary_file_sha256": _digest("v2-summary-file"),
        "source_shard_summary_sha256": _digest("v2-summary"),
        "source_shard_audit_file_sha256": _digest("v2-audit-file"),
        "source_shard_audit_sha256": _digest("v2-audit"),
        "reuse_scope": "assistant_and_evaluator_query_artifacts",
        "provider_model_call_count": 0,
        "rejected_candidate_use": "diagnostic_only",
        "runtime_lock_sha256": _GCS_V2_RUNTIME_SHA256,
        "launch_plan_sha256": _GCS_V2_LAUNCH_SHA256,
    }
    unsigned.update(updates or {})
    return {**unsigned, "alias_receipt_sha256": _hash_json(unsigned)}


def _gcs_v2_alias_receipt_bytes(receipt: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(_jsonable(receipt))


def _gcs_v2_physical_fixture(
    queries: Sequence[Query],
) -> tuple[
    dict[
        tuple[str, AssistantRunConfig],
        tuple[
            AssistantResult,
            AssistantExecutionReceipt,
            PublicScorerEvidenceV2,
            bytes,
        ],
    ],
    tuple[PortfolioLaunchInstance, ...],
    tuple[Any, ...],
    dict[str, object],
]:
    artifacts: dict[
        tuple[str, AssistantRunConfig],
        tuple[
            AssistantResult,
            AssistantExecutionReceipt,
            PublicScorerEvidenceV2,
            bytes,
        ],
    ] = {}
    launch_instances = []
    physical_bindings = []
    for config in _GCS_V2_PHYSICAL_CONFIGS:
        for query_ordinal, query in enumerate(queries):
            (
                launch_instance,
                result,
                receipt,
                sidecar,
                checkpoint_bytes,
            ) = _gcs_v2_execution_bundle(
                query,
                config,
                query_ordinal=query_ordinal,
            )
            launch_instances.append(launch_instance)
            artifacts[(query.query_id, config)] = (
                result,
                receipt,
                sidecar,
                checkpoint_bytes,
            )
            physical_bindings.append(
                make_gcs_physical_sidecar_binding_v2(
                    launch_instance=launch_instance,
                    query=query,
                    result=result,
                    receipt=receipt,
                    sidecar=sidecar,
                    query_artifact_sha256=result.query_artifact_sha256,
                    split_manifest_sha256=result.split_manifest_sha256,
                    assistant_checkpoint_file_bytes=checkpoint_bytes,
                    runtime_lock_sha256=_GCS_V2_RUNTIME_SHA256,
                    launch_plan_sha256=_GCS_V2_LAUNCH_SHA256,
                )
            )
    for query_ordinal, query in enumerate(queries):
        launch_instances.append(
            _gcs_v2_launch_instance(query, "full", query_ordinal=query_ordinal)
        )
    return (
        artifacts,
        tuple(launch_instances),
        tuple(physical_bindings),
        _gcs_v2_treatment_alias(),
    )


def _gcs_v2_sidecar_map(
    queries: Sequence[Query],
    *,
    alias_receipt_updates: Mapping[str, object] | None = None,
) -> tuple[
    GCSFiveConfigSidecarMapV2,
    dict[
        tuple[str, AssistantRunConfig],
        tuple[
            AssistantResult,
            AssistantExecutionReceipt,
            PublicScorerEvidenceV2,
            bytes,
        ],
    ],
]:
    (
        artifacts,
        launch_instances,
        physical_bindings,
        treatment_alias,
    ) = _gcs_v2_physical_fixture(queries)
    receipt = _gcs_v2_alias_receipt(
        tuple(query.query_id for query in queries),
        treatment_alias,
        updates=alias_receipt_updates,
    )
    alias = make_gcs_full_alias_binding_v2(
        receipt,
        treatment_alias,
        alias_receipt_file_bytes=_gcs_v2_alias_receipt_bytes(receipt),
    )
    sidecar_map = build_gcs_five_config_sidecar_map_v2(
        queries,
        launch_instances,
        physical_bindings,
        (alias,),
        (alias.treatment_alias,),
    )
    return sidecar_map, artifacts


def _gcs_v2_scores(
    queries: Sequence[Query],
    sidecar_map: GCSFiveConfigSidecarMapV2,
    artifacts: Mapping[
        tuple[str, AssistantRunConfig],
        tuple[
            AssistantResult,
            AssistantExecutionReceipt,
            PublicScorerEvidenceV2,
            bytes,
        ],
    ],
) -> GCSFiveConfigScoresV2:
    population = build_gcs_population_v2(queries)
    rows = []
    for query in queries:
        for logical_config in _GCS_V2_LOGICAL_CONFIGS:
            physical_config: AssistantRunConfig = (
                "s1s2" if logical_config == "full" else logical_config
            )
            result, receipt, sidecar, checkpoint_bytes = artifacts[
                (query.query_id, physical_config)
            ]
            rows.append(
                score_portfolio_gcs_mapped_v2(
                    query,
                    result,
                    receipt,
                    sidecar,
                    _TASK_SPEC,
                    portfolio_gcs_oracles_v2(),
                    logical_config=logical_config,
                    assistant_checkpoint_file_bytes=checkpoint_bytes,
                    population=population,
                    sidecar_map=sidecar_map,
                )
            )
    return build_gcs_five_config_scores_v2(rows, queries, sidecar_map)


def test_policy_payload_and_registry_freeze_the_six_capability_contract() -> None:
    payload = gcs_policy_payload()

    assert payload["policy_version"] == GCS_POLICY_VERSION
    assert tuple(payload["capabilities"]) == GCS_CAPABILITY_ORDER
    assert set(_ORACLES) == set(GCS_CAPABILITY_ORDER)
    assert _hash_json(payload) == GCS_POLICY_SHA256
    payload["formula"] = "tampered"
    assert gcs_policy_payload()["formula"] == "int(all_five_components)"


def test_gcs_v2_projects_four_physical_sidecars_to_five_exact_logical_rows() -> None:
    query = _query("product.exact_match", query_id="gcs-v2-five-config")
    queries = (query,)
    sidecar_map, artifacts = _gcs_v2_sidecar_map(queries)

    mapping_payload = gcs_v2_alias_mapping_policy_payload()
    assert mapping_payload["gcs_policy_sha256"] == GCS_V2_POLICY_SHA256
    assert _hash_json(mapping_payload) == sidecar_map.mapping_policy_sha256
    assert sidecar_map.policy_sha256 == GCS_V2_POLICY_SHA256
    assert sidecar_map.physical_sidecar_count == 4
    assert sidecar_map.logical_row_count == 5
    assert tuple(row.logical_config for row in sidecar_map.rows) == (
        _GCS_V2_LOGICAL_CONFIGS
    )
    source_projection = next(
        row for row in sidecar_map.rows if row.logical_config == "s1s2"
    )
    full_projection = next(
        row for row in sidecar_map.rows if row.logical_config == "full"
    )
    assert full_projection.projection_kind == "execution_artifact_alias"
    assert full_projection.physical_config == "s1s2"
    assert full_projection.physical_shard_id == source_projection.physical_shard_id
    assert (
        full_projection.physical_binding_sha256
        == source_projection.physical_binding_sha256
    )
    assert full_projection.scorer_evidence_sha256 == (
        source_projection.scorer_evidence_sha256
    )
    assert full_projection.scorer_evidence_document_sha256 == (
        source_projection.scorer_evidence_document_sha256
    )

    matrix = _gcs_v2_scores(queries, sidecar_map, artifacts)

    assert matrix.logical_row_count == 5
    assert tuple(row.logical_score.config for row in matrix.rows) == (
        _GCS_V2_LOGICAL_CONFIGS
    )
    assert all(row.logical_score.gcs == 1 for row in matrix.rows)
    source_row = next(row for row in matrix.rows if row.logical_score.config == "s1s2")
    full_row = next(row for row in matrix.rows if row.logical_score.config == "full")
    assert full_row.physical_score == source_row.physical_score
    assert full_row.logical_score.model_dump(
        mode="python", exclude={"config"}
    ) == source_row.logical_score.model_dump(mode="python", exclude={"config"})

    source_scores = gcs_five_config_logical_scores_v2(matrix, "s1s2")
    full_scores = gcs_five_config_logical_scores_v2(matrix, "full")
    assert len(source_scores) == len(full_scores) == 1
    bootstrap = paired_component_bootstrap_five_config_v2(
        matrix, "s1s2", "full", queries, "gcs-v2-full-alias"
    ).bootstrap
    assert bootstrap.zero_variance is True
    assert bootstrap.macro_delta_pp is None
    assert bootstrap.macro_ci95 is None
    assert bootstrap.query_micro_delta_pp == 0.0
    assert bootstrap.query_micro_ci95 == GCSBootstrapInterval(low_pp=0.0, high_pp=0.0)
    assert bootstrap.hard_error_delta_pp == 0.0
    assert bootstrap.hard_error_ci95 == GCSBootstrapInterval(low_pp=0.0, high_pp=0.0)
    exact = next(
        item
        for item in bootstrap.per_capability
        if item.capability_id == "product.exact_match"
    )
    assert exact.point_delta_pp == 0.0
    assert exact.ci95 == GCSBootstrapInterval(low_pp=0.0, high_pp=0.0)


def test_gcs_v2_missing_s1s2_sidecar_is_fatal_before_full_projection() -> None:
    query = _query("product.exact_match", query_id="gcs-v2-missing-sidecar")
    launch, result, receipt, _sidecar, checkpoint = _gcs_v2_execution_bundle(
        query, "s1s2", query_ordinal=0
    )

    with pytest.raises(PortfolioGCSIntegrityError, match="embedded valid scorer"):
        make_gcs_unavailable_physical_sidecar_binding_v2(
            launch_instance=launch,
            config="s1s2",
            query=query,
            result=result,
            receipt=receipt,
            query_artifact_sha256=result.query_artifact_sha256,
            split_manifest_sha256=result.split_manifest_sha256,
            assistant_checkpoint_file_bytes=checkpoint,
            sidecar_status="missing",
            runtime_lock_sha256=_GCS_V2_RUNTIME_SHA256,
            launch_plan_sha256=_GCS_V2_LAUNCH_SHA256,
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"treatment_alias_sha256": _digest("wrong-treatment-alias")},
        {"source_config": "s1"},
    ),
    ids=("alias-sha", "alias-config"),
)
def test_gcs_v2_rejects_wrong_alias_identity(
    updates: Mapping[str, object],
) -> None:
    query = _query("product.exact_match", query_id="gcs-v2-bad-alias")
    treatment_alias = _gcs_v2_treatment_alias()
    receipt = _gcs_v2_alias_receipt((query.query_id,), treatment_alias, updates=updates)

    with pytest.raises(
        PortfolioGCSError,
        match="Full .* alias .* invalid|differs|binding is inconsistent",
    ):
        make_gcs_full_alias_binding_v2(
            receipt,
            treatment_alias,
            alias_receipt_file_bytes=_gcs_v2_alias_receipt_bytes(receipt),
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"runtime_lock_sha256": _digest("wrong-runtime")},
        {"accepted_batch_id": "wrong-accepted-batch"},
    ),
    ids=("runtime", "batch"),
)
def test_gcs_v2_rejects_alias_root_or_batch_drift(
    updates: Mapping[str, object],
) -> None:
    query = _query("product.exact_match", query_id="gcs-v2-root-drift")
    queries = (query,)
    (
        _artifacts,
        launch_instances,
        physical_bindings,
        treatment_alias,
    ) = _gcs_v2_physical_fixture(queries)
    receipt = _gcs_v2_alias_receipt((query.query_id,), treatment_alias, updates=updates)
    alias = make_gcs_full_alias_binding_v2(
        receipt,
        treatment_alias,
        alias_receipt_file_bytes=_gcs_v2_alias_receipt_bytes(receipt),
    )

    with pytest.raises(PortfolioGCSError, match="sidecar mapping is invalid"):
        build_gcs_five_config_sidecar_map_v2(
            queries,
            launch_instances,
            physical_bindings,
            (alias,),
            (alias.treatment_alias,),
        )


def test_gcs_v2_rejects_full_alias_query_order_drift() -> None:
    queries = (
        _query("product.exact_match", query_id="gcs-v2-order-001"),
        _query("product.exact_match", query_id="gcs-v2-order-002"),
    )
    (
        _artifacts,
        launch_instances,
        physical_bindings,
        treatment_alias,
    ) = _gcs_v2_physical_fixture(queries)
    receipt = _gcs_v2_alias_receipt(
        tuple(query.query_id for query in reversed(queries)), treatment_alias
    )
    alias = make_gcs_full_alias_binding_v2(
        receipt,
        treatment_alias,
        alias_receipt_file_bytes=_gcs_v2_alias_receipt_bytes(receipt),
    )

    with pytest.raises(PortfolioGCSError, match="sidecar mapping is invalid"):
        build_gcs_five_config_sidecar_map_v2(
            queries,
            launch_instances,
            physical_bindings,
            (alias,),
            (alias.treatment_alias,),
        )


def test_gcs_v2_rejects_a_physical_full_sidecar() -> None:
    query = _query("product.exact_match", query_id="gcs-v2-physical-full")
    launch_instance, result, receipt, sidecar, checkpoint_bytes = (
        _gcs_v2_execution_bundle(query, "full", query_ordinal=0)
    )

    with pytest.raises(PortfolioGCSError, match="physical Full"):
        make_gcs_physical_sidecar_binding_v2(
            launch_instance=launch_instance,
            query=query,
            result=result,
            receipt=receipt,
            sidecar=sidecar,
            query_artifact_sha256=result.query_artifact_sha256,
            split_manifest_sha256=result.split_manifest_sha256,
            assistant_checkpoint_file_bytes=checkpoint_bytes,
            runtime_lock_sha256=_GCS_V2_RUNTIME_SHA256,
            launch_plan_sha256=_GCS_V2_LAUNCH_SHA256,
        )


def test_gcs_v2_envelopes_bind_full_alias_zero_contrast() -> None:
    queries = tuple(
        _query(capability, query_id=f"gcs-v2-envelope-{index}")
        for index, capability in enumerate(GCS_CAPABILITY_ORDER)
    )
    sidecar_map, artifacts = _gcs_v2_sidecar_map(queries)
    matrix = _gcs_v2_scores(queries, sidecar_map, artifacts)

    summary = summarize_gcs_five_config_v2(matrix, queries, "full", "gcs-v2-envelope")
    bootstrap = paired_component_bootstrap_five_config_v2(
        matrix,
        "s1s2",
        "full",
        queries,
        "gcs-v2-envelope",
    )
    decision = evaluate_gcs_system_gain_five_config_v2(summary, bootstrap)

    assert summary.sidecar_mapping_sha256 == sidecar_map.mapping_sha256
    assert bootstrap.sidecar_mapping_sha256 == sidecar_map.mapping_sha256
    assert decision.sidecar_mapping_sha256 == sidecar_map.mapping_sha256
    assert summary.score_matrix_sha256 == matrix.scores_sha256
    assert bootstrap.score_matrix_sha256 == matrix.scores_sha256
    assert decision.score_matrix_sha256 == matrix.scores_sha256
    assert decision.summary_sha256 == summary.summary_sha256
    assert decision.bootstrap_sha256 == bootstrap.bootstrap_sha256
    assert bootstrap.bootstrap.macro_delta_pp == 0.0
    assert bootstrap.bootstrap.macro_ci95 is None
    assert "replicate_capability_missing" in bootstrap.bootstrap.precision_warnings
    assert bootstrap.bootstrap.query_micro_delta_pp == 0.0
    assert bootstrap.bootstrap.hard_error_delta_pp == 0.0
    assert all(
        item.point_delta_pp == 0.0 for item in bootstrap.bootstrap.per_capability
    )

    drifted_summary = summary.model_copy(
        update={"sidecar_mapping_sha256": _digest("wrong-v2-map")}
    )
    with pytest.raises(
        PortfolioGCSError,
        match="summary/bootstrap contract is invalid|provenance differs",
    ):
        evaluate_gcs_system_gain_five_config_v2(drifted_summary, bootstrap)


def test_gcs_v2_uses_global_zero_based_ordinals_and_distinct_full_instances() -> None:
    queries = (
        _query("product.exact_match", query_id="gcs-v2-ordinal-000"),
        _query("product.exact_match", query_id="gcs-v2-ordinal-001"),
    )
    sidecar_map, _artifacts = _gcs_v2_sidecar_map(queries)
    instances = {
        (item.query_id, item.config): item for item in sidecar_map.launch_instances
    }

    for expected_ordinal, query in enumerate(queries):
        per_config = tuple(
            instances[(query.query_id, config)] for config in _GCS_V2_LOGICAL_CONFIGS
        )
        assert {item.query_ordinal for item in per_config} == {expected_ordinal}
        source = instances[(query.query_id, "s1s2")]
        full = instances[(query.query_id, "full")]
        assert full.instance_sha256 != source.instance_sha256
        assert full.shard_id != source.shard_id
        assert full.assistant_output_relpath != source.assistant_output_relpath

        full_projection = next(
            row
            for row in sidecar_map.rows
            if row.query_id == query.query_id and row.logical_config == "full"
        )
        assert full_projection.logical_instance_sha256 == full.instance_sha256
        assert full_projection.physical_instance_sha256 == source.instance_sha256


def test_gcs_v2_rejects_invalid_query_before_mapping() -> None:
    invalid_query = _query(
        "product.exact_match", query_id="gcs-v2-invalid-query"
    ).model_copy(update={"acceptable_capabilities": []})

    with pytest.raises(PortfolioGCSError, match="invalid Query"):
        build_gcs_five_config_sidecar_map_v2(
            (invalid_query,),
            (),
            (),
            (),
            (),
        )


def test_gcs_v2_rejects_checkpoint_prompt_from_a_stale_query() -> None:
    query = _query("product.exact_match", query_id="gcs-v2-stale-prompt")
    launch_instance, result, receipt, sidecar, checkpoint_bytes = (
        _gcs_v2_execution_bundle(query, "llm_static", query_ordinal=0)
    )
    stale_text = "A different prompt with the same query id"
    stale_query = query.model_copy(
        update={
            "text": stale_text,
            "turns": [query.turns[0].model_copy(update={"content": stale_text})],
        }
    )

    with pytest.raises(
        PortfolioGCSError,
        match="checkpoint/launch/result/receipt binding is inconsistent",
    ):
        make_gcs_physical_sidecar_binding_v2(
            launch_instance=launch_instance,
            query=stale_query,
            result=result,
            receipt=receipt,
            sidecar=sidecar,
            query_artifact_sha256=result.query_artifact_sha256,
            split_manifest_sha256=result.split_manifest_sha256,
            assistant_checkpoint_file_bytes=checkpoint_bytes,
            runtime_lock_sha256=_GCS_V2_RUNTIME_SHA256,
            launch_plan_sha256=_GCS_V2_LAUNCH_SHA256,
        )


def test_public_sidecar_is_self_hashed_and_rejects_private_asset_identity() -> None:
    query = _query("product.exact_match")
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            "answer: Supported match\n"
            "product_cards: Blue Shoe tool-call-1-evidence-1 "
            "tool-call-1-product-1\n"
            "uncertainty: Candidate-only result."
        ),
        calls=[
            {
                "tool_name": "image_product_search",
                "payload_kind": "product_candidates_v1",
                "payload": {"candidates": [_product_candidate()]},
                "cards": (_product_card(),),
            }
        ],
    )
    assert sidecar.query_artifact_sha256 == result.query_artifact_sha256
    assert sidecar.assistant_receipt_sha256 == receipt.receipt_sha256

    call = sidecar.calls[0].model_dump(mode="json")
    call["argument_projection"] = {"asset_id": query.asset_id}
    call["payload_sha256"] = _hash_json(call["payload"])
    with pytest.raises(ValidationError, match="forbidden private key"):
        make_public_scorer_call_evidence(
            call_index=call["call_index"],
            tool_name=call["tool_name"],
            arguments_sha256=call["arguments_sha256"],
            result_sha256=call["result_sha256"],
            argument_projection=call["argument_projection"],
            payload_kind=call["payload_kind"],
            payload=call["payload"],
        )

    tampered = sidecar.model_dump(mode="json")
    tampered["query_id"] = "different-query"
    with pytest.raises(ValidationError, match="self hash mismatch"):
        PublicScorerEvidenceV1.model_validate(tampered, strict=True)


def test_exact_supported_closes_candidate_card_title_and_handles() -> None:
    query = _query("product.exact_match")
    response = (
        "answer: Supported match\n"
        "product_cards: Blue Shoe tool-call-1-evidence-1 "
        "tool-call-1-product-1\n"
        "uncertainty: Candidate-only result."
    )
    result, receipt, sidecar = _execute(
        query,
        response_text=response,
        calls=[
            {
                "tool_name": "image_product_search",
                "payload_kind": "product_candidates_v1",
                "payload": {"candidates": [_product_candidate()]},
                "cards": (_product_card(),),
            }
        ],
    )

    score = _score(query, result, receipt, sidecar)

    assert score.gcs == 1
    assert score.answer_mode == "supported"
    assert result.visible_cards[0].title == "Blue Shoe"
    assert "title" not in dict(result.visible_cards[0].fields)

    bad_result = result.model_copy(
        update={
            "response_text": response.replace(
                "Candidate-only result.",
                "Unknown tool-call-2-evidence-1.",
            )
        }
    )
    bad_score = _score(query, bad_result, receipt, sidecar)
    assert bad_score.gcs == 0
    assert bad_score.evidence_grounded == 0
    assert "evidence_handle_unknown" in bad_score.reason_codes


def test_exact_zero_candidate_fallback_is_valid_but_missing_sidecar_fails_closed() -> (
    None
):
    query = _query("product.exact_match")
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            "answer: No exact match\n"
            "product_cards: No supported match\n"
            "uncertainty: Search returned no eligible candidate."
        ),
        calls=[
            {
                "tool_name": "image_product_search",
                "payload_kind": "product_candidates_v1",
                "payload": {"candidates": []},
                "visible_text": "No eligible candidates.",
            }
        ],
    )

    assert _score(query, result, receipt, sidecar).gcs == 1
    missing = _score(query, result, receipt, None)
    assert missing.gcs == 0
    assert missing.tool_contract_pass == 0
    assert missing.evidence_grounded == 0
    assert "scorer_sidecar_missing" in missing.reason_codes

    ineligible_candidate = _product_candidate()
    ineligible_candidate["eligible"] = False
    ineligible_result, ineligible_receipt, ineligible_sidecar = _execute(
        query,
        response_text=result.response_text,
        calls=[
            {
                "tool_name": "image_product_search",
                "payload_kind": "product_candidates_v1",
                "payload": {"candidates": [ineligible_candidate]},
                "visible_text": "Candidates were returned but none was eligible.",
            }
        ],
    )
    ineligible_fallback = _score(
        query, ineligible_result, ineligible_receipt, ineligible_sidecar
    )
    assert ineligible_fallback.gcs == 1
    assert ineligible_fallback.answer_mode == "fallback"


def test_multi_supported_closes_expected_items_mapping_cards_and_quantities() -> None:
    query = _query("product.multi_search")
    candidate = _product_candidate(title="Shared Blue Shoe")
    card = VisibleCard(
        title="Shared Blue Shoe",
        body="One candidate satisfies the first requested item.",
        fields=(
            ("evidence_reference", "tool-call-1-evidence-1"),
            ("item_refs", "item-001"),
            ("product_id", "tool-call-1-product-1"),
            ("quantity", "1"),
        ),
    )
    expected = (
        ExpectedMultiItemV1(item_ref="item-001", label="blue shoe"),
        ExpectedMultiItemV1(item_ref="item-002", label="red hat"),
    )
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            "answer: One item matched and one remains unresolved.\n"
            "item_mapping: item-001 matched candidate-1\n"
            "item-002 unresolved\n"
            "product_cards: Shared Blue Shoe tool-call-1-evidence-1 "
            "tool-call-1-product-1\n"
            "uncertainty: The second item is unresolved."
        ),
        calls=[
            {
                "tool_name": "multi_product_search",
                "payload_kind": "multi_mapping_v1",
                "payload": {
                    "items": [
                        {
                            "item_ref": "item-001",
                            "label": "blue shoe",
                            "status": "matched",
                            "candidate_ordinal": 1,
                        },
                        {
                            "item_ref": "item-002",
                            "label": "red hat",
                            "status": "unresolved",
                            "candidate_ordinal": None,
                        },
                    ],
                    "candidates": [candidate],
                },
                "cards": (card,),
            }
        ],
        expected_multi_items=expected,
    )

    assert _score(query, result, receipt, sidecar).gcs == 1
    missing_expected = make_public_scorer_evidence(
        query_id=query.query_id,
        config=result.config,
        query_artifact_sha256=result.query_artifact_sha256,
        assistant_receipt_sha256=receipt.receipt_sha256,
        calls=sidecar.calls,
        expected_multi_items=None,
    )
    unresolved = _score(query, result, receipt, missing_expected)
    assert unresolved.gcs == 0
    assert unresolved.answer_mode == "unresolved"
    assert "requested_item_coverage_unresolved" in unresolved.reason_codes

    drifted_card = card.model_copy(
        update={
            "fields": (
                ("evidence_reference", "tool-call-1-evidence-1"),
                ("item_refs", "item-001"),
                ("product_id", "tool-call-1-product-1"),
                ("quantity", "2"),
            )
        }
    )
    drifted_evidence = result.visible_tool_evidence[0].model_copy(
        update={"cards": (drifted_card,)}
    )
    drifted_result = result.model_copy(
        update={
            "visible_cards": (drifted_card,),
            "visible_tool_evidence": (drifted_evidence,),
        }
    )
    drifted = _score(query, drifted_result, receipt, sidecar)
    assert drifted.gcs == 0
    assert "multi_mapping_invalid" in drifted.reason_codes


def test_encyclopedia_requires_every_material_statement_to_cite_a_bound_source() -> (
    None
):
    query = _query("knowledge.visual_encyclopedia")
    handle = "tool-call-1-source-1"
    source_text = "Oak is a hardwood tree."
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            f"answer: Oak is a hardwood tree {handle}\n"
            f"evidence: Oak is a hardwood tree {handle}\n"
            "uncertainty: Identification is limited to the cited source."
        ),
        calls=[
            {
                "tool_name": "encyclopedia_lookup",
                "payload_kind": "knowledge_sources_v1",
                "payload": {
                    "sources": [
                        {
                            "evidence_reference": handle,
                            "title": "Oak",
                            "text": source_text,
                            "uri": "https://example.org/oak",
                        }
                    ]
                },
                "visible_text": f"{handle}\nOak\n{source_text}",
            }
        ],
    )

    assert _score(query, result, receipt, sidecar).gcs == 1
    uncited = result.model_copy(
        update={
            "response_text": result.response_text.replace(
                f"answer: Oak is a hardwood tree {handle}",
                "answer: Oak is a hardwood tree",
            )
        }
    )
    bad_score = _score(query, uncited, receipt, sidecar)
    assert bad_score.gcs == 0
    assert "material_claim_uncited" in bad_score.reason_codes


def test_recipe_supports_bound_values_and_rejects_unsupported_safety_guarantee() -> (
    None
):
    query = _query("utility.recipe_guidance")
    handle = "tool-call-1-source-1"
    source_text = "Bake bread at 180C for 20 minutes."
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            f"answer: Bake bread at 180C for 20 minutes {handle}\n"
            f"evidence: Bake bread at 180C for 20 minutes {handle}\n"
            "uncertainty: Follow the cited recipe only."
        ),
        calls=[
            {
                "tool_name": "recipe_lookup",
                "payload_kind": "knowledge_sources_v1",
                "payload": {
                    "sources": [
                        {
                            "evidence_reference": handle,
                            "title": "Bread",
                            "text": source_text,
                            "uri": "https://example.org/bread",
                        }
                    ]
                },
                "visible_text": f"{handle}\nBread\n{source_text}",
            }
        ],
    )

    assert _score(query, result, receipt, sidecar).gcs == 1
    unsafe = result.model_copy(
        update={
            "response_text": result.response_text.replace(
                f"answer: Bake bread at 180C for 20 minutes {handle}",
                f"answer: Bake bread at 180C for 20 minutes; guaranteed safe {handle}",
            )
        }
    )
    bad_score = _score(query, unsafe, receipt, sidecar)
    assert bad_score.gcs == 0
    assert "unsupported_claim" in bad_score.reason_codes


def test_document_requires_line_bound_substrings_and_untrusted_content_marker() -> None:
    query = _query("utility.document_reading")
    handle = "tool-call-1-line-1"
    line_text = "Total: $12.50"
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            f"answer: total: $12.50 {handle}\n"
            f"evidence: total: $12.50 {handle}\n"
            "uncertainty: Untrusted transcription only."
        ),
        calls=[
            {
                "tool_name": "document_ocr",
                "payload_kind": "ocr_lines_v1",
                "payload": {
                    "lines": [
                        {
                            "line_reference": handle,
                            "text": line_text,
                            "content_trust": "untrusted_document_text",
                            "truncated": False,
                            "fields": [],
                        }
                    ]
                },
                "visible_text": f"{handle}\n{line_text}",
            }
        ],
    )

    assert _score(query, result, receipt, sidecar).gcs == 1
    unsupported = result.model_copy(
        update={"response_text": result.response_text.replace("$12.50", "$99.00")}
    )
    bad_value = _score(query, unsupported, receipt, sidecar)
    assert bad_value.gcs == 0
    assert "unsupported_claim" in bad_value.reason_codes

    trusted_claim = result.model_copy(
        update={
            "response_text": result.response_text.replace(
                "Untrusted transcription only.", "Verified transcription."
            )
        }
    )
    no_marker = _score(query, trusted_claim, receipt, sidecar)
    assert no_marker.gcs == 0
    assert "unsupported_claim" in no_marker.reason_codes


def test_multi_knowledge_recipe_and_document_zero_result_fallbacks_are_valid() -> None:
    cases: tuple[
        tuple[
            str,
            str,
            dict[str, Any],
            tuple[ExpectedMultiItemV1, ...] | None,
        ],
        ...,
    ] = (
        (
            "product.multi_search",
            "answer: All requested items remain unresolved\n"
            "item_mapping: item-001 unresolved\n"
            "product_cards: No supported item\n"
            "uncertainty: Unable to identify a supported item.",
            {
                "tool_name": "multi_product_search",
                "payload_kind": "multi_mapping_v1",
                "payload": {
                    "items": [
                        {
                            "item_ref": "item-001",
                            "label": "blue shoe",
                            "status": "unresolved",
                            "candidate_ordinal": None,
                        }
                    ],
                    "candidates": [],
                },
                "visible_text": "No matched candidates.",
            },
            (ExpectedMultiItemV1(item_ref="item-001", label="blue shoe"),),
        ),
        (
            "knowledge.visual_encyclopedia",
            "answer: Not enough evidence\n"
            "evidence: Unable to verify\n"
            "uncertainty: Ambiguous.",
            {
                "tool_name": "encyclopedia_lookup",
                "payload_kind": "knowledge_sources_v1",
                "payload": {"sources": []},
                "visible_text": "No public sources found.",
            },
            None,
        ),
        (
            "utility.recipe_guidance",
            "answer: No supported recipe\n"
            "evidence: Not enough evidence\n"
            "uncertainty: Dish is unresolved.",
            {
                "tool_name": "recipe_lookup",
                "payload_kind": "knowledge_sources_v1",
                "payload": {"sources": []},
                "visible_text": "No public recipe source found.",
            },
            None,
        ),
        (
            "utility.document_reading",
            "answer: Unable to read\n"
            "evidence: Insufficient text\n"
            "uncertainty: Unreadable.",
            {
                "tool_name": "document_ocr",
                "payload_kind": "ocr_lines_v1",
                "payload": {"lines": []},
                "visible_text": "OCR returned no lines.",
            },
            None,
        ),
    )

    for capability, response, call, expected_items in cases:
        query = _query(capability, query_id=f"fallback-{capability}")
        result, receipt, sidecar = _execute(
            query,
            response_text=response,
            calls=[call],
            expected_multi_items=expected_items,
        )

        score = _score(query, result, receipt, sidecar)

        assert score.gcs == 1, (capability, score.reason_codes)
        assert score.answer_mode == "fallback"


def test_style_oracle_fails_closed_while_retaining_generic_components() -> None:
    query = _query("product.style_recommendation")
    card = _product_card(title="Alternative Shoe")
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            "answer: Supported alternative\n"
            "diversity_rationale: Different silhouette\n"
            "product_cards: Alternative Shoe tool-call-1-evidence-1 "
            "tool-call-1-product-1\n"
            "uncertainty: Candidate-only result."
        ),
        calls=[
            {
                "tool_name": "style_similar_search",
                "payload_kind": "product_candidates_v1",
                "payload": {
                    "candidates": [_product_candidate(title="Alternative Shoe")]
                },
                "cards": (card,),
            }
        ],
    )

    score = _score(query, result, receipt, sidecar)

    assert score.gcs == 0
    assert score.oracle_available is False
    assert score.route_acceptable == 1
    assert score.no_hard_error == 1
    assert score.output_contract_pass == 1
    assert score.tool_contract_pass == 0
    assert score.evidence_grounded == 0
    assert "style_oracle_unavailable" in score.reason_codes


def test_noskill_not_applicable_route_passes_and_wrong_skill_route_fails() -> None:
    query = _query(
        "product.exact_match",
        acceptable_capabilities=(
            "product.exact_match",
            "product.style_recommendation",
        ),
    )
    response = (
        "answer: Supported match\n"
        "product_cards: Blue Shoe tool-call-1-evidence-1 "
        "tool-call-1-product-1\n"
        "uncertainty: Candidate-only result."
    )
    call = {
        "tool_name": "image_product_search",
        "payload_kind": "product_candidates_v1",
        "payload": {"candidates": [_product_candidate()]},
        "cards": (_product_card(),),
    }
    result, receipt, sidecar = _execute(
        query, response_text=response, calls=[call], config="noskill"
    )
    noskill = _score(query, result, receipt, sidecar)
    assert noskill.gcs == 1
    assert noskill.route_disposition == "not_applicable"

    routed_result, routed_receipt, routed_sidecar = _execute(
        query,
        response_text=response,
        calls=[call],
        selected_capability="knowledge.visual_encyclopedia",
    )
    routed = _score(query, routed_result, routed_receipt, routed_sidecar)
    assert routed.gcs == 0
    assert routed.route_acceptable == 0
    assert routed.route_disposition == "fail"
    assert "route_unacceptable" in routed.reason_codes


def test_wrong_tool_order_is_rejected_even_when_sidecar_and_trace_agree() -> None:
    query = _query("knowledge.visual_encyclopedia")
    handle = "tool-call-1-source-1"
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            f"answer: Oak is a tree {handle}\n"
            f"evidence: Oak is a tree {handle}\n"
            "uncertainty: Limited."
        ),
        calls=[
            {
                "tool_name": "encyclopedia_lookup",
                "payload_kind": "knowledge_sources_v1",
                "payload": {
                    "sources": [
                        {
                            "evidence_reference": handle,
                            "title": "Oak",
                            "text": "Oak is a tree.",
                            "uri": None,
                        }
                    ]
                },
                "visible_text": f"{handle}\nOak\nOak is a tree.",
            },
            {
                "tool_name": "object_detect",
                "payload_kind": "detections_v1",
                "payload": {"detections": []},
                "visible_text": "No detections retained.",
            },
        ],
    )

    score = _score(query, result, receipt, sidecar)

    assert score.gcs == 0
    assert score.tool_contract_pass == 0
    assert "tool_contract_failed" in score.reason_codes


def test_receipt_trace_and_success_evidence_are_bound_bidirectionally() -> None:
    query = _query("product.exact_match", query_id="binding-negative")
    result, receipt, sidecar = _execute(
        query,
        response_text=(
            "answer: Supported match\n"
            "product_cards: Blue Shoe tool-call-1-evidence-1 "
            "tool-call-1-product-1\n"
            "uncertainty: Candidate-only result."
        ),
        calls=[
            {
                "tool_name": "image_product_search",
                "payload_kind": "product_candidates_v1",
                "payload": {"candidates": [_product_candidate()]},
                "cards": (_product_card(),),
            }
        ],
    )

    receipt_drift = receipt.model_copy(update={"tool_trace": ()})
    receipt_score = _score(query, result, receipt_drift, sidecar)
    assert receipt_score.gcs == 0
    assert "receipt_trace_mismatch" in receipt_score.reason_codes

    evidence_drift = result.model_copy(update={"visible_tool_evidence": ()})
    evidence_score = _score(query, evidence_drift, receipt, sidecar)
    assert evidence_score.gcs == 0
    assert "tool_evidence_mismatch" in evidence_score.reason_codes
    assert "evidence_card_union_mismatch" in evidence_score.reason_codes


def test_failed_tool_trace_remains_a_terminal_zero_score_row() -> None:
    query = _query("product.exact_match")
    failed_trace = AssistantToolTrace(
        call_index=1,
        tool_name="image_product_search",
        status="timeout",
        arguments_sha256=_argument_hash(query, "image_product_search"),
        result_sha256=None,
        runtime_binding_sha256=_digest("runtime:image_product_search"),
        latency_ms=100,
        error_code="timeout",
    )
    receipt = _receipt(query, (failed_trace,), config="llm_static")
    result = AssistantResult(
        run_id="gcs-fixture-run",
        query_id=query.query_id,
        config="llm_static",
        response_text=(
            "answer: No exact match\n"
            "product_cards: No supported match\n"
            "uncertainty: Search timed out."
        ),
        visible_cards=(),
        visible_tool_evidence=(),
        tool_trace=(failed_trace,),
        selected_capability=query.canonical_capability,
        skill_slug="fixture-skill",
        bank_sha256=_digest("fixture-bank"),
        route_trace_sha256=_digest(
            f"route:{query.query_id}:{query.canonical_capability}"
        ),
        query_artifact_sha256=_digest("query-artifact"),
        split_manifest_sha256=_digest("split-manifest"),
        registry_sha256=_digest("registry"),
        registry_runtime_sha256=_digest("registry-runtime"),
        backbone_provider="qwen",
        backbone_model="fixture-model",
        backbone_request_id=f"answer-{query.query_id}",
        usage=LLMUsage(input_tokens=11, output_tokens=7),
        latency_ms=100,
    )
    sidecar = make_public_scorer_evidence(
        query_id=query.query_id,
        config="llm_static",
        query_artifact_sha256=result.query_artifact_sha256,
        assistant_receipt_sha256=receipt.receipt_sha256,
        calls=(),
    )

    score = _score(query, result, receipt, sidecar)

    assert score.gcs == 0
    assert score.hard_error == 1
    assert score.no_hard_error == 0
    assert "assistant_hard_error" in score.reason_codes
    assert "tool_contract_failed" in score.reason_codes


def test_acceptable_boundary_alternate_route_uses_selected_oracle_but_keeps_canonical_label() -> (
    None
):
    query = _query(
        "utility.document_reading",
        query_id="gcs-boundary-alternate-route",
        acceptable_capabilities=(
            "utility.document_reading",
            "knowledge.visual_encyclopedia",
        ),
    )
    handle = "tool-call-1-source-1"
    source_text = "Oak is a hardwood tree."
    result, receipt, sidecar = _execute(
        query,
        selected_capability="knowledge.visual_encyclopedia",
        response_text=(
            f"answer: Oak is a hardwood tree {handle}\n"
            f"evidence: Oak is a hardwood tree {handle}\n"
            "uncertainty: Identification is limited to the cited source."
        ),
        calls=[
            {
                "tool_name": "encyclopedia_lookup",
                "payload_kind": "knowledge_sources_v1",
                "payload": {
                    "sources": [
                        {
                            "evidence_reference": handle,
                            "title": "Oak",
                            "text": source_text,
                            "uri": "https://example.org/oak",
                        }
                    ]
                },
                "visible_text": f"{handle}\nOak\n{source_text}",
            }
        ],
    )

    score = _score(query, result, receipt, sidecar)

    assert score.gcs == 1
    assert score.route_disposition == "pass"
    assert score.canonical_capability == "utility.document_reading"
    summary = summarize_gcs((score,), (query,), "llm_static", "boundary")
    counts = {item.capability_id: item.query_count for item in summary.capabilities}
    assert counts["utility.document_reading"] == 1
    assert counts["knowledge.visual_encyclopedia"] == 0


def test_population_uses_group_field_transitive_closure_and_hashes_metadata() -> None:
    first = _query("product.exact_match", query_id="closure-a").model_copy(
        update={
            "leakage_group_id": "leakage-a",
            "template_family": "template-bridge",
            "generator_batch_id": "batch-a",
        }
    )
    second = _query("product.multi_search", query_id="closure-b").model_copy(
        update={
            "leakage_group_id": "leakage-b",
            "template_family": "template-bridge",
            "generator_batch_id": "batch-b",
        }
    )
    third = _query("utility.document_reading", query_id="closure-c").model_copy(
        update={
            "leakage_group_id": "leakage-b",
            "template_family": "template-c",
            "generator_batch_id": "batch-c",
        }
    )
    independent = _query("utility.recipe_guidance", query_id="closure-independent")
    queries = (first, second, third, independent)

    population = build_gcs_population(queries)
    reordered = build_gcs_population(tuple(reversed(queries)))

    by_id = {item.query_id: item for item in population.bindings}
    assert population.component_count == 2
    assert (
        len(
            {
                by_id["closure-a"].component_id,
                by_id["closure-b"].component_id,
                by_id["closure-c"].component_id,
            }
        )
        == 1
    )
    assert tuple(name for name, _value in by_id["closure-a"].group_values) == (
        GROUP_FIELDS
    )
    assert reordered == population

    metadata_drift = third.model_copy(update={"template_family": "template-drift"})
    drifted = build_gcs_population((first, second, metadata_drift, independent))
    assert drifted.component_count == population.component_count
    assert drifted.population_mapping_sha256 != population.population_mapping_sha256


def test_summary_uses_fixed_six_capability_macro_and_rejects_nonrectangular_rows() -> (
    None
):
    queries = tuple(
        [
            _query("product.exact_match", query_id="summary-exact-a"),
            _query("product.exact_match", query_id="summary-exact-b"),
        ]
        + [
            _query(capability, query_id=f"summary-{index}")
            for index, capability in enumerate(GCS_CAPABILITY_ORDER)
            if capability != "product.exact_match"
        ]
    )
    successful = {
        "summary-exact-a",
        "summary-exact-b",
    }
    scores = _manual_scores(
        queries,
        config="llm_static",
        successful_query_ids=successful,
    )

    summary = summarize_gcs(scores, queries, "llm_static", "summary-fixture")

    assert tuple(item.capability_id for item in summary.capabilities) == (
        GCS_CAPABILITY_ORDER
    )
    assert summary.query_micro_rate == pytest.approx(2 / 7)
    assert summary.headline_macro_rate == pytest.approx(1 / 6)
    assert summary.query_micro_rate != summary.headline_macro_rate

    with pytest.raises(PortfolioGCSError, match="duplicate terminal rows"):
        summarize_gcs((*scores, scores[0]), queries, "llm_static", "summary-fixture")
    with pytest.raises(
        PortfolioGCSError, match="not a rectangular terminal population"
    ):
        summarize_gcs(scores[:-1], queries, "llm_static", "summary-fixture")
    drifted = scores[0].model_copy(update={"component_id": _digest("wrong-component")})
    with pytest.raises(PortfolioGCSError, match="metadata drifted"):
        summarize_gcs(
            (drifted, *scores[1:]),
            queries,
            "llm_static",
            "summary-fixture",
        )


def test_component_bootstrap_is_reproducible_reorder_invariant_and_sign_symmetric() -> (
    None
):
    queries = _component_queries(2, prefix="bootstrap")
    first_component = {
        query.query_id
        for query in queries
        if query.leakage_group_id == "bootstrap-leakage-000"
    }
    baseline = _manual_scores(
        queries,
        config="llm_static",
        successful_query_ids=set(),
    )
    treatment = _manual_scores(
        queries,
        config="full",
        successful_query_ids=first_component,
    )

    forward = paired_component_bootstrap(
        baseline, treatment, queries, "bootstrap-fixture"
    )
    repeated = paired_component_bootstrap(
        baseline, treatment, queries, "bootstrap-fixture"
    )
    reordered = paired_component_bootstrap(
        tuple(reversed(baseline)),
        tuple(reversed(treatment)),
        tuple(reversed(queries)),
        "bootstrap-fixture",
    )
    reverse = paired_component_bootstrap(
        treatment, baseline, queries, "bootstrap-fixture"
    )

    assert forward.model_dump_json() == repeated.model_dump_json()
    assert forward.model_dump_json() == reordered.model_dump_json()
    assert forward.draw_stream_sha256 == reverse.draw_stream_sha256
    assert forward.derived_seed == reverse.derived_seed
    assert reverse.macro_delta_pp == -forward.macro_delta_pp
    assert reverse.query_micro_delta_pp == -forward.query_micro_delta_pp
    assert reverse.hard_error_delta_pp == -forward.hard_error_delta_pp
    assert reverse.macro_ci95 is not None
    assert forward.macro_ci95 is not None
    assert reverse.macro_ci95.low_pp == -forward.macro_ci95.high_pp
    assert reverse.macro_ci95.high_pp == -forward.macro_ci95.low_pp
    for forward_capability, reverse_capability in zip(
        forward.per_capability, reverse.per_capability, strict=True
    ):
        assert reverse_capability.point_delta_pp == (-forward_capability.point_delta_pp)
        assert forward_capability.ci95 is not None
        assert reverse_capability.ci95 is not None
        assert reverse_capability.ci95.low_pp == -forward_capability.ci95.high_pp
        assert reverse_capability.ci95.high_pp == -forward_capability.ci95.low_pp

    alias = _manual_scores(
        queries,
        config="full",
        successful_query_ids=set(),
    )
    alias_bootstrap = paired_component_bootstrap(
        baseline, alias, queries, "bootstrap-alias"
    )
    assert alias_bootstrap.zero_variance is True
    assert alias_bootstrap.macro_ci95 == GCSBootstrapInterval(low_pp=0.0, high_pp=0.0)
    assert alias_bootstrap.query_micro_ci95 == GCSBootstrapInterval(
        low_pp=0.0, high_pp=0.0
    )
    assert all(
        item.ci95 == GCSBootstrapInterval(low_pp=0.0, high_pp=0.0)
        for item in alias_bootstrap.per_capability
    )


@pytest.mark.parametrize(
    ("query_count", "expected_components", "prefix"),
    (
        (75, 3, "selection-75"),
        (75, 3, "replay-75"),
        (50, 2, "holdout-50"),
        (300, 12, "test-300"),
    ),
)
def test_expected_25_query_component_counts(
    query_count: int, expected_components: int, prefix: str
) -> None:
    capabilities = tuple(
        GCS_CAPABILITY_ORDER[index % len(GCS_CAPABILITY_ORDER)] for index in range(25)
    )
    queries = _component_queries(
        expected_components,
        capabilities_per_component=capabilities,
        prefix=prefix,
    )

    population = build_gcs_population(queries)

    assert len(queries) == query_count
    assert population.query_count == query_count
    assert population.component_count == expected_components
    assert sorted(size for _component, size in population.component_sizes) == (
        [25] * expected_components
    )


def test_style_unavailable_makes_system_gain_gate_unavailable() -> None:
    queries = _component_queries(1, prefix="style-gate")
    style_ids = {
        query.query_id
        for query in queries
        if query.canonical_capability == "product.style_recommendation"
    }
    successes = {query.query_id for query in queries} - style_ids
    baseline = _manual_scores(
        queries,
        config="llm_static",
        successful_query_ids=successes,
        unavailable_query_ids=style_ids,
    )
    treatment = _manual_scores(
        queries,
        config="full",
        successful_query_ids=successes,
        unavailable_query_ids=style_ids,
    )
    summary = summarize_gcs(treatment, queries, "full", "style-gate")
    bootstrap = paired_component_bootstrap(baseline, treatment, queries, "style-gate")

    decision = evaluate_gcs_system_gain(summary, bootstrap)

    assert summary.headline_available is False
    assert bootstrap.baseline_oracle_coverage_complete is False
    assert bootstrap.treatment_oracle_coverage_complete is False
    assert decision.status == "unavailable"
    assert "baseline_oracle_coverage_incomplete" in decision.reason_codes
    assert "treatment_oracle_coverage_incomplete" in decision.reason_codes


def _threshold_gate_fixture() -> tuple[GCSConfigSummary, PairedGCSBootstrap]:
    queries = _component_queries(2, prefix="threshold-gate")
    treatment = _manual_scores(
        queries,
        config="full",
        successful_query_ids={query.query_id for query in queries},
    )
    summary = summarize_gcs(treatment, queries, "full", "threshold-gate")
    capability_points = (-3.0, -3.0, -3.0, -3.0, -3.0, 27.0)
    bootstrap = PairedGCSBootstrap(
        baseline_config="llm_static",
        treatment_config="full",
        scope=summary.scope,
        population_mapping_sha256=summary.population_mapping_sha256,
        query_count=summary.query_count,
        component_count=summary.component_count,
        component_size_histogram=summary.component_size_histogram,
        derived_seed=1,
        draw_stream_sha256=_digest("threshold-draws"),
        macro_delta_pp=2.0,
        macro_ci95=GCSBootstrapInterval(low_pp=0.0, high_pp=4.0),
        macro_available_replicates=GCS_BOOTSTRAP_REPLICATES,
        query_micro_delta_pp=2.0,
        query_micro_ci95=GCSBootstrapInterval(low_pp=0.0, high_pp=4.0),
        hard_error_delta_pp=1.0,
        hard_error_ci95=GCSBootstrapInterval(low_pp=0.0, high_pp=1.0),
        per_capability=tuple(
            GCSCapabilityBootstrapDelta(
                capability_id=capability,
                point_delta_pp=point,
                ci95=GCSBootstrapInterval(low_pp=point, high_pp=point),
                available_replicates=GCS_BOOTSTRAP_REPLICATES,
            )
            for capability, point in zip(
                GCS_CAPABILITY_ORDER, capability_points, strict=True
            )
        ),
        baseline_oracle_coverage_complete=True,
        treatment_oracle_coverage_complete=True,
        zero_variance=False,
        descriptive_only=False,
        precision_warnings=("low_component_count_precision",),
    )
    return summary, bootstrap


def test_system_gain_thresholds_pass_exactly_on_every_boundary() -> None:
    summary, bootstrap = _threshold_gate_fixture()

    decision = evaluate_gcs_system_gain(summary, bootstrap)

    assert decision.status == "passed"
    assert all(item.passed is True for item in decision.criteria)

    missing_baseline_coverage = bootstrap.model_copy(
        update={"baseline_oracle_coverage_complete": False}
    )
    unavailable = evaluate_gcs_system_gain(summary, missing_baseline_coverage)
    assert unavailable.status == "unavailable"
    assert "baseline_oracle_coverage_incomplete" in unavailable.reason_codes


@pytest.mark.parametrize(
    ("mutation", "failed_name"),
    (
        ("macro", "macro_delta_pp"),
        ("macro_ci", "macro_ci95_low_pp"),
        ("hard_error", "hard_error_delta_pp"),
        (
            "capability",
            f"capability_delta_pp:{GCS_CAPABILITY_ORDER[0]}",
        ),
    ),
)
def test_system_gain_fails_when_any_single_threshold_is_crossed(
    mutation: str, failed_name: str
) -> None:
    summary, bootstrap = _threshold_gate_fixture()
    payload = bootstrap.model_dump(mode="json")
    if mutation == "macro":
        payload["macro_delta_pp"] = 1.999
    elif mutation == "macro_ci":
        payload["macro_ci95"] = {"low_pp": -0.001, "high_pp": 4.0}
    elif mutation == "hard_error":
        payload["hard_error_delta_pp"] = 1.001
    else:
        payload["per_capability"][0]["point_delta_pp"] = -3.001
    crossed = PairedGCSBootstrap.model_validate(payload, strict=True)

    decision = evaluate_gcs_system_gain(summary, crossed)

    assert decision.status == "failed"
    assert f"threshold_failed:{failed_name}" in decision.reason_codes


def _style_trace_v2(
    query: Query,
    *,
    submode: str,
    candidate_category: str | None,
    unsupported: bool = False,
) -> ProductSearchTrace:
    binding = RetrievalArtifactBinding(mode="provisional")
    hits: tuple[StyleHit, ...] = ()
    if candidate_category is not None:
        cross = submode == "cross_category_coordination"
        hits = (
            StyleHit(
                rank=1,
                score=0.91,
                product=Product(
                    product_id="private-product-001",
                    title="Blue Loafer",
                    category_l1=candidate_category,
                    image_path="private/catalog/001.jpg",
                    source="fixture",
                ),
                artifact_binding=binding,
                mmr_score=0.88,
                anchor_category_l1="apparel" if cross else candidate_category,
                style_submode=submode,
                similarity_source=(
                    "verified_coordination_graph"
                    if cross
                    else "verified_embedding_cosine"
                ),
                facet_evidence=(
                    StyleFacetEvidence(
                        facet="color",
                        value="blue",
                        confidence=0.93,
                        provenance=(
                            "portfolio_curated_coordination_rule"
                            if cross
                            else "verified_embedding"
                        ),
                        source_record_id="private-style-source-001",
                    ),
                ),
            ),
        )
    return ProductSearchTrace(
        tool_name="style_similar_search",
        input_kind="image",
        query_input_sha256=_digest(f"image:{query.query_id}"),
        query_image_sha256=_digest(f"image:{query.query_id}"),
        query_asset_id=query.asset_id,
        query_vector_sha256=_digest(f"vector:{query.query_id}"),
        artifact_binding=binding,
        hits=hits,
        style_submode=submode,
        unsupported_reason=("unsupported target" if unsupported else None),
    )


def _style_v2_execution(
    query: Query,
    trace_output: ProductSearchTrace,
) -> tuple[AssistantResult, AssistantExecutionReceipt, PublicScorerEvidenceV2]:
    arguments = {"asset_id": query.asset_id, "query": query.text}
    arguments_sha256 = _hash_json(arguments)
    result_sha256 = _hash_json(trace_output)
    scorer_call = build_public_scorer_call_v2(
        query=query,
        call_index=1,
        tool_name="style_similar_search",
        bound_arguments=arguments,
        arguments_sha256=arguments_sha256,
        raw_validated_output=trace_output.model_dump(mode="json"),
        result_sha256=result_sha256,
    )
    style_payload = scorer_call.payload
    candidates = style_payload["candidates"]
    assert isinstance(candidates, list)
    cards: tuple[VisibleCard, ...]
    if candidates:
        candidate = candidates[0]
        assert isinstance(candidate, dict)
        facets = candidate["style_evidence"]
        assert isinstance(facets, list) and isinstance(facets[0], dict)
        facet = facets[0]
        fields = (
            ("evidence_reference", str(candidate["evidence_reference"])),
            ("product_id", str(candidate["product_id"])),
            ("similarity_source", str(candidate["similarity_source"])),
            ("style_evidence_1", str(facet["evidence_reference"])),
            ("style_submode", str(candidate["style_submode"])),
        )
        if candidate["style_submode"] == "cross_category_coordination":
            fields = tuple(item for item in fields if item[0] != "similarity_source")
        card = VisibleCard(
            title=str(candidate["title"]),
            body=(f"{candidate['category']} · {facet['facet']}: {facet['value']}"),
            fields=tuple(sorted(fields)),
        )
        cards = (card,)
        response_text = (
            "answer: Evidence-backed Style candidate\n"
            f"diversity_rationale: {facet['evidence_reference']} "
            f"{facet['facet']} {facet['value']}\n"
            f"product_cards: {candidate['title']} "
            f"{candidate['evidence_reference']} {candidate['product_id']}\n"
            "uncertainty: Candidate-only evidence."
        )
        visible_text = "One evidence-backed Style candidate."
    else:
        cards = ()
        response_text = (
            "answer: No supported alternative\n"
            "diversity_rationale: Unable to recommend from verified evidence\n"
            "product_cards: No supported alternative\n"
            "uncertainty: Unable to recommend without supported evidence."
        )
        visible_text = "No supported Style candidate."
    tool_trace = AssistantToolTrace(
        call_index=1,
        tool_name="style_similar_search",
        status="success",
        arguments_sha256=arguments_sha256,
        result_sha256=result_sha256,
        runtime_binding_sha256=_digest("runtime:style_similar_search:v2"),
        latency_ms=2,
    )
    receipt = _receipt(query, (tool_trace,), config="llm_static")
    result = AssistantResult(
        run_id="gcs-v2-style-fixture",
        query_id=query.query_id,
        config="llm_static",
        response_text=response_text,
        visible_cards=cards,
        visible_tool_evidence=(
            VisibleToolEvidence(
                tool_name="style_similar_search",
                status="success",
                visible_text=visible_text,
                cards=cards,
            ),
        ),
        tool_trace=(tool_trace,),
        selected_capability=query.canonical_capability,
        skill_slug="fixture-skill",
        bank_sha256=_digest("fixture-bank"),
        route_trace_sha256=_digest(
            f"route:{query.query_id}:{query.canonical_capability}"
        ),
        query_artifact_sha256=_digest("query-artifact"),
        split_manifest_sha256=_digest("split-manifest"),
        registry_sha256=_digest("registry"),
        registry_runtime_sha256=_digest("registry-runtime"),
        backbone_provider="qwen",
        backbone_model="fixture-model",
        backbone_request_id=f"answer-{query.query_id}",
        usage=LLMUsage(input_tokens=11, output_tokens=7),
        latency_ms=9,
    )
    sidecar = make_public_scorer_evidence_v2(
        matrix_run_id="gcs-v2-fixture-matrix",
        instance_id=f"instance:{query.query_id}:llm_static",
        request_sha256=receipt.request_sha256,
        query_id=query.query_id,
        config="llm_static",
        query_artifact_sha256=result.query_artifact_sha256,
        assistant_result_sha256=_hash_json(result),
        assistant_receipt_sha256=receipt.receipt_sha256,
        calls=(scorer_call,),
    )
    return result, receipt, sidecar


def _score_v2(
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    sidecar: PublicScorerEvidenceV2,
) -> GCSQueryScoreV2:
    return score_portfolio_gcs_v2(
        query,
        result,
        receipt,
        sidecar,
        _TASK_SPEC,
        portfolio_gcs_oracles_v2(),
        population=build_gcs_population_v2((query,)),
    )


def test_gcs_v2_policy_is_distinct_and_matches_the_tracked_spec() -> None:
    import json
    from pathlib import Path

    assert GCS_POLICY_SHA256 == (
        "826cc089771fa33f0aef962fbc60b337f30b3bd13c08a4c6b67108bc5284bcfe"
    )
    assert GCS_V2_POLICY_VERSION != GCS_POLICY_VERSION
    assert GCS_V2_POLICY_SHA256 == (
        "ccb838617c857719e8a864f1638b3aa01f2e2aacbb7344629bcd9087b9e20651"
    )
    assert GCS_V2_POLICY_SHA256 != GCS_POLICY_SHA256
    tracked = json.loads(
        Path("specs/evaluation/portfolio-gcs-v2.json").read_text(encoding="utf-8")
    )
    assert tracked == gcs_v2_policy_payload()
    assert _hash_json(tracked) == GCS_V2_POLICY_SHA256


def test_gcs_v2_empty_multi_payload_keeps_call_without_empty_expected_set() -> None:
    query = _query("product.multi_search", query_id="empty-multi")
    payload: dict[str, JSONValue] = {"items": [], "candidates": []}
    result, receipt, _ = _execute(
        query,
        response_text=(
            "answer: No detectable requested items.\n"
            "item_mapping: No supported mapping.\n"
            "product_cards: No supported item.\n"
            "uncertainty: Unable to identify supported items."
        ),
        calls=(
            {
                "tool_name": "multi_product_search",
                "payload_kind": "multi_mapping_v1",
                "payload": payload,
                "visible_text": "No detectable items.",
            },
        ),
    )
    trace = result.tool_trace[0]
    call = PublicScorerCallEvidenceV2.model_validate(
        {
            "call_index": 1,
            "tool_name": "multi_product_search",
            "arguments_sha256": trace.arguments_sha256,
            "result_sha256": trace.result_sha256,
            "argument_projection": {"asset_handle": "query_asset"},
            "payload_kind": "multi_mapping_v1",
            "payload": payload,
            "payload_sha256": _hash_json(payload),
        },
        strict=True,
    )

    expected = expected_multi_items_from_call_v2(call)
    sidecar = make_public_scorer_evidence_v2(
        matrix_run_id="gcs-v2-empty-multi-fixture",
        instance_id="instance:empty-multi:llm_static",
        request_sha256=receipt.request_sha256,
        query_id=query.query_id,
        config="llm_static",
        query_artifact_sha256=result.query_artifact_sha256,
        assistant_result_sha256=_hash_json(result),
        assistant_receipt_sha256=receipt.receipt_sha256,
        calls=(call,),
        expected_multi_items=expected,
    )
    score = _score_v2(query, result, receipt, sidecar)

    assert expected is None
    assert sidecar.expected_multi_items is None
    assert sidecar.calls == (call,)
    assert score.gcs == 0
    assert score.answer_mode == "unresolved"
    assert "requested_item_coverage_unresolved" in score.reason_codes

    invalid = sidecar.model_dump(mode="json")
    invalid["expected_multi_items"] = []
    invalid["evidence_sha256"] = _hash_json(
        {key: value for key, value in invalid.items() if key != "evidence_sha256"}
    )
    with pytest.raises(ValidationError, match="non-empty and contiguous"):
        PublicScorerEvidenceV2.model_validate(invalid, strict=True)


def test_gcs_v2_same_category_style_closes_cards_facets_and_handles() -> None:
    query = _query("product.style_recommendation", query_id="style-v2-same")
    trace_output = _style_trace_v2(
        query,
        submode="same_category_alternative",
        candidate_category="footwear",
    )
    result, receipt, sidecar = _style_v2_execution(query, trace_output)

    score = _score_v2(query, result, receipt, sidecar)

    assert score.gcs == 1
    assert score.oracle_available is True
    assert score.style_support_status == "candidates"
    serialized = sidecar.model_dump(mode="json")
    assert "private-product-001" not in str(serialized)
    assert "private-style-source-001" not in str(serialized)
    assert "private/catalog" not in str(serialized)

    tampered = sidecar.model_dump(mode="json")
    tampered["calls"][0]["payload"]["candidates"][0]["category"] = "apparel"
    tampered["calls"][0]["payload_sha256"] = _hash_json(tampered["calls"][0]["payload"])
    tampered["evidence_sha256"] = _hash_json(
        {key: value for key, value in tampered.items() if key != "evidence_sha256"}
    )
    changed = PublicScorerEvidenceV2.model_validate(tampered, strict=True)
    changed_score = _score_v2(query, result, receipt, changed)
    assert changed_score.gcs == 0
    assert "style_mode_mismatch" in changed_score.reason_codes


def test_gcs_v2_cross_category_unsupported_fallback_is_a_valid_row() -> None:
    query = _query(
        "product.style_recommendation", query_id="style-v2-cross-fallback"
    ).model_copy(
        update={
            "text": "Coordinate this dress with footwear",
            "turns": [
                {"role": "user", "content": "Coordinate this dress with footwear"}
            ],
        }
    )
    trace_output = _style_trace_v2(
        query,
        submode="cross_category_coordination",
        candidate_category=None,
        unsupported=True,
    )
    result, receipt, sidecar = _style_v2_execution(query, trace_output)

    score = _score_v2(query, result, receipt, sidecar)

    assert score.gcs == 1
    assert score.answer_mode == "fallback"
    assert score.style_support_status == "unsupported"


def test_gcs_v2_cross_category_candidate_must_cover_the_requested_family() -> None:
    query = _query(
        "product.style_recommendation", query_id="style-v2-cross-supported"
    ).model_copy(
        update={
            "text": "Coordinate this dress with footwear",
            "turns": [
                {"role": "user", "content": "Coordinate this dress with footwear"}
            ],
        }
    )
    result, receipt, sidecar = _style_v2_execution(
        query,
        _style_trace_v2(
            query,
            submode="cross_category_coordination",
            candidate_category="footwear",
        ),
    )

    score = _score_v2(query, result, receipt, sidecar)

    assert score.gcs == 1
    assert score.style_support_status == "candidates"

    tampered = sidecar.model_dump(mode="json")
    tampered["calls"][0]["payload"]["candidates"][0]["category"] = "bag"
    tampered["calls"][0]["payload_sha256"] = _hash_json(tampered["calls"][0]["payload"])
    tampered["evidence_sha256"] = _hash_json(
        {key: value for key, value in tampered.items() if key != "evidence_sha256"}
    )
    changed = PublicScorerEvidenceV2.model_validate(tampered, strict=True)
    changed_score = _score_v2(query, result, receipt, changed)
    assert changed_score.gcs == 0
    assert "style_target_family_mismatch" in changed_score.reason_codes


def test_gcs_v2_sidecar_binding_failures_are_fatal_not_row_zero() -> None:
    query = _query("product.style_recommendation", query_id="style-v2-fatal")
    result, receipt, sidecar = _style_v2_execution(
        query,
        _style_trace_v2(
            query,
            submode="same_category_alternative",
            candidate_category="footwear",
        ),
    )
    with pytest.raises(PortfolioGCSIntegrityError, match="invalid v2"):
        _score_v2(query, result, receipt, None)  # type: ignore[arg-type]

    tampered = sidecar.model_dump(mode="json")
    tampered["assistant_result_sha256"] = _digest("different-result")
    tampered["evidence_sha256"] = _hash_json(
        {key: value for key, value in tampered.items() if key != "evidence_sha256"}
    )
    rebound = PublicScorerEvidenceV2.model_validate(tampered, strict=True)
    with pytest.raises(PortfolioGCSIntegrityError, match="differs"):
        _score_v2(query, result, receipt, rebound)


def test_gcs_v2_summary_keeps_semantic_failures_in_an_available_headline() -> None:
    queries = tuple(
        _query(capability, query_id=f"v2-summary-{index}")
        for index, capability in enumerate(GCS_CAPABILITY_ORDER)
    )
    population = build_gcs_population_v2(queries)
    bindings = {item.query_id: item for item in population.bindings}
    rows = tuple(
        GCSQueryScoreV2(
            query_id=query.query_id,
            config="llm_static",
            canonical_capability=query.canonical_capability,
            evaluated_capability=query.canonical_capability,
            component_id=bindings[query.query_id].component_id,
            route_disposition="pass",
            answer_mode="unresolved",
            oracle_available=True,
            semantic_claim_support_resolved=False,
            route_acceptable=1,
            no_hard_error=0,
            tool_contract_pass=0,
            evidence_grounded=0,
            output_contract_pass=0,
            hard_error=1,
            gcs=0,
            style_support_status=(
                "unresolved"
                if query.canonical_capability == "product.style_recommendation"
                else None
            ),
            reason_codes=("assistant_hard_error",),
        )
        for query in queries
    )

    summary = summarize_gcs_v2(rows, queries, "llm_static", "v2-row-zero")

    assert summary.headline_available is True
    assert summary.oracle_coverage_complete is True
    assert summary.headline_macro_rate == 0.0
    assert dict(summary.style_support_status_counts)["unresolved"] == 1


def test_gcs_v2_ocr_projection_reuses_public_line_handles_without_private_ids() -> None:
    line = OCRLine(
        line_id=_digest("private-ocr-line"),
        text="Total $12.50",
        polygon=((1.0, 1.0), (30.0, 1.0), (30.0, 8.0), (1.0, 8.0)),
        confidence=0.95,
    )
    result = DocumentOCRResult(
        input_binding=OCRInputBinding(
            asset_id="asset.private.ocr",
            image_sha256=_digest("ocr-image"),
            image_bytes=128,
            width=64,
            height=32,
            safety_decision="approved_no_pii",
            safety_approval_sha256=_digest("ocr-approval"),
        ),
        runtime_binding=ModelRuntimeBinding(
            artifact_kind="document_ocr",
            model_id="fixture-ocr",
            backend_name="fixture",
            backend_version="1.0",
            manifest_sha256=_digest("ocr-manifest"),
            artifacts=(),
        ),
        languages=("en",),
        full_text=line.text,
        lines=(line,),
        fields=(
            OCRField(
                field_id=_digest("private-ocr-field"),
                field_name="total",
                value="$12.50",
                evidence_line_ids=(line.line_id,),
            ),
        ),
        truncated=False,
    )

    projected = project_document_ocr_lines_v2(
        result.model_dump(mode="json"), call_index=3
    )

    assert projected == {
        "lines": [
            {
                "line_reference": "tool-call-3-line-1",
                "text": "Total $12.50",
                "content_trust": "untrusted_document_text",
                "truncated": False,
                "fields": [["total", "$12.50"]],
            }
        ]
    }
    serialized = str(projected)
    assert line.line_id not in serialized
    assert result.input_binding.asset_id not in serialized
    assert result.runtime_binding.manifest_sha256 not in serialized
