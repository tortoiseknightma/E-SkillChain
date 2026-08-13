from __future__ import annotations

import base64
from dataclasses import dataclass
from decimal import Decimal
import json

import httpx
from openai import APIStatusError
import pytest

from skillchain import config as project_config
from skillchain.evaluation.assistant_runs import (
    AssistantBackendResponse,
    AssistantRequestSnapshot,
    AssistantRouteCallEvidence,
    MatrixTreatment,
    _registry_lock,
    build_assistant_query_input,
    make_backbone_lock,
    make_inference_budget,
)
from skillchain.evaluation.portfolio_execution import (
    PORTFOLIO_QWEN_MODEL,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
    PortfolioBudgetError,
    PortfolioBudgetExceededError,
    initialize_portfolio_budget_ledger,
    load_portfolio_budget_ledger,
)
from skillchain.evaluation.portfolio_gcs_evidence import (
    PublicScorerCallEvidenceV2,
    PublicScorerEvidenceIntegrityError,
)
from skillchain.evolution.s1_gcs_gate import (
    make_s1_round2_response_contract_diagnostic,
)
from skillchain.llm import LLMResponse, LLMTimeoutError, LLMToolCall, LLMUsage
from skillchain.runners.assistant import (
    NOSKILL_EXECUTION_CONTRACT_SHA256,
    NOSKILL_EXECUTION_POLICY_VERSION,
    PORTFOLIO_ROUTER_CONTRACT_SHA256,
    PORTFOLIO_ROUTER_CONTRACT_VERSION,
    SHARED_STAGE2_ROUTE_POLICY_VERSION,
    AssistantFatalProviderConfigurationError,
    AssistantProviderCallGateError,
    CoreFastAssistantRunner,
    PortfolioAssistantBudgetContext,
    PortfolioAssistantRunner,
    ProductionAssistantRunner,
    SharedStage2RouteArtifact,
    assistant_route_output_json_schema,
    assistant_router_contract_payload,
    evolution_bank_boundary_violations,
    noskill_execution_contract_payload,
)
from skillchain.runners.assistant_deterministic_contract import (
    DeterministicSemanticPolicy,
    render_deterministic_semantic_policy,
)
from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.tools.registry import ToolInvocationResult
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


_SYSTEM_PROMPT = "Frozen Assistant action-contract prompt."
_MODEL = PORTFOLIO_QWEN_MODEL


@dataclass(frozen=True)
class _Skill:
    slug: str
    capability_id: str
    description: str
    body: str
    operators: tuple[str, ...]
    static_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Binding:
    capability_id: str
    skill_slug: str


@dataclass(frozen=True)
class _Bank:
    bank_sha256: str
    skills: tuple[_Skill, ...]
    capability_map: tuple[_Binding, ...]


def _one_skill_bank(skill: _Skill, digest_character: str) -> _Bank:
    return _Bank(
        bank_sha256=digest_character * 64,
        skills=(skill,),
        capability_map=(
            _Binding(
                capability_id=skill.capability_id,
                skill_slug=skill.slug,
            ),
        ),
    )


def test_evolution_boundaries_ignore_lineage_metadata_and_reject_content_drift() -> (
    None
):
    capability = "utility.document_reading"
    s1 = _Skill(
        slug="s1-document",
        capability_id=capability,
        description="Initial document route.",
        body="# Body\n\nRead the document.\n",
        operators=("document_ocr",),
    )
    s1s2 = _Skill(
        slug="s1s2-document",
        capability_id=capability,
        description="Route visible-text extraction requests here.",
        body=s1.body,
        operators=s1.operators,
    )
    full = _Skill(
        slug="full-document",
        capability_id=capability,
        description=s1s2.description,
        body="# Body\n\nRead the document and ground every claim.\n",
        operators=s1s2.operators,
    )
    valid = {
        "s1": _one_skill_bank(s1, "a"),
        "s1s2": _one_skill_bank(s1s2, "b"),
        "full": _one_skill_bank(full, "c"),
    }

    assert evolution_bank_boundary_violations(valid) == ()

    invalid_s2 = _Skill(
        slug=s1s2.slug,
        capability_id=capability,
        description=s1s2.description,
        body="# Body\n\nS2 illegally changed execution.\n",
        operators=s1s2.operators,
    )
    invalid_full = _Skill(
        slug=full.slug,
        capability_id=capability,
        description="S3 illegally changed routing.",
        body=full.body,
        operators=full.operators,
    )
    violations = evolution_bank_boundary_violations(
        {
            "s1": valid["s1"],
            "s1s2": _one_skill_bank(invalid_s2, "d"),
            "full": _one_skill_bank(invalid_full, "e"),
        }
    )

    assert violations == (
        "S2 may only change Description; utility.document_reading changed body",
        "S3 may only change Body; utility.document_reading changed description",
    )


def test_noskill_execution_contract_has_stable_content_identity() -> None:
    payload = noskill_execution_contract_payload()

    assert (
        payload["policy_version"]
        == NOSKILL_EXECUTION_POLICY_VERSION
        == "noskill-native-function-calling-v2"
    )
    assert payload["model_tool_projection_policy"] == (
        "runner-authoritative-text-binding-v1"
    )
    assert payload["text_product_search_model_arguments"] == "forbidden"
    assert payload["text_product_search_argument_source"] == "frozen_query_text"
    assert (
        sha256_bytes(canonical_json_bytes(payload)) == NOSKILL_EXECUTION_CONTRACT_SHA256
    )


@pytest.mark.parametrize(
    "config",
    ("noskill", "llm_static", "s1", "s1s2", "full"),
)
def test_action_prompt_makes_runtime_response_contract_mandatory_for_all_configs(
    canonical_registry_factory,
    config: str,
) -> None:
    fixture = canonical_registry_factory(name=f"response-contract-{config}")
    selected_slug = None
    bank_sha256 = None
    banks = {}
    if config != "noskill":
        skill = _Skill(
            slug=f"{config}-exact",
            capability_id="product.exact_match",
            description="Find the exact product.",
            body="# Output contract\n\nUse the frozen response contract.",
            operators=("image_product_search",),
        )
        bank = _one_skill_bank(skill, "a")
        banks[config] = bank
        bank_sha256 = bank.bank_sha256
        selected_slug = skill.slug
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config=config,
        bank_sha256=bank_sha256,
    )

    prompt = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        banks=banks,
    )._action_prompt_for(request, selected_skill_slug=selected_slug)

    assert "answer the user directly in plain text" in prompt
    assert "runner-authored final_response_contract" in prompt
    assert "mandatory runtime response policy, not as retrieved evidence" in prompt
    assert (
        "common_rules, required_sections, supported_rules, and fallback_rule" in prompt
    )
    assert "do not translate, rename, omit, reorder, or duplicate" in prompt
    assert "answer the user directly in natural language" not in prompt


def test_router_contract_has_stable_v6_content_identity() -> None:
    payload = assistant_router_contract_payload()

    assert (
        payload["policy_version"]
        == PORTFOLIO_ROUTER_CONTRACT_VERSION
        == "portfolio-assistant-router-v6"
    )
    assert payload["route_user_input_fields"] == ["turns"]
    assert payload["route_image_attachment"] is False
    assert payload["route_request_max_output_tokens"] == 64
    assert payload["route_pricing_reservation_max_output_tokens"] == 512
    assert payload["parser"] == "full_strict_json_no_prefix_recovery"
    assert payload["unknown_top_level_response_key_policy"] == "reject"
    assert payload["fixed_repair_eligible_failures"] == [
        "length",
        "empty",
        "invalid_json",
        "non_object",
        "unexpected_keys",
        "schema_invalid",
    ]
    assert payload["fixed_repair_eligible_failure_subtypes"] == [
        "invalid_route_json",
        "length",
        "response_empty_text",
    ]
    assert payload["fixed_repair_eligible_payload_statuses"] == [
        "invalid_json",
        "non_object",
        "schema_invalid",
        "unexpected_keys",
    ]
    assert payload["repair_previous_response_input"] == "forbidden"
    assert (
        sha256_bytes(canonical_json_bytes(payload)) == PORTFOLIO_ROUTER_CONTRACT_SHA256
    )


def _query_for_catalog(catalog, *, query_id: str) -> Query:
    asset = next(
        item for item in catalog.assets if item.source_record_id == "query-only"
    )
    capability = "utility.document_reading"
    return Query(
        schema_version=2,
        taxonomy_version="runner-action-contract-v1",
        task_spec_version="runner-action-contract-v1",
        query_id=query_id,
        asset_id=asset.asset_id,
        image_path=asset.local_path,
        leakage_group_id=catalog.component_for_asset(asset.asset_id),
        template_family="runner-action-contract",
        generator_batch_id="runner-action-contract",
        text="请看一下这张图片。",
        turns=(ConversationTurn(role="user", content="请看一下这张图片。"),),
        canonical_intent="utility",
        canonical_capability=capability,
        acceptable_capabilities=(capability,),
        requires_card=False,
        split="dev_mini",
        label_status="auto",
        label_provenance=(
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="runner-action-contract",
                canonical_intent="utility",
                canonical_capability=capability,
                acceptable_capabilities=(capability,),
            ),
        ),
    )


def _request(
    *,
    registry,
    catalog,
    config: str,
    max_output_tokens: int = 64,
    max_turns: int = 4,
    bank_sha256: str | None = None,
    query_id: str | None = None,
    matrix_run_id: str = "runner-action-contract",
) -> AssistantRequestSnapshot:
    query = build_assistant_query_input(
        _query_for_catalog(catalog, query_id=query_id or f"action-contract-{config}")
    )
    if config == "noskill":
        treatment = MatrixTreatment(
            config="noskill",
            bank_sha256=None,
            router_stage="disabled",
            body_stage="disabled",
        )
    else:
        assert bank_sha256 is not None
        stages = {
            "llm_static": ("llm_static", "llm_static"),
            "s1": ("s1", "s1"),
            "s1s2": ("s2", "s1"),
            "full": ("s2", "s3"),
        }
        router_stage, body_stage = stages[config]
        treatment = MatrixTreatment(
            config=config,
            bank_sha256=bank_sha256,
            router_stage=router_stage,
            body_stage=body_stage,
        )
    backbone = make_backbone_lock(
        provider="qwen",
        model=_MODEL,
        endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
        temperature=0.0,
        top_p=1.0,
        seed=7,
        system_prompt_sha256=sha256_bytes(_SYSTEM_PROMPT.encode("utf-8")),
    )
    budget = make_inference_budget(
        max_input_tokens=2_000,
        max_output_tokens=max_output_tokens,
        max_tool_calls=2,
        max_turns=max_turns,
        timeout_ms=30_000,
    )
    registry_lock = _registry_lock(registry)
    unsigned = {
        "schema_version": 1,
        "matrix_run_id": matrix_run_id,
        "config": config,
        "query_ordinal": 0,
        "query": query,
        "treatment": treatment,
        "backbone": backbone,
        "budget": budget,
        "registry": registry_lock,
    }
    hash_payload = {
        key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        for key, value in unsigned.items()
    }
    return AssistantRequestSnapshot.model_validate(
        {
            **unsigned,
            "request_sha256": sha256_bytes(canonical_json_bytes(hash_payload)),
        },
        strict=True,
    )


def _runner(
    *,
    registry,
    catalog,
    runner_type=ProductionAssistantRunner,
    banks: dict[str, _Bank] | None = None,
    qwen_call_start_waiter=None,
):
    runner = object.__new__(runner_type)
    object.__setattr__(runner, "_registry", registry)
    object.__setattr__(runner, "_system_prompt", _SYSTEM_PROMPT)
    object.__setattr__(runner, "_banks", banks or {})
    object.__setattr__(runner, "_asset_catalog", catalog)
    object.__setattr__(runner, "_qwen_call_start_waiter", qwen_call_start_waiter)
    object.__setattr__(runner, "_deterministic_action_contract_version", None)
    if runner_type is PortfolioAssistantRunner:
        object.__setattr__(runner, "_runtime_lock_sha256", "f" * 64)
    return runner


def test_core_fast_deterministic_contract_owns_tool_and_response(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="core-fast-deterministic-contract")
    skill = _Skill(
        slug="exact-contract-skill",
        capability_id="product.exact_match",
        description="Find the exact visible product.",
        body="# Objective\n\nFind the exact product with image search.",
        operators=("image_product_search",),
    )
    bank = _one_skill_bank(skill, "e")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="llm_static",
        bank_sha256=bank.bank_sha256,
    )
    model_calls = []

    def fake_chat(provider, messages, **kwargs):
        model_calls.append((provider, messages, kwargs))
        return _route_response(
            request_id="deterministic-route-only",
            selected_capability="product.exact_match",
            output_tokens=8,
        )

    def fake_invoke(_registry, name, arguments, _context):
        assert name == "image_product_search"
        assert arguments["asset_id"] == request.query.asset_binding.asset_id
        output = {
            "hits": [
                {
                    "score": 1.0,
                    "product": {
                        "product_id": "private-product",
                        "title": "Deterministic shoe",
                        "category_l1": "Shoes",
                        "image_path": "private.jpg",
                        "source": "fixture",
                    },
                }
            ]
        }
        arguments_bytes = canonical_json_bytes(arguments)
        output_bytes = canonical_json_bytes(output)
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256="1" * 64,
            arguments=arguments,
            arguments_bytes=arguments_bytes,
            arguments_sha256=sha256_bytes(arguments_bytes),
            output=output,
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", fake_invoke)
    runner = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=CoreFastAssistantRunner,
        banks={name: bank for name in ("llm_static", "s1", "s1s2", "full")},
    )
    object.__setattr__(
        runner,
        "_deterministic_action_contract_version",
        "core-fast-deterministic-action-response-v5",
    )

    execution = runner.execute(request)

    assert len(model_calls) == 1
    assert [item.tool_name for item in execution.response.tool_trace] == [
        "image_product_search"
    ]
    assert execution.response.error_code is None
    assert execution.response.response_text == (
        "answer:\nThe tool returned eligible candidates listed below.\n"
        "product_cards:\ntool-call-1-evidence-1 | "
        "tool-call-1-product-1 | Deterministic shoe\n"
        "uncertainty:\nOnly the returned public candidate evidence is shown."
    )


def test_core_fast_deterministic_replay_reuses_trace_and_consumes_candidate_policy(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="core-fast-deterministic-replay")
    parent_skill = _Skill(
        slug="exact-contract-skill",
        capability_id="product.exact_match",
        description="Find the exact visible product.",
        body="# Objective\n\nFind the exact product with image search.",
        operators=("image_product_search",),
    )
    candidate_skill = _Skill(
        **{
            **parent_skill.__dict__,
            "body": parent_skill.body
            + "\n"
            + render_deterministic_semantic_policy(
                DeterministicSemanticPolicy(
                    capability_id="product.exact_match",
                    evidence_terms=("bag",),
                )
            ),
        }
    )
    parent_bank = _one_skill_bank(parent_skill, "e")
    candidate_bank = _one_skill_bank(candidate_skill, "f")
    query_id = "deterministic-replay-query"
    scorer_query = _query_for_catalog(fixture.asset_catalog, query_id=query_id)
    parent_request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="llm_static",
        bank_sha256=parent_bank.bank_sha256,
        query_id=query_id,
    )
    candidate_request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=candidate_bank.bank_sha256,
        query_id=query_id,
    )
    model_calls = []

    def fake_chat(provider, messages, **kwargs):
        model_calls.append((provider, messages, kwargs))
        return _route_response(
            request_id="deterministic-replay-parent-route",
            selected_capability="product.exact_match",
            output_tokens=8,
        )

    def fake_invoke(_registry, name, arguments, _context):
        output = {
            "hits": [
                {
                    "score": 1.0,
                    "product": {
                        "product_id": "private-product",
                        "title": "Deterministic shoe",
                        "category_l1": "Shoes",
                        "image_path": "private.jpg",
                        "source": "fixture",
                    },
                }
            ]
        }
        arguments_bytes = canonical_json_bytes(arguments)
        output_bytes = canonical_json_bytes(output)
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256="1" * 64,
            arguments=arguments,
            arguments_bytes=arguments_bytes,
            arguments_sha256=sha256_bytes(arguments_bytes),
            output=output,
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", fake_invoke)
    runner = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=CoreFastAssistantRunner,
        banks={
            "llm_static": parent_bank,
            "s1": candidate_bank,
            "s1s2": candidate_bank,
            "full": candidate_bank,
        },
    )
    object.__setattr__(
        runner,
        "_deterministic_action_contract_version",
        "core-fast-deterministic-action-response-v5",
    )
    parent = runner.execute(parent_request)
    trace = parent.response.tool_trace[0]
    scorer_payload = {
        "candidates": [
            {
                "candidate_ordinal": 1,
                "eligible": True,
                "evidence_reference": "tool-call-1-evidence-1",
                "product_id": "tool-call-1-product-1",
                "public_attributes": [["category", "catalog_product"]],
                "title": "Deterministic shoe",
            }
        ]
    }
    scorer_calls = (
        PublicScorerCallEvidenceV2(
            call_index=1,
            tool_name="image_product_search",
            arguments_sha256=trace.arguments_sha256,
            result_sha256=trace.result_sha256,
            argument_projection={"asset_handle": "query_asset"},
            payload_kind="product_candidates_v1",
            payload=scorer_payload,
            payload_sha256=sha256_bytes(canonical_json_bytes(scorer_payload)),
        ),
    )

    replay = runner.execute_deterministic_body_replay(
        candidate_request,
        parent_response=parent.response,
        parent_receipt=parent.receipt,
        parent_scorer_calls=scorer_calls,
        scorer_query=scorer_query,
    )

    assert len(model_calls) == 1
    assert replay.response.response_text.startswith("answer:\nno supported match")
    assert replay.response.tool_trace == parent.response.tool_trace
    assert replay.scorer_calls == scorer_calls
    assert replay.receipt.model_calls == ()
    assert replay.receipt.aggregate_usage == LLMUsage(input_tokens=0, output_tokens=0)
    assert (
        replay.receipt.deterministic_replay_source_receipt_sha256
        == parent.receipt.receipt_sha256
    )


def _budget_context(tmp_path, request, *, name: str = "budget-ledger"):
    ledger_root = tmp_path / name
    initialize_portfolio_budget_ledger(
        ledger_root,
        matrix_run_id=request.matrix_run_id,
        phase_cap_cny=Decimal("100.000000000000"),
    )
    return PortfolioAssistantBudgetContext(
        ledger_root=ledger_root,
        shard_id="test-shard-001",
        instance_sha256=request.query.query_sha256,
        attempt_index=1,
    )


def _budget_context_with_cap(tmp_path, request, *, name: str, cap: str):
    ledger_root = tmp_path / name
    initialize_portfolio_budget_ledger(
        ledger_root,
        matrix_run_id=request.matrix_run_id,
        phase_cap_cny=Decimal(cap),
    )
    return PortfolioAssistantBudgetContext(
        ledger_root=ledger_root,
        shard_id="test-shard-001",
        instance_sha256=request.query.query_sha256,
        attempt_index=1,
    )


def _response(
    *,
    request_id: str,
    text: str,
    output_tokens: int,
    finish_reason: str = "stop",
    tool_calls: tuple[LLMToolCall, ...] = (),
) -> LLMResponse:
    return LLMResponse(
        provider="qwen",
        endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
        requested_model=_MODEL,
        response_model=_MODEL,
        request_id=request_id,
        text=text,
        tool_calls=tool_calls,
        usage=LLMUsage(input_tokens=11, output_tokens=output_tokens),
        finish_reason=finish_reason,
        latency_ms=3,
    )


def _route_response(
    *,
    request_id: str,
    selected_capability: str,
    output_tokens: int = 8,
    arguments: dict[str, object] | None = None,
    function_name: str | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    payload = (
        arguments
        if arguments is not None
        else {"selected_capability": selected_capability}
    )
    if function_name is None:
        return _response(
            request_id=request_id,
            text=canonical_json_bytes(payload).decode("utf-8"),
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )
    return _response(
        request_id=request_id,
        text="",
        output_tokens=output_tokens,
        finish_reason="tool_calls",
        tool_calls=(
            LLMToolCall(
                call_id=f"{request_id}-call",
                name=function_name,
                arguments_json=canonical_json_bytes(payload).decode("utf-8"),
            ),
        ),
    )


def test_noskill_final_is_plain_text_and_has_no_model_supplied_identity(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="noskill-plain-final")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        return _response(
            request_id="noskill-plain-final",
            text="这是直接展示给用户的回答。",
            output_tokens=8,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
    ).execute(request)

    assert len(calls) == 1
    assert execution.response.error_code is None
    assert execution.response.response_text == "这是直接展示给用户的回答。"
    assert execution.response.selected_capability is None
    assert execution.response.skill_slug is None


def test_encyclopedia_contract_gets_one_no_tool_format_repair(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="encyclopedia-response-repair")
    skill = _Skill(
        slug="s1-encyclopedia",
        capability_id="knowledge.visual_encyclopedia",
        description="Explain the visible entity from encyclopedia evidence.",
        body="# Objective\n\nUse retrieval evidence or a fail-closed fallback.",
        operators=("encyclopedia_lookup",),
    )
    bank = _one_skill_bank(skill, "e")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id="encyclopedia-repair-route",
                selected_capability=skill.capability_id,
            )
        if len(calls) == 2:
            return _response(
                request_id="encyclopedia-repair-tool",
                text="",
                output_tokens=5,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="encyclopedia-repair-tool-call",
                        name="encyclopedia_lookup",
                        arguments_json='{"entity":"unknown plant"}',
                    ),
                ),
            )
        if len(calls) == 3:
            return _response(
                request_id="encyclopedia-invalid-final",
                text="identity remains unresolved <|im_end|>",
                output_tokens=7,
            )
        if len(calls) == 4:
            return _response(
                request_id="encyclopedia-fixed-final",
                text=(
                    "answer:\nnot enough evidence\n"
                    "evidence:\nnot enough evidence\n"
                    "uncertainty:\nidentity remains unresolved"
                ),
                output_tokens=12,
            )
        pytest.fail("response contract allows at most one repair")

    def fake_invoke(_registry, name, arguments, _context):
        argument_bytes = canonical_json_bytes(arguments)
        output = []
        output_bytes = canonical_json_bytes(output)
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256="1" * 64,
            arguments=arguments,
            arguments_bytes=argument_bytes,
            arguments_sha256=sha256_bytes(argument_bytes),
            output=output,
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", fake_invoke)
    budget_context = _budget_context(tmp_path, request, name="response-repair-ledger")
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks={"s1": bank},
    ).execute(request, budget_context=budget_context)

    assert len(calls) == 4
    assert calls[-1][2]["tools"] is None
    assert calls[-1][2]["images"] is None
    assert len(execution.response.tool_trace) == 1
    assert execution.response.error_code is None
    repair = execution.receipt.response_contract_repair
    assert repair is not None
    assert repair.status == "passed"
    assert repair.attempt_count == 1
    assert repair.repair_call_index == 4
    assert repair.repair_call_usage.output_tokens == 12
    assert repair.ambiguity_signal_policy == (
        "empty_sources_only_no_reliable_public_ambiguity_field"
    )
    ledger = load_portfolio_budget_ledger(budget_context.ledger_root)
    assert [item.identity.stage for item in ledger.reservations] == [
        "assistant_route",
        "assistant_action",
        "assistant_action",
        "assistant_action",
    ]
    assert [item.identity.call_index for item in ledger.reservations] == [1, 1, 2, 3]
    assert len(ledger.settlements) == 4
    assert [item.actual_output_tokens for item in ledger.settlements] == [8, 5, 7, 12]
    assert ledger.settlements[-1].provider_request_id == "encyclopedia-fixed-final"
    assert ledger.settlements[-1].actual_input_tokens == (
        repair.repair_call_usage.input_tokens
    )
    assert ledger.settlements[-1].actual_output_tokens == (
        repair.repair_call_usage.output_tokens
    )
    assert ledger.settlements[-1].actual_cost_cny > Decimal("0")
    assert execution.receipt.aggregate_usage == LLMUsage(
        input_tokens=sum(item.actual_input_tokens for item in ledger.settlements),
        output_tokens=sum(item.actual_output_tokens for item in ledger.settlements),
    )
    assert ledger.settled_actual_cost_cny == sum(
        (item.actual_cost_cny for item in ledger.settlements), Decimal("0")
    )


@pytest.mark.parametrize("config", ("llm_static", "s1"))
@pytest.mark.parametrize(
    "invalid_text",
    (
        ("answer:\nno supported match\nnone\nuncertainty:\ninsufficient evidence"),
        (
            "answer:\nno supported match\n"
            "notes:\nnone\n"
            "product_cards:\nnone\n"
            "uncertainty:\ninsufficient evidence"
        ),
    ),
    ids=("missing-section", "extra-section"),
)
def test_static_and_s1_direct_final_share_one_section_repair(
    monkeypatch,
    canonical_registry_factory,
    config: str,
    invalid_text: str,
) -> None:
    fixture = canonical_registry_factory(name=f"direct-final-contract-{config}")
    skill = _Skill(
        slug=f"{config}-exact",
        capability_id="product.exact_match",
        description="Find the exact product with public evidence.",
        body="# Objective\n\nUse public retrieval evidence or fail closed.",
        operators=("image_product_search", "text_product_search"),
    )
    bank = _one_skill_bank(skill, "e" if config == "llm_static" else "f")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config=config,
        bank_sha256=bank.bank_sha256,
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id=f"direct-final-{config}-route",
                selected_capability=skill.capability_id,
            )
        if len(calls) == 2:
            return _response(
                request_id=f"direct-final-{config}-invalid",
                text=invalid_text,
                output_tokens=10,
            )
        if len(calls) == 3:
            return _response(
                request_id=f"direct-final-{config}-repaired",
                text=(
                    "answer:\nno supported match\n"
                    "product_cards:\nnone\n"
                    "uncertainty:\ninsufficient evidence"
                ),
                output_tokens=12,
            )
        pytest.fail("direct final contract allows only one repair")

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        banks={config: bank},
    ).execute(request)

    assert len(calls) == 3
    assert '"final_response_contract"' in calls[1][1][0]["content"]
    assert calls[2][2]["tools"] is None
    assert calls[2][2]["images"] is None
    assert execution.response.error_code is None
    assert execution.response.tool_trace == ()
    repair = execution.receipt.response_contract_repair
    assert repair is not None
    assert repair.attempt_count == 1
    assert repair.status == "passed"
    assert repair.initial_reason_codes == ("response_section_invalid",)
    diagnostic = make_s1_round2_response_contract_diagnostic(
        query_id=request.query.query_id,
        config=config,
        checkpoint_file_sha256="a" * 64,
        checkpoint_row_sha256="b" * 64,
        response=execution.response,
        receipt=execution.receipt,
    )
    assert diagnostic.repair_status == "passed"
    assert diagnostic.repair_attempt_count == 1
    assert diagnostic.repair_receipt_sha256 == repair.receipt_sha256
    assert diagnostic.mapped_contract_reason_codes == ()


def test_second_invalid_encyclopedia_response_fails_without_another_repair(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="encyclopedia-response-repair-fails")
    skill = _Skill(
        slug="s1-encyclopedia",
        capability_id="knowledge.visual_encyclopedia",
        description="Explain the visible entity from encyclopedia evidence.",
        body="# Objective\n\nUse retrieval evidence or a fail-closed fallback.",
        operators=("encyclopedia_lookup",),
    )
    bank = _one_skill_bank(skill, "f")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id="encyclopedia-failed-repair-route",
                selected_capability=skill.capability_id,
            )
        if len(calls) == 2:
            return _response(
                request_id="encyclopedia-failed-repair-tool",
                text="",
                output_tokens=5,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="encyclopedia-failed-repair-tool-call",
                        name="encyclopedia_lookup",
                        arguments_json='{"entity":"unknown plant"}',
                    ),
                ),
            )
        return _response(
            request_id=f"encyclopedia-still-invalid-{len(calls)}",
            text="identity remains unresolved <|im_end|>",
            output_tokens=7,
        )

    def fake_invoke(_registry, name, arguments, _context):
        argument_bytes = canonical_json_bytes(arguments)
        output_bytes = canonical_json_bytes([])
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256="1" * 64,
            arguments=arguments,
            arguments_bytes=argument_bytes,
            arguments_sha256=sha256_bytes(argument_bytes),
            output=[],
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", fake_invoke)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        banks={"s1": bank},
    ).execute(request)

    assert len(calls) == 4
    assert execution.response.error_code == "response_contract_error"
    assert execution.response.response_text == ""
    assert execution.receipt.response_contract_repair is not None
    assert execution.receipt.response_contract_repair.status == "failed"
    diagnostic = make_s1_round2_response_contract_diagnostic(
        query_id=request.query.query_id,
        config="s1",
        checkpoint_file_sha256="a" * 64,
        checkpoint_row_sha256="b" * 64,
        response=execution.response,
        receipt=execution.receipt,
    )
    assert diagnostic.assistant_response_sha256 == execution.receipt.response_sha256
    assert diagnostic.assistant_receipt_sha256 == execution.receipt.receipt_sha256
    assert diagnostic.repair_receipt_sha256 == (
        execution.receipt.response_contract_repair.receipt_sha256
    )
    assert diagnostic.repair_status == "failed"
    assert diagnostic.repair_attempt_count == 1
    assert diagnostic.repair_input_tokens == 11
    assert diagnostic.repair_output_tokens == 7
    assert diagnostic.mapped_contract_reason_codes == (
        "fallback_contract_failed",
        "output_section_invalid",
    )


def test_response_text_with_nested_pseudo_tool_json_is_not_executed(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="nested-pseudo-tool")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
    )
    calls = []
    final_text = (
        '用户可见说明。{"tool_response":{"tool_name":"encyclopedia_lookup",'
        '"arguments":{"entity":"dress"}}'
    )

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        return _response(
            request_id=f"nested-pseudo-tool-{len(calls)}",
            text=final_text,
            output_tokens=12,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
    ).execute(request)

    assert len(calls) == 1
    assert execution.response.error_code is None
    assert execution.response.response_text == final_text
    assert execution.response.tool_trace == ()
    assert execution.receipt.tool_trace == ()


def test_exhausted_output_budget_does_not_issue_one_token_retry(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="exhausted-output-budget")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
        max_output_tokens=7,
        max_turns=3,
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        return _response(
            request_id=f"exhausted-output-budget-{len(calls)}",
            text="",
            output_tokens=7,
            finish_reason="length",
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
    ).execute(request)

    assert len(calls) == 1
    assert calls[0][2]["max_tokens"] == 7
    assert execution.response.error_code is not None
    assert execution.response.usage.output_tokens == 7


def test_multi_product_final_uses_remaining_budget_and_still_fails_closed_on_length(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(
        name="multi-product-output-budget",
    )
    skill = _Skill(
        slug="s1-multi-product",
        capability_id="product.multi_search",
        description="Find every visible product separately.",
        body="# Objective\n\nMap each visible item to grounded product evidence.",
        operators=("image_product_search",),
    )
    bank = _one_skill_bank(skill, "b")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        max_output_tokens=2_048,
        max_turns=4,
        bank_sha256=bank.bank_sha256,
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id="multi-route",
                selected_capability="product.multi_search",
                output_tokens=10,
            )
        if len(calls) == 2:
            return _response(
                request_id="multi-tool",
                text="",
                output_tokens=20,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="multi-tool-call",
                        name="image_product_search",
                        arguments_json=canonical_json_bytes(
                            {"asset_id": "query_asset"}
                        ).decode("utf-8"),
                    ),
                ),
            )
        return _response(
            request_id="multi-final-length",
            text="unfinished",
            output_tokens=2_018,
            finish_reason="length",
        )

    def fake_invoke(_registry, name, arguments, _context):
        assert name == "image_product_search"
        argument_bytes = canonical_json_bytes(arguments)
        output = {
            "hits": [
                {
                    "score": 0.9,
                    "product": {
                        "title": "Public candidate",
                        "category_l1": "Shoes",
                    },
                }
            ]
        }
        output_bytes = canonical_json_bytes(output)
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256="1" * 64,
            arguments=arguments,
            arguments_bytes=argument_bytes,
            arguments_sha256=sha256_bytes(argument_bytes),
            output=output,
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", fake_invoke)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        banks={"s1": bank},
    ).execute(request)

    assert len(calls) == 3
    assert (
        calls[0][2]["max_tokens"]
        == PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS
        == 64
    )
    assert calls[0][2]["json_mode"] is True
    assert calls[0][2]["tools"] is None
    assert calls[0][2]["tool_choice"] is None
    assert calls[0][2]["images"] is None
    assert set(json.loads(calls[0][1][1]["content"])) == {"turns"}
    assert _SYSTEM_PROMPT not in calls[0][1][0]["content"]
    assert '"capability_id":"product.multi_search"' in calls[0][1][0]["content"]
    assert calls[1][2]["images"] is not None
    assert calls[2][2]["images"] is not None
    assert calls[2][2]["max_tokens"] == 2_018
    action_prompt = calls[1][1][0]["content"]
    assert action_prompt.index("Map each visible item") < action_prompt.index(
        "Runtime multi-product presentation contract"
    )
    assert "within 300 words" in action_prompt
    assert "no more than 12 compact item lines" in action_prompt
    assert "matching public candidate ordinal" in action_prompt
    assert "product_cards section must still repeat" in action_prompt
    assert execution.response.error_code == "runtime_error"
    assert execution.response.response_text == ""
    assert execution.response.usage.output_tokens == 2_048
    assert execution.response.tool_trace[0].status == "success"


def test_context_violation_is_not_folded_into_runtime_error() -> None:
    from skillchain.runners.assistant import _error_code

    assert _error_code("context_violation") == "context_violation"
    assert _error_code("unclassified_backend_failure") == "runtime_error"


def test_text_product_search_has_empty_model_schema_and_runner_binds_query(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="text-search-authoritative-binding")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
    )
    query_text = json.loads(request.query.public_input_json)["text"]
    calls = []
    invocations = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _response(
                request_id="text-search-tool",
                text="",
                output_tokens=5,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="text-search-tool-call",
                        name="text_product_search",
                        arguments_json="{}",
                    ),
                ),
            )
        return _response(
            request_id="text-search-final",
            text=(
                "answer:\nno supported match\n"
                "product_cards:\nnone\n"
                "uncertainty:\nNo exact identifier candidate was found."
            ),
            output_tokens=8,
        )

    def fake_invoke(_registry, name, arguments, _context):
        invocations.append((name, arguments))
        argument_bytes = canonical_json_bytes(arguments)
        output = {"hits": []}
        output_bytes = canonical_json_bytes(output)
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256="1" * 64,
            arguments=arguments,
            arguments_bytes=argument_bytes,
            arguments_sha256=sha256_bytes(argument_bytes),
            output=output,
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", fake_invoke)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
    ).execute(request)

    text_tool = next(
        item
        for item in calls[0][2]["tools"]
        if item["function"]["name"] == "text_product_search"
    )["function"]
    assert text_tool["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert "exact identifier fallback" in text_tool["description"]
    assert "not semantic or style search" in text_tool["description"]
    assert "may return no candidates" in text_tool["description"]
    assert invocations == [("text_product_search", {"query": query_text})]
    assert execution.response.error_code is None
    assert execution.response.tool_trace[0].status == "success"


def test_text_product_search_rejects_model_supplied_query(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="text-search-free-query-rejected")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _response(
                request_id="free-text-search-tool",
                text="",
                output_tokens=5,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="free-text-search-tool-call",
                        name="text_product_search",
                        arguments_json='{"query":"model-chosen query"}',
                    ),
                ),
            )
        return _response(
            request_id="free-text-search-final",
            text="The unsupported free query was ignored.",
            output_tokens=8,
        )

    def forbidden_invoke(*_args, **_kwargs):
        raise AssertionError("registry must not receive model-supplied query text")

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", forbidden_invoke)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
    ).execute(request)

    assert execution.response.error_code is None
    assert execution.response.tool_trace[0].status == "error"
    assert execution.response.tool_trace[0].error_code == "invalid_arguments"


def test_style_search_model_schema_hides_query_and_runner_binds_it(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="style-authoritative-query-binding")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
    )
    query_text = json.loads(request.query.public_input_json)["text"]
    authoritative_asset_id = request.query.asset_binding.asset_id
    calls = []
    invocations = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _response(
                request_id="style-tool",
                text="",
                output_tokens=5,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="style-tool-call",
                        name="style_similar_search",
                        arguments_json='{"asset_id":"query_asset"}',
                    ),
                ),
            )
        return _response(
            request_id="style-final",
            text=(
                "answer:\nunable to recommend\n"
                "diversity_rationale:\nnone\n"
                "product_cards:\nnone\n"
                "uncertainty:\nNo evidence-backed style candidate was found."
            ),
            output_tokens=8,
        )

    def fake_invoke(_registry, name, arguments, _context):
        invocations.append((name, arguments))
        argument_bytes = canonical_json_bytes(arguments)
        output = {
            "style_submode": "same_category_alternative",
            "hits": [],
        }
        output_bytes = canonical_json_bytes(output)
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256="1" * 64,
            arguments=arguments,
            arguments_bytes=argument_bytes,
            arguments_sha256=sha256_bytes(argument_bytes),
            output=output,
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", fake_invoke)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
    ).execute(request)

    style_tool = next(
        item
        for item in calls[0][2]["tools"]
        if item["function"]["name"] == "style_similar_search"
    )["function"]
    assert style_tool["parameters"] == {
        "type": "object",
        "properties": {
            "asset_id": {
                "type": "string",
                "const": "query_asset",
                "description": "Use the opaque query_asset handle.",
            }
        },
        "required": ["asset_id"],
        "additionalProperties": False,
    }
    assert invocations == [
        (
            "style_similar_search",
            {"asset_id": authoritative_asset_id, "query": query_text},
        )
    ]
    assert query_text not in json.dumps(style_tool, ensure_ascii=False)
    assert authoritative_asset_id not in json.dumps(calls, ensure_ascii=False)
    assert execution.response.error_code is None
    assert execution.response.tool_trace[0].status == "success"


def test_action_messages_expose_only_aliases_and_public_product_handles(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="action-public-surface")
    skill = _Skill(
        slug="private-exact-skill",
        capability_id="product.exact_match",
        description="Find the same visible product.",
        body="# Objective\n\nReturn one grounded candidate card.",
        operators=("image_product_search",),
    )
    bank = _one_skill_bank(skill, "c")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
    )
    public_query = json.loads(request.query.public_input_json)
    query_asset_token = public_query["asset_id"]
    assert request.query.asset_binding is not None
    query_asset_id = request.query.asset_binding.asset_id
    query_image_path = request.query.asset_binding.image_path
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id="public-surface-route",
                selected_capability=skill.capability_id,
                output_tokens=8,
            )
        if len(calls) == 2:
            return _response(
                request_id="public-surface-tool",
                text="",
                output_tokens=8,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="public-surface-tool-call",
                        name="image_product_search",
                        arguments_json='{"asset_id":"query_asset"}',
                    ),
                ),
            )
        return _response(
            request_id="public-surface-final",
            text=(
                "answer:\n候选已找到。\n"
                "product_cards:\ntool-call-1-product-1 "
                "tool-call-1-evidence-1\n"
                "uncertainty:\n仅依据返回候选。"
            ),
            output_tokens=12,
        )

    def fake_invoke(_registry, name, arguments, _context):
        argument_bytes = canonical_json_bytes(arguments)
        output = {
            "query_asset_id": query_asset_id,
            "artifact_binding": {"private": "runtime-digest"},
            "hits": [
                {
                    "score": 1.0,
                    "product": {
                        "product_id": "private:secret-product-123",
                        "title": "PRIVATE item secret-product-123",
                        "category_l1": "Shoes",
                        "image_path": "private/catalog.jpg",
                        "source": "private",
                    },
                }
            ],
        }
        output_bytes = canonical_json_bytes(output)
        return ToolInvocationResult(
            tool_name=name,
            spec_sha256="1" * 64,
            arguments=arguments,
            arguments_bytes=argument_bytes,
            arguments_sha256=sha256_bytes(argument_bytes),
            output=output,
            output_bytes=output_bytes,
            output_sha256=sha256_bytes(output_bytes),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(type(fixture.registry), "invoke", fake_invoke)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        banks={"s1": bank},
    ).execute(request)

    assert len(calls) == 3
    action_messages = canonical_json_bytes(calls[-1][1]).decode("utf-8")
    for hidden in (
        skill.capability_id,
        skill.slug,
        query_asset_token,
        query_asset_id,
        query_image_path,
        "private:secret-product-123",
        "secret-product-123",
        "private/catalog.jpg",
        "runtime-digest",
    ):
        assert hidden not in action_messages
    assert "query_asset" in action_messages
    assert "tool-call-1-product-1" in action_messages
    assert "tool-call-1-evidence-1" in action_messages
    assert "final_response_contract" in action_messages
    assert "portfolio-gcs-v2-model-visible-response-contract-v2" in action_messages
    final_tool_message = json.loads(calls[-1][1][-1]["content"])
    assert final_tool_message["final_response_contract"]["required_sections"] == [
        "answer",
        "product_cards",
        "uncertainty",
    ]
    assert "no supported match" in action_messages
    assert execution.response.error_code is None
    fields = dict(execution.response.visible_cards[0].fields)
    assert fields["product_id"] == "tool-call-1-product-1"
    assert fields["evidence_reference"] == "tool-call-1-evidence-1"


def test_ocr_public_handle_binds_to_authoritative_asset_without_leaking(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="ocr-authoritative-asset-binding")
    skill = _Skill(
        slug="private-document-skill",
        capability_id="utility.document_reading",
        description="Read visible document content.",
        body="# Objective\n\nExtract the visible text with document OCR.",
        operators=("document_ocr",),
    )
    bank = _one_skill_bank(skill, "d")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
        query_id="fixture-query",
    )
    public_query = json.loads(request.query.public_input_json)
    public_asset_token = public_query["asset_id"]
    assert request.query.asset_binding is not None
    authoritative_asset_id = request.query.asset_binding.asset_id
    authoritative_image_path = request.query.asset_binding.image_path
    assert public_asset_token != authoritative_asset_id
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id="ocr-authoritative-route",
                selected_capability=skill.capability_id,
                output_tokens=8,
            )
        if len(calls) == 2:
            return _response(
                request_id="ocr-authoritative-tool",
                text="",
                output_tokens=8,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="ocr-authoritative-tool-call",
                        name="document_ocr",
                        arguments_json='{"asset_id":"query_asset"}',
                    ),
                ),
            )
        return _response(
            request_id="ocr-authoritative-final",
            text=(
                "answer:\nname: Alice tool-call-1-line-1\n"
                "evidence:\nname: Alice tool-call-1-line-1\n"
                "uncertainty:\nuntrusted document text"
            ),
            output_tokens=9,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        banks={"s1": bank},
    ).execute(
        request,
        scorer_query=_query_for_catalog(
            fixture.asset_catalog,
            query_id="fixture-query",
        ),
    )

    assert len(calls) == 3
    assert execution.response.error_code is None
    assert len(execution.response.tool_trace) == 1
    trace = execution.response.tool_trace[0]
    assert trace.tool_name == "document_ocr"
    assert trace.status == "success"
    assert execution.scorer_capture_policy_version == (
        "portfolio-gcs-scorer-evidence-v2"
    )
    assert len(execution.scorer_calls) == 1
    scorer_call = execution.scorer_calls[0]
    assert scorer_call.call_index == 1
    assert scorer_call.tool_name == "document_ocr"
    assert scorer_call.arguments_sha256 == trace.arguments_sha256
    assert scorer_call.result_sha256 == trace.result_sha256
    assert scorer_call.payload_kind == "ocr_lines_v1"
    assert scorer_call.payload["lines"][0]["line_reference"] == ("tool-call-1-line-1")
    expected_arguments = canonical_json_bytes({"asset_id": authoritative_asset_id})
    assert trace.arguments_sha256 == sha256_bytes(expected_arguments)
    assert execution.receipt.query_asset_id == authoritative_asset_id
    assert len(execution.response.visible_tool_evidence) == 1
    evidence = execution.response.visible_tool_evidence[0]
    assert evidence.tool_name == "document_ocr"
    assert evidence.status == "success"
    assert evidence.visible_text == "[tool-call-1-line-1] Alice"

    model_surface = canonical_json_bytes(
        [messages for _provider, messages, _kwargs in calls]
    ).decode("utf-8")
    response_surface = execution.response.model_dump_json()
    assert "query_asset" in model_surface
    assert "Alice" in model_surface
    assert "tool-call-1-line-1" in model_surface
    for hidden in (
        public_asset_token,
        authoritative_asset_id,
        authoritative_image_path,
    ):
        assert hidden not in model_surface
        assert hidden not in response_surface


def test_scorer_integrity_failure_after_successful_tool_invoke_is_fatal(
    monkeypatch,
    canonical_registry_factory,
) -> None:
    fixture = canonical_registry_factory(name="scorer-integrity-fatal")
    skill = _Skill(
        slug="document-skill",
        capability_id="utility.document_reading",
        description="Read visible document content.",
        body="# Objective\n\nExtract the visible text with document OCR.",
        operators=("document_ocr",),
    )
    bank = _one_skill_bank(skill, "d")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
        query_id="fixture-query",
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id="scorer-integrity-route",
                selected_capability=skill.capability_id,
                output_tokens=8,
            )
        if len(calls) == 2:
            return _response(
                request_id="scorer-integrity-tool",
                text="",
                output_tokens=8,
                finish_reason="tool_calls",
                tool_calls=(
                    LLMToolCall(
                        call_id="scorer-integrity-tool-call",
                        name="document_ocr",
                        arguments_json='{"asset_id":"query_asset"}',
                    ),
                ),
            )
        pytest.fail("fatal scorer capture must stop before another model call")

    captured = []

    def reject_scorer_capture(**kwargs):
        captured.append(kwargs)
        raise ValueError("fixture payload validation failure")

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(
        (
            "skillchain.evaluation.portfolio_gcs_evidence."
            "project_validated_tool_result_v2"
        ),
        reject_scorer_capture,
    )

    with pytest.raises(
        PublicScorerEvidenceIntegrityError,
        match="invalid v2 public scorer call: fixture payload validation failure",
    ):
        _runner(
            registry=fixture.registry,
            catalog=fixture.asset_catalog,
            banks={"s1": bank},
        ).execute(
            request,
            scorer_query=_query_for_catalog(
                fixture.asset_catalog,
                query_id="fixture-query",
            ),
        )

    assert len(calls) == 2
    assert len(captured) == 1
    assert captured[0]["tool_name"] == "document_ocr"
    assert captured[0]["raw_validated_output"]


def test_skilled_final_is_not_forced_to_use_bank_external_evidence_tool(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="bank-owned-evidence-policy")
    skill = _Skill(
        slug="s1-skill",
        capability_id="utility.document_reading",
        description="Answer the user's request.",
        body="# Objective\n\nAnswer using the selected Skill.",
        operators=(),
    )
    bank = _Bank(
        bank_sha256="b" * 64,
        skills=(skill,),
        capability_map=(
            _Binding(
                capability_id=skill.capability_id,
                skill_slug=skill.slug,
            ),
        ),
    )
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
    )
    calls = []
    final_text = (
        "answer:\n无需调用 Bank 未声明的工具即可完成回答。\n"
        "evidence:\n无需调用 Bank 未声明的工具即可完成回答。\n"
        "uncertainty:\n未调用外部工具。"
    )

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id="bank-owned-evidence-route",
                selected_capability=skill.capability_id,
                output_tokens=8,
            )
        return _response(
            request_id=f"bank-owned-evidence-final-{len(calls)}",
            text=final_text,
            output_tokens=10,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    budget_context = _budget_context(tmp_path, request)
    qwen_call_starts = []
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks={"s1": bank},
        qwen_call_start_waiter=lambda label: qwen_call_starts.append(label) or 0.0,
    ).execute(request, budget_context=budget_context)

    assert len(calls) == 2
    assert qwen_call_starts == [
        f"test-shard-001:{request.query.query_id}:assistant_route:attempt-1:call-1",
        f"test-shard-001:{request.query.query_id}:assistant_action:attempt-1:call-1",
    ]
    assert (
        "Runtime multi-product presentation contract" not in calls[1][1][0]["content"]
    )
    assert execution.response.error_code is None
    assert execution.response.response_text == final_text
    assert execution.response.selected_capability == skill.capability_id
    assert execution.response.skill_slug == skill.slug
    assert execution.response.tool_trace == ()
    ledger = load_portfolio_budget_ledger(budget_context.ledger_root)
    assert [item.identity.stage for item in ledger.reservations] == [
        "assistant_route",
        "assistant_action",
    ]
    assert [item.identity.call_index for item in ledger.reservations] == [1, 1]
    assert all(
        item.identity.wire_request_sha256 is not None for item in ledger.reservations
    )
    assert len(ledger.settlements) == 2


def test_qwen_call_start_gate_failure_is_fatal_before_budget_reservation(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="qwen-call-start-gate-failure")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
        bank_sha256=None,
    )
    provider_calls = []
    monkeypatch.setattr(
        "skillchain.llm.chat",
        lambda *_args, **_kwargs: provider_calls.append(True),
    )
    budget_context = _budget_context(tmp_path, request, name="call-gate-ledger")

    def fail_gate(_label):
        raise RuntimeError("local SQLite gate failed")

    with pytest.raises(AssistantProviderCallGateError, match="before reservation"):
        _runner(
            registry=fixture.registry,
            catalog=fixture.asset_catalog,
            runner_type=PortfolioAssistantRunner,
            qwen_call_start_waiter=fail_gate,
        ).execute(request, budget_context=budget_context)

    ledger = load_portfolio_budget_ledger(budget_context.ledger_root)
    assert provider_calls == []
    assert ledger.reservations == ()
    assert ledger.settlements == ()
    assert ledger.forfeits == ()


def test_action_pre_response_failure_retains_validated_route_identity(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="route-survives-provider-failure")
    skill = _Skill(
        slug="s1-skill",
        capability_id="utility.document_reading",
        description="Answer the user's request.",
        body="# Objective\n\nAnswer using the selected Skill.",
        operators=(),
    )
    bank = _Bank(
        bank_sha256="b" * 64,
        skills=(skill,),
        capability_map=(
            _Binding(
                capability_id=skill.capability_id,
                skill_slug=skill.slug,
            ),
        ),
    )
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
    )
    calls = []
    provider_failed = {"value": False}

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _route_response(
                request_id="validated-route",
                selected_capability=skill.capability_id,
                output_tokens=8,
            )
        provider_failed["value"] = True
        raise LLMTimeoutError("fixture provider timeout")

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(
        "skillchain.runners.assistant.time.perf_counter_ns",
        lambda: (
            (request.budget.timeout_ms + 1) * 1_000_000
            if provider_failed["value"]
            else 0
        ),
    )
    budget_context = _budget_context(tmp_path, request)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks={"s1": bank},
    ).execute(request, budget_context=budget_context)

    assert len(calls) == 2
    assert execution.response.error_code == "provider_pre_response_action"
    assert execution.response.provider_exception_type == "LLMTimeoutError"
    assert execution.response.selected_capability == skill.capability_id
    assert execution.response.skill_slug == skill.slug
    assert execution.response.route_trace_sha256 is not None
    assert len(execution.receipt.model_calls) == 1
    ledger = load_portfolio_budget_ledger(budget_context.ledger_root)
    assert len(ledger.reservations) == 2
    assert len(ledger.settlements) == 1
    assert len(ledger.forfeits) == 1
    assert ledger.unresolved_reservations == ()
    assert (
        ledger.forfeits[0].reservation_sha256
        == ledger.reservations[1].reservation_sha256
    )
    assert ledger.forfeits[0].reason == "provider_call_ended_without_captured_response"
    assert (
        execution.response.forfeited_reservation_sha256
        == ledger.forfeits[0].reservation_sha256
    )
    assert execution.response.budget_forfeit_sha256 == ledger.forfeits[0].forfeit_sha256

    incomplete = execution.response.model_dump(mode="json")
    incomplete.pop("budget_forfeit_sha256")
    with pytest.raises(ValueError, match="binding must be complete"):
        AssistantBackendResponse.model_validate_json(
            canonical_json_bytes(incomplete),
            strict=True,
        )

    legacy = execution.response.model_dump(mode="json")
    legacy["schema_version"] = 1
    legacy.pop("forfeited_reservation_sha256")
    legacy.pop("budget_forfeit_sha256")
    reparsed_legacy = AssistantBackendResponse.model_validate_json(
        canonical_json_bytes(legacy),
        strict=True,
    )
    assert reparsed_legacy.schema_version == 1


def test_route_pre_response_failure_is_retryable_without_route_identity(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="route-provider-failure")
    skill = _Skill(
        slug="s1-skill",
        capability_id="utility.document_reading",
        description="Answer the user's request.",
        body="# Objective\n\nAnswer using the selected Skill.",
        operators=(),
    )
    bank = _Bank(
        bank_sha256="b" * 64,
        skills=(skill,),
        capability_map=(
            _Binding(
                capability_id=skill.capability_id,
                skill_slug=skill.slug,
            ),
        ),
    )
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
    )
    provider_failed = {"value": False}

    def fake_chat(*args, **kwargs):
        provider_failed["value"] = True
        raise LLMTimeoutError("fixture provider timeout")

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    monkeypatch.setattr(
        "skillchain.runners.assistant.time.perf_counter_ns",
        lambda: (
            (request.budget.timeout_ms + 1) * 1_000_000
            if provider_failed["value"]
            else 0
        ),
    )
    budget_context = _budget_context(tmp_path, request)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks={"s1": bank},
    ).execute(request, budget_context=budget_context)

    assert execution.response.error_code == "provider_pre_response_route"
    assert execution.response.provider_exception_type == "LLMTimeoutError"
    assert execution.response.selected_capability is None
    assert execution.response.skill_slug is None
    assert execution.response.route_trace_sha256 is None
    assert execution.response.route_failure_subtype is None
    assert execution.response.route_failure_shape is None
    assert execution.receipt.model_calls == ()
    ledger = load_portfolio_budget_ledger(budget_context.ledger_root)
    assert len(ledger.reservations) == 1
    assert ledger.settlements == ()
    assert len(ledger.forfeits) == 1
    assert ledger.unresolved_reservations == ()
    assert (
        ledger.forfeits[0].reservation_sha256
        == ledger.reservations[0].reservation_sha256
    )
    assert ledger.forfeits[0].reason == "provider_call_ended_without_captured_response"
    assert (
        execution.response.forfeited_reservation_sha256
        == ledger.forfeits[0].reservation_sha256
    )
    assert execution.response.budget_forfeit_sha256 == ledger.forfeits[0].forfeit_sha256


@pytest.mark.parametrize(
    ("variant", "expected_error", "expected_subtype", "expected_payload_status"),
    [
        ("length", "route_length", "length", "not_examined"),
        (
            "wrong_function",
            "route_contract_error",
            "response_tool_calls",
            "not_examined",
        ),
        (
            "extra_argument",
            "route_contract_error",
            "invalid_route_json",
            "unexpected_keys",
        ),
        (
            "out_of_enum",
            "route_contract_error",
            "out_of_enum",
            "out_of_enum",
        ),
        (
            "empty",
            "route_contract_error",
            "response_empty_text",
            "empty",
        ),
        (
            "wrong_finish",
            "route_contract_error",
            "response_finish_reason",
            "not_examined",
        ),
        (
            "malformed",
            "route_contract_error",
            "invalid_route_json",
            "invalid_json",
        ),
        (
            "json_with_suffix",
            "route_contract_error",
            "invalid_route_json",
            "invalid_json",
        ),
        (
            "non_object",
            "route_contract_error",
            "invalid_route_json",
            "non_object",
        ),
        (
            "schema_invalid",
            "route_contract_error",
            "invalid_route_json",
            "schema_invalid",
        ),
        (
            "response_contract",
            "route_contract_error",
            "response_contract",
            "not_examined",
        ),
        (
            "route_budget",
            "route_contract_error",
            "route_budget",
            "not_examined",
        ),
        (
            "route_cap",
            "route_contract_error",
            "route_budget",
            "not_examined",
        ),
    ],
)
def test_captured_route_failures_are_typed_terminal_errors_without_retry(
    monkeypatch,
    canonical_registry_factory,
    variant: str,
    expected_error: str,
    expected_subtype: str,
    expected_payload_status: str,
) -> None:
    fixture = canonical_registry_factory(name=f"typed-route-{variant}")
    skill = _Skill(
        slug="s1-route-skill",
        capability_id="utility.document_reading",
        description="Read the visible document.",
        body="# Objective\n\nRead the document.",
        operators=(),
    )
    bank = _one_skill_bank(skill, "b")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
        max_output_tokens=(
            4 if variant == "route_budget" else 128 if variant == "route_cap" else 64
        ),
    )
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if variant == "length":
            return _response(
                request_id="route-length",
                text=canonical_json_bytes(
                    {"selected_capability": skill.capability_id}
                ).decode("utf-8"),
                output_tokens=64,
                finish_reason="length",
            )
        if variant == "wrong_function":
            return _route_response(
                request_id="route-wrong-function",
                selected_capability=skill.capability_id,
                function_name="other_route_function",
            )
        if variant == "extra_argument":
            return _route_response(
                request_id="route-extra-argument",
                selected_capability=skill.capability_id,
                arguments={
                    "selected_capability": skill.capability_id,
                    "skill_slug": skill.slug,
                },
            )
        if variant == "empty":
            return _response(
                request_id="route-empty",
                text="",
                output_tokens=1,
            )
        if variant == "wrong_finish":
            return _response(
                request_id="route-wrong-finish",
                text='{"selected_capability":"utility.document_reading"}',
                output_tokens=8,
                finish_reason="tool_calls",
            )
        if variant == "malformed":
            return _response(
                request_id="route-malformed",
                text="{not-json",
                output_tokens=8,
            )
        if variant == "json_with_suffix":
            return _response(
                request_id="route-json-with-suffix",
                text=(
                    canonical_json_bytes(
                        {"selected_capability": skill.capability_id}
                    ).decode("utf-8")
                    + "\nextra prose"
                ),
                output_tokens=12,
            )
        if variant == "non_object":
            return _response(
                request_id="route-non-object",
                text='["utility.document_reading"]',
                output_tokens=8,
            )
        if variant == "schema_invalid":
            return _response(
                request_id="route-schema-invalid",
                text='{"selected_capability":7}',
                output_tokens=8,
            )
        if variant == "response_contract":
            return _route_response(
                request_id="route-response-contract",
                selected_capability=skill.capability_id,
            ).model_copy(update={"response_model": "unexpected-model"})
        if variant == "route_budget":
            return _route_response(
                request_id="route-budget",
                selected_capability=skill.capability_id,
                output_tokens=8,
            )
        if variant == "route_cap":
            return _route_response(
                request_id="route-cap",
                selected_capability=skill.capability_id,
                output_tokens=65,
            )
        return _route_response(
            request_id="route-out-of-enum",
            selected_capability="product.exact_match",
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        banks={"s1": bank},
    ).execute(request)

    assert len(calls) == 1
    if variant == "route_cap":
        assert calls[0][2]["max_tokens"] == 64
    assert execution.response.error_code == expected_error
    assert execution.response.route_failure_subtype == expected_subtype
    assert execution.response.route_failure_shape is not None
    assert (
        execution.response.route_failure_shape.payload_status == expected_payload_status
    )
    assert (
        execution.response.route_failure_shape.normalized_response_sha256
        == execution.receipt.model_calls[0].response_sha256
    )
    assert execution.response.selected_capability is None
    assert execution.response.skill_slug is None
    assert execution.response.route_trace_sha256 is None
    assert len(execution.receipt.model_calls) == 1


def test_route_schema_and_fixed_retry_are_minimal_and_receipted(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="route-equivalent-retry")
    skill = _Skill(
        slug="private-route-skill",
        capability_id="utility.document_reading",
        description="Read visible document content.",
        body="# Objective\n\nRead the document.",
        operators=(),
    )
    bank = _one_skill_bank(skill, "b")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
        max_output_tokens=64,
    )
    schema = assistant_route_output_json_schema((skill.capability_id,))
    assert schema == {
        "type": "object",
        "properties": {
            "selected_capability": {
                "type": "string",
                "enum": [skill.capability_id],
            }
        },
        "required": ["selected_capability"],
        "additionalProperties": False,
    }

    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if len(calls) == 1:
            return _response(
                request_id="route-format-failure",
                text="{not-json",
                output_tokens=6,
            )
        if len(calls) == 2:
            return _route_response(
                request_id="route-retry-selected",
                selected_capability=skill.capability_id,
                output_tokens=7,
                arguments={
                    "selected_capability": skill.capability_id,
                    "asset_id": "echoed-asset",
                    "description": "echoed-description",
                    "text": "echoed-text",
                    "turns": [{"role": "user", "content": "echoed-turn"}],
                },
            )
        return _response(
            request_id="route-retry-action",
            text=(
                "answer:\nGrounded answer.\n"
                "evidence:\nGrounded answer.\n"
                "uncertainty:\nNo external tool evidence."
            ),
            output_tokens=8,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    runner = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks={"s1": bank},
    )
    first_context = _budget_context(tmp_path, request, name="route-retry-ledger")
    first = runner.execute(request, budget_context=first_context)
    first_evidence = first.receipt.route_call_evidence
    assert first.response.error_code == "route_contract_error"
    assert first_evidence is not None
    assert first_evidence.failure_reason == "invalid_route_json"
    assert first_evidence.payload_status == "invalid_json"
    assert base64.b64decode(first_evidence.response_text_prefix_base64) == b"{not-json"
    assert first_evidence.call_receipt.finish_reason == "stop"
    assert first_evidence.call_receipt.input_tokens == 11
    assert first_evidence.call_receipt.output_tokens == 6
    assert first_evidence.request_variant == "initial"
    assert first_evidence.repair_of_wire_request_sha256 is None
    assert first_evidence.ignored_response_keys == ()
    legacy_payload = first_evidence.model_dump(mode="json")
    legacy_payload["policy_version"] = "assistant-route-call-evidence-v1"
    legacy_payload.pop("request_variant")
    legacy_payload.pop("repair_of_wire_request_sha256")
    legacy_payload.pop("ignored_response_keys")
    legacy_payload.pop("evidence_sha256")
    legacy_evidence = AssistantRouteCallEvidence.model_validate(
        {
            **legacy_payload,
            "evidence_sha256": sha256_bytes(canonical_json_bytes(legacy_payload)),
        },
        strict=True,
    )
    legacy_serialized = legacy_evidence.model_dump(mode="json")
    assert "request_variant" not in legacy_serialized
    assert "repair_of_wire_request_sha256" not in legacy_serialized
    assert "ignored_response_keys" not in legacy_serialized

    second_context = PortfolioAssistantBudgetContext(
        ledger_root=first_context.ledger_root,
        shard_id=first_context.shard_id,
        instance_sha256=first_context.instance_sha256,
        attempt_index=2,
        repair_of_route_wire_request_sha256=first_evidence.wire_request_sha256,
    )
    second = runner.execute(request, budget_context=second_context)
    second_evidence = second.receipt.route_call_evidence
    assert second.response.error_code is None
    assert second_evidence is not None
    assert second_evidence.failure_reason is None
    assert second_evidence.attempt_index == 2
    assert second_evidence.request_variant == "fixed_repair"
    assert (
        second_evidence.repair_of_wire_request_sha256
        == first_evidence.wire_request_sha256
    )
    assert second_evidence.ignored_response_keys == (
        "asset_id",
        "description",
        "text",
        "turns",
    )
    assert second_evidence.wire_request_sha256 != first_evidence.wire_request_sha256
    assert calls[0][0] == calls[1][0]
    assert calls[0][1][0] != calls[1][1][0]
    assert calls[0][1][1] == calls[1][1][1]
    assert set(json.loads(calls[0][1][1]["content"])) == {"turns"}
    assert "One-time fixed JSON repair" in calls[1][1][0]["content"]
    assert "{not-json" not in calls[1][1][0]["content"]
    assert calls[0][2] == calls[1][2]
    assert calls[0][2]["images"] is None
    serialized = second_evidence.model_dump_json()
    assert skill.slug not in serialized
    assert str(fixture.asset_catalog.asset_root) not in serialized

    bad_root = tmp_path / "route-drift-ledger"
    initialize_portfolio_budget_ledger(
        bad_root,
        matrix_run_id=request.matrix_run_id,
        phase_cap_cny=Decimal("100.000000000000"),
    )
    bad_context = PortfolioAssistantBudgetContext(
        ledger_root=bad_root,
        shard_id=first_context.shard_id,
        instance_sha256=first_context.instance_sha256,
        attempt_index=2,
        repair_of_route_wire_request_sha256="0" * 64,
    )
    call_count = len(calls)
    with pytest.raises(
        PortfolioBudgetError,
        match="settled initial route identity",
    ):
        runner.execute(request, budget_context=bad_context)
    assert len(calls) == call_count


def test_nonretryable_provider_status_escapes_without_fixed_zero(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="fatal-provider-status")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
    )
    provider_request = httpx.Request("POST", request.backbone.endpoint)
    provider_response = httpx.Response(401, request=provider_request)

    def fake_chat(*args, **kwargs):
        raise APIStatusError(
            "fixture authentication failure",
            response=provider_response,
            body={"error": "redacted"},
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    budget_context = _budget_context(tmp_path, request)
    with pytest.raises(AssistantFatalProviderConfigurationError) as captured:
        _runner(
            registry=fixture.registry,
            catalog=fixture.asset_catalog,
            runner_type=PortfolioAssistantRunner,
        ).execute(request, budget_context=budget_context)

    assert captured.value.failure_stage == "action"
    assert captured.value.status_code == 401
    ledger = load_portfolio_budget_ledger(budget_context.ledger_root)
    assert len(ledger.reservations) == len(ledger.forfeits) == 1
    assert ledger.settlements == ()
    assert ledger.unresolved_reservations == ()
    assert (
        captured.value.forfeited_reservation_sha256
        == ledger.forfeits[0].reservation_sha256
    )
    assert captured.value.budget_forfeit_sha256 == ledger.forfeits[0].forfeit_sha256


def test_portfolio_budget_rejection_precedes_provider_call(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="budget-rejects-before-provider")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="noskill",
    )
    context = _budget_context_with_cap(
        tmp_path,
        request,
        name="reject-budget",
        cap="0.100000000000",
    )
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("provider must not be reached")

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    with pytest.raises(PortfolioBudgetError, match="explicit hard-budget context"):
        _runner(
            registry=fixture.registry,
            catalog=fixture.asset_catalog,
            runner_type=PortfolioAssistantRunner,
        ).execute(request)
    with pytest.raises(PortfolioBudgetExceededError):
        _runner(
            registry=fixture.registry,
            catalog=fixture.asset_catalog,
            runner_type=PortfolioAssistantRunner,
        ).execute(request, budget_context=context)

    assert calls == []
    ledger = load_portfolio_budget_ledger(context.ledger_root)
    assert ledger.reservations == ()
    assert ledger.settlements == ()


def test_captured_route_response_contract_error_is_already_settled(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="captured-route-settlement")
    skill = _Skill(
        slug="s1-document",
        capability_id="utility.document_reading",
        description="Read the visible document.",
        body="# Objective\n\nRead the document.",
        operators=(),
    )
    bank = _one_skill_bank(skill, "b")
    request = _request(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        config="s1",
        bank_sha256=bank.bank_sha256,
    )
    context = _budget_context(tmp_path, request, name="captured-budget")

    def fake_chat(*args, **kwargs):
        return _response(
            request_id="captured-invalid-route",
            text="",
            output_tokens=8,
            finish_reason="tool_calls",
            tool_calls=(
                LLMToolCall(
                    call_id="route-a",
                    name="submit_route_decision",
                    arguments_json=canonical_json_bytes(
                        {"selected_capability": skill.capability_id}
                    ).decode("utf-8"),
                ),
                LLMToolCall(
                    call_id="route-b",
                    name="submit_route_decision",
                    arguments_json=canonical_json_bytes(
                        {"selected_capability": skill.capability_id}
                    ).decode("utf-8"),
                ),
            ),
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    execution = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks={"s1": bank},
    ).execute(request, budget_context=context)

    assert execution.response.error_code == "route_contract_error"
    assert execution.response.route_failure_subtype == "response_tool_calls"
    assert execution.response.route_failure_shape is not None
    assert execution.response.route_failure_shape.tool_call_count == 2
    ledger = load_portfolio_budget_ledger(context.ledger_root)
    assert len(ledger.reservations) == len(ledger.settlements) == 1
    assert ledger.unresolved_reservations == ()
    assert ledger.reservations[0].identity.stage == "assistant_route"
    assert ledger.reservations[0].identity.wire_request_sha256 is not None


def test_s1s2_and_full_share_capability_route_with_equal_action_budget(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="shared-stage2-route")
    capability = "utility.document_reading"
    s1s2_skill = _Skill(
        slug="s1s2-document",
        capability_id=capability,
        description="Read and explain the visible document.",
        body="# S1 body\n\nUse the original frozen procedure.",
        operators=(),
    )
    full_skill = _Skill(
        slug="full-document",
        capability_id=capability,
        description=s1s2_skill.description,
        body="# S3 body\n\nUse the refined frozen procedure.",
        operators=(),
    )
    banks = {
        "s1s2": _Bank(
            bank_sha256="c" * 64,
            skills=(s1s2_skill,),
            capability_map=(
                _Binding(
                    capability_id=capability,
                    skill_slug=s1s2_skill.slug,
                ),
            ),
        ),
        "full": _Bank(
            bank_sha256="d" * 64,
            skills=(full_skill,),
            capability_map=(
                _Binding(
                    capability_id=capability,
                    skill_slug=full_skill.slug,
                ),
            ),
        ),
    }
    requests = {
        config: _request(
            registry=fixture.registry,
            catalog=fixture.asset_catalog,
            config=config,
            bank_sha256=banks[config].bank_sha256,
            query_id="shared-stage2-query",
            max_output_tokens=40,
            max_turns=2,
        )
        for config in ("s1s2", "full")
    }
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        prompt = messages[0]["content"]
        if len(calls) == 1:
            return _route_response(
                request_id="one-shared-stage2-call",
                selected_capability=capability,
                output_tokens=7,
            )
        answer = (
            "S1+S2 action answer."
            if "Use the original frozen procedure." in prompt
            else "Full action answer."
        )
        return _response(
            request_id=f"action-{len(calls)}",
            text=(
                f"answer:\n{answer}\n"
                f"evidence:\n{answer}\n"
                "uncertainty:\nNo external tool evidence."
            ),
            output_tokens=9,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    runner = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks=banks,
    )

    budget_context = _budget_context(tmp_path, requests["s1s2"])
    artifact = runner.prepare_shared_stage2_route(
        requests["s1s2"], budget_context=budget_context
    )
    s1s2 = runner.execute(
        requests["s1s2"],
        shared_stage2_route=artifact,
        budget_context=budget_context,
    )
    full = runner.execute(
        requests["full"],
        shared_stage2_route=artifact,
        budget_context=budget_context,
    )

    assert len(calls) == 3
    ledger = load_portfolio_budget_ledger(budget_context.ledger_root)
    assert [item.identity.stage for item in ledger.reservations] == [
        "shared_route",
        "assistant_action",
        "assistant_action",
    ]
    assert [item.identity.config for item in ledger.reservations] == [
        "s1s2",
        "s1s2",
        "full",
    ]
    assert len(ledger.settlements) == 3
    assert (
        artifact.policy_version
        == SHARED_STAGE2_ROUTE_POLICY_VERSION
        == "shared-stage2-route-v6"
    )
    route_prompt = calls[0][1][0]["content"]
    assert calls[0][2]["max_tokens"] == 40
    assert calls[0][2]["json_mode"] is True
    assert calls[0][2]["tools"] is None
    assert calls[0][2]["tool_choice"] is None
    assert calls[0][2]["images"] is None
    assert set(json.loads(calls[0][1][1]["content"])) == {"turns"}
    assert _SYSTEM_PROMPT not in route_prompt
    assert s1s2_skill.slug not in route_prompt
    assert full_skill.slug not in route_prompt
    assert '"capability_id":"utility.document_reading"' in route_prompt
    assert full_skill.description in route_prompt
    for action_prompt in (calls[1][1][0]["content"], calls[2][1][0]["content"]):
        assert s1s2_skill.slug not in action_prompt
        assert full_skill.slug not in action_prompt
        assert s1s2_skill.description not in action_prompt
    assert calls[1][2]["max_tokens"] == calls[2][2]["max_tokens"] == 33
    assert s1s2.response.error_code is None
    assert full.response.error_code is None
    assert s1s2.response.selected_capability == capability
    assert full.response.selected_capability == capability
    assert s1s2.response.skill_slug == s1s2_skill.slug
    assert full.response.skill_slug == full_skill.slug
    assert (
        s1s2.response.route_trace_sha256
        == full.response.route_trace_sha256
        == artifact.artifact_sha256
    )
    assert len(s1s2.receipt.model_calls) == len(full.receipt.model_calls) == 1
    assert (
        s1s2.receipt.aggregate_usage
        == full.receipt.aggregate_usage
        == LLMUsage(
            input_tokens=11,
            output_tokens=9,
        )
    )
    assert (
        s1s2.receipt.shared_route_reference
        == full.receipt.shared_route_reference
        == type(s1s2.receipt.shared_route_reference)(
            artifact_sha256=artifact.artifact_sha256,
            route_call_response_sha256=artifact.route_call.response_sha256,
            reserved_usage=LLMUsage(input_tokens=11, output_tokens=7),
        )
    )
    artifact_json = artifact.model_dump_json()
    assert requests["s1s2"].request_sha256 not in artifact_json
    assert requests["full"].request_sha256 not in artifact_json
    assert banks["s1s2"].bank_sha256 not in artifact_json
    assert banks["full"].bank_sha256 not in artifact_json
    assert s1s2_skill.slug not in artifact_json
    assert full_skill.slug not in artifact_json


def test_legacy_shared_route_is_limited_to_exact_paired_full_diagnostic(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="legacy-shared-stage2-route")
    capability = "utility.document_reading"
    description = "Read and explain the visible document."
    banks = {}
    for config, slug in (("s1s2", "s1s2-document"), ("full", "full-document")):
        skill = _Skill(
            slug=slug,
            capability_id=capability,
            description=description,
            body=f"# {config} body\n",
            operators=(),
        )
        banks[config] = _Bank(
            bank_sha256=("7" if config == "s1s2" else "8") * 64,
            skills=(skill,),
            capability_map=(_Binding(capability_id=capability, skill_slug=slug),),
        )

    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        if "Route using descriptions only" in messages[0]["content"]:
            return _route_response(
                request_id=f"legacy-source-route-{len(calls)}",
                selected_capability=capability,
            )
        return _response(
            request_id=f"legacy-action-{len(calls)}",
            text=(
                "answer:\nGrounded document answer.\n"
                "evidence:\nGrounded document answer.\n"
                "uncertainty:\nNo external tool evidence."
            ),
            output_tokens=8,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    runner = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks=banks,
    )

    def request(config: str, matrix_run_id: str) -> AssistantRequestSnapshot:
        return _request(
            registry=fixture.registry,
            catalog=fixture.asset_catalog,
            config=config,
            bank_sha256=banks[config].bank_sha256,
            query_id="legacy-shared-route-query",
            matrix_run_id=matrix_run_id,
            max_output_tokens=40,
            max_turns=2,
        )

    diagnostic_id = "portfolio-treatment-smoke-stage2-" + "9" * 16
    diagnostic_source_request = request("s1s2", diagnostic_id)
    diagnostic_budget = _budget_context(
        tmp_path, diagnostic_source_request, name="diagnostic-budget"
    )
    source = runner.prepare_shared_stage2_route(
        diagnostic_source_request, budget_context=diagnostic_budget
    )
    legacy_payload = source.model_dump(mode="json")
    legacy_payload["policy_version"] = "shared-stage2-route-v1"
    legacy_payload["route_call"]["finish_reason"] = "stop"
    legacy_payload.pop("route_call_evidence")
    legacy_payload.pop("artifact_sha256")
    legacy = SharedStage2RouteArtifact.model_validate(
        {
            **legacy_payload,
            "artifact_sha256": sha256_bytes(canonical_json_bytes(legacy_payload)),
        },
        strict=True,
    )

    execution = runner.execute(
        request("full", diagnostic_id),
        shared_stage2_route=legacy,
        budget_context=diagnostic_budget,
    )
    assert execution.response.error_code is None
    assert execution.response.route_trace_sha256 == legacy.artifact_sha256
    assert len(calls) == 2

    formal_id = "portfolio-dev-mini-200x5-formal"
    formal_source_request = request("s1s2", formal_id)
    formal_budget = _budget_context(
        tmp_path, formal_source_request, name="formal-budget"
    )
    formal_source = runner.prepare_shared_stage2_route(
        formal_source_request, budget_context=formal_budget
    )
    for superseded_policy in (
        "shared-stage2-route-v2",
        "shared-stage2-route-v3",
        "shared-stage2-route-v4",
        "shared-stage2-route-v5",
    ):
        superseded_payload = formal_source.model_dump(mode="json")
        superseded_payload["policy_version"] = superseded_policy
        if superseded_policy == "shared-stage2-route-v2":
            superseded_payload["route_call"]["finish_reason"] = "tool_calls"
        superseded_payload.pop("route_call_evidence")
        superseded_payload.pop("artifact_sha256")
        superseded_route = SharedStage2RouteArtifact.model_validate(
            {
                **superseded_payload,
                "artifact_sha256": sha256_bytes(
                    canonical_json_bytes(superseded_payload)
                ),
            },
            strict=True,
        )
        rejected_superseded = runner.execute(
            request("full", formal_id),
            shared_stage2_route=superseded_route,
            budget_context=formal_budget,
        )
        assert rejected_superseded.response.error_code == "runtime_error"
        assert rejected_superseded.receipt.model_calls == ()

    formal_payload = formal_source.model_dump(mode="json")
    formal_payload["policy_version"] = "shared-stage2-route-v1"
    formal_payload["route_call"]["finish_reason"] = "stop"
    formal_payload.pop("route_call_evidence")
    formal_payload.pop("artifact_sha256")
    formal_legacy = SharedStage2RouteArtifact.model_validate(
        {
            **formal_payload,
            "artifact_sha256": sha256_bytes(canonical_json_bytes(formal_payload)),
        },
        strict=True,
    )
    rejected = runner.execute(
        request("full", formal_id),
        shared_stage2_route=formal_legacy,
        budget_context=formal_budget,
    )
    assert rejected.response.error_code == "runtime_error"
    assert rejected.receipt.model_calls == ()


def test_terminal_shared_route_is_reused_with_zero_action_calls(
    monkeypatch,
    canonical_registry_factory,
    tmp_path,
) -> None:
    fixture = canonical_registry_factory(name="terminal-shared-stage2-route")
    capability = "utility.document_reading"
    description = "Read and explain the visible document."
    banks = {}
    for config, slug in (
        ("s1s2", "s1s2-document"),
        ("full", "full-document"),
    ):
        skill = _Skill(
            slug=slug,
            capability_id=capability,
            description=description,
            body=f"# {config} body",
            operators=(),
        )
        banks[config] = _Bank(
            bank_sha256=("e" if config == "s1s2" else "f") * 64,
            skills=(skill,),
            capability_map=(_Binding(capability_id=capability, skill_slug=slug),),
        )
    requests = {
        config: _request(
            registry=fixture.registry,
            catalog=fixture.asset_catalog,
            config=config,
            bank_sha256=banks[config].bank_sha256,
            query_id="terminal-shared-stage2-query",
            max_output_tokens=40,
            max_turns=2,
        )
        for config in ("s1s2", "full")
    }
    calls = []

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        return _route_response(
            request_id="terminal-shared-route-call",
            selected_capability=capability,
            arguments={
                "selected_capability": capability,
                "skill_slug": "model-must-not-select-this",
            },
            output_tokens=7,
        )

    monkeypatch.setattr("skillchain.llm.chat", fake_chat)
    runner = _runner(
        registry=fixture.registry,
        catalog=fixture.asset_catalog,
        runner_type=PortfolioAssistantRunner,
        banks=banks,
    )

    budget_context = _budget_context(tmp_path, requests["s1s2"])
    artifact = runner.prepare_shared_stage2_route(
        requests["s1s2"], budget_context=budget_context
    )
    executions = tuple(
        runner.execute(
            requests[config],
            shared_stage2_route=artifact,
            budget_context=budget_context,
        )
        for config in ("s1s2", "full")
    )

    assert len(calls) == 1
    assert artifact.status == "terminal_route_error"
    assert artifact.failure_subtype == "invalid_route_json"
    assert artifact.failure_shape is not None
    assert artifact.failure_shape.payload_status == "unexpected_keys"
    assert (
        artifact.failure_shape.normalized_response_sha256
        == artifact.route_call.response_sha256
    )
    assert artifact.selected_capability is None
    references = tuple(
        execution.receipt.shared_route_reference for execution in executions
    )
    assert references[0] == references[1]
    assert references[0] is not None
    assert references[0].artifact_sha256 == artifact.artifact_sha256
    assert references[0].reserved_usage == LLMUsage(
        input_tokens=11,
        output_tokens=7,
    )
    for execution in executions:
        assert execution.response.error_code == "route_contract_error"
        assert execution.response.route_failure_subtype == "invalid_route_json"
        assert execution.response.route_failure_shape == artifact.failure_shape
        assert execution.response.selected_capability is None
        assert execution.response.skill_slug is None
        assert execution.response.route_trace_sha256 is None
        assert execution.response.turn_count == 1
        assert execution.receipt.model_calls == ()
        assert execution.receipt.aggregate_usage == LLMUsage(
            input_tokens=0,
            output_tokens=0,
        )
