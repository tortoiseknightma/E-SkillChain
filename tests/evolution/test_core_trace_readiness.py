"""Focused readiness checks for the internal CoreTraceRecord adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import pytest
from pydantic import ValidationError

from skillchain import config as project_config
from skillchain.evaluation.assistant_runs import (
    AssistantRouteAttempt,
    create_runner_owned_assistant_run_bundle,
    load_verified_assistant_run_bundle,
)
from skillchain.evolution import models as trace_models
from skillchain.evolution.models import (
    CoreTraceError,
    CoreTraceRecord,
    CoreTraceRuleScore,
    build_core_trace_record,
)
from skillchain.llm import LLMResponse, LLMUsage
from skillchain.runners.assistant import ProductionAssistantRunner
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

_ROOT = Path(__file__).resolve().parents[2]
_PHASE4_HELPERS = runpy.run_path(
    str(_ROOT / "tests" / "evaluation" / "test_phase4_assurance.py")
)


def _runner_bundle(
    tmp_path: Path,
    monkeypatch,
    canonical_registry_factory,
    *,
    scenario: str,
):
    """Create a real runner-owned bundle without calling an external model."""

    name = f"core-trace-{scenario}"
    matrix_run_id = f"{name}-run"
    registry = canonical_registry_factory(name=name).registry
    runner_bank = _PHASE4_HELPERS["_runner_bank"]
    phase4_inputs_with_catalog = _PHASE4_HELPERS["_phase4_inputs_with_catalog"]
    verified_plan = _PHASE4_HELPERS["_verified_plan"]
    banks = {
        config: runner_bank(registry, config)
        for config in ("llm_static", "s1", "s1s2", "full")
    }
    inputs, catalog = phase4_inputs_with_catalog(
        tmp_path,
        name=name,
        matrix_run_id=matrix_run_id,
        cloud_upload_allowed=True,
    )
    system_prompt = "Core Trace test Assistant prompt."
    plan = verified_plan(
        tmp_path,
        registry,
        name=name,
        inputs=inputs,
        system_prompt_sha256=sha256_bytes(system_prompt.encode()),
        endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
        bank_sha256_by_config={
            config: bank.bank_sha256 for config, bank in banks.items()
        },
    )
    calls: list[tuple[object, object]] = []

    def response(text: str) -> LLMResponse:
        return LLMResponse(
            provider="qwen",
            endpoint=project_config.PROVIDER_ENDPOINTS["qwen"],
            requested_model="qwen-plus-frozen",
            response_model="qwen-plus-frozen",
            request_id=f"core-trace-{scenario}-{len(calls)}",
            text=text,
            usage=LLMUsage(input_tokens=17, output_tokens=9),
            finish_reason="stop",
            latency_ms=5,
        )

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages))
        prompt = messages[0]["content"]
        if "Route using descriptions only" in prompt:
            if scenario == "route_failed":
                return response("not canonical JSON")
            decision = {
                "kind": "route",
                "selected_capability": "utility.document_reading",
                "skill_slug": "s1-skill",
            }
        elif scenario == "tool_failed" and len(messages) == 2:
            # The selected S1 test bank deliberately declares no operators, so
            # this creates a retained permission_denied tool trace.
            decision = {
                "arguments": {"query": "document"},
                "kind": "tool",
                "response_text": None,
                "selected_capability": None,
                "skill_slug": None,
                "tool_name": "text_product_search",
            }
        elif scenario == "tool_failed":
            # A later malformed action turns the run into a runtime-error row;
            # the earlier selected route and tool trace must nevertheless stay.
            return response("not canonical JSON")
        else:
            decision = {
                "arguments": None,
                "kind": "final",
                "response_text": "A public answer.",
                "selected_capability": "utility.document_reading",
                "skill_slug": "s1-skill",
                "tool_name": None,
            }
        return response(canonical_json_bytes(decision).decode())

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
        output_dir=tmp_path / f"{name}-bundle",
    )
    return inputs, load_verified_assistant_run_bundle(
        created.root,
        plan,
        expected_manifest_sha256=created.external_manifest_sha256,
        registry=registry,
    )


def test_core_trace_adapter_records_success_and_opt_in_score_gate(
    tmp_path: Path, monkeypatch, canonical_registry_factory
) -> None:
    inputs, run = _runner_bundle(
        tmp_path,
        monkeypatch,
        canonical_registry_factory,
        scenario="success",
    )

    record = build_core_trace_record(inputs, run, query_ordinal=0)

    assert record.query_id == inputs.selected_queries[0].query_id
    assert record.component_id == inputs.selected_queries[0].leakage_group_id
    assert record.route_attempt.status == "selected"
    assert record.route_correctness == "correct"
    assert record.tool_outcome == "not_called"
    assert record.included_in_denominator is True
    assert record.evaluation is None

    score = CoreTraceRuleScore(rule_id="format-complete", score=1.0)
    with pytest.raises(CoreTraceError, match="allow_judge_outcomes=True"):
        build_core_trace_record(
            inputs,
            run,
            query_ordinal=0,
            rule_scores=(score,),
        )
    allowed = build_core_trace_record(
        inputs,
        run,
        query_ordinal=0,
        allow_judge_outcomes=True,
        rule_scores=(score,),
    )
    assert allowed.evaluation is not None
    assert allowed.evaluation.rule_scores == (score,)


def test_core_trace_adapter_preserves_failed_route_and_failed_tool_attribution(
    tmp_path: Path, monkeypatch, canonical_registry_factory
) -> None:
    route_inputs, route_run = _runner_bundle(
        tmp_path,
        monkeypatch,
        canonical_registry_factory,
        scenario="route_failed",
    )
    route_failure = build_core_trace_record(
        route_inputs, route_run, query_ordinal=0
    )
    assert route_failure.route_attempt.status == "failed"
    assert route_failure.selected_capability is None
    assert route_failure.route_correctness == "not_routed"
    assert route_failure.error_code == "runtime_error"
    assert route_failure.included_in_denominator is True

    tool_inputs, tool_run = _runner_bundle(
        tmp_path,
        monkeypatch,
        canonical_registry_factory,
        scenario="tool_failed",
    )
    tool_failure = build_core_trace_record(tool_inputs, tool_run, query_ordinal=0)
    assert tool_failure.error_code == "runtime_error"
    assert tool_failure.route_attempt.status == "selected"
    assert tool_failure.selected_capability == "utility.document_reading"
    assert tool_failure.route_trace_sha256 is not None
    assert tool_failure.route_correctness == "correct"
    assert tool_failure.tool_outcome == "error"
    assert tool_failure.tool_error_codes == ("permission_denied",)
    assert tool_failure.tool_trace[0].status == "error"
    assert tool_failure.included_in_denominator is True


def test_core_trace_rejects_tampered_route_hash_and_join_ids(
    tmp_path: Path, monkeypatch, canonical_registry_factory
) -> None:
    inputs, run = _runner_bundle(
        tmp_path,
        monkeypatch,
        canonical_registry_factory,
        scenario="success",
    )
    row = run.rows[0]
    assert row.execution_receipt is not None
    attempt = row.execution_receipt.route_attempt
    assert attempt is not None
    tampered_attempt = attempt.model_dump(mode="python")
    tampered_attempt["selected_capability"] = "product.exact_match"
    with pytest.raises(ValidationError, match="self hash mismatch"):
        AssistantRouteAttempt.model_validate(tampered_attempt, strict=True)

    baseline = build_core_trace_record(inputs, run, query_ordinal=0)
    tampered_record = baseline.model_dump(mode="python")
    tampered_record["query_id"] = "query-tampered"
    with pytest.raises(ValidationError, match="core trace self hash mismatch"):
        CoreTraceRecord.model_validate(tampered_record, strict=True)

    # The normal adapter re-verifies both immutable handles first.  Bypass only
    # that outer guard here to prove the inner join rejects a forged request,
    # result, receipt, or input-query identity before emitting a trace.
    monkeypatch.setattr(
        trace_models, "require_verified_phase4_inputs", lambda value: value
    )
    monkeypatch.setattr(
        trace_models, "require_verified_assistant_run_bundle", lambda value: value
    )
    bad_request = run.requests[0].model_copy(update={"matrix_run_id": "wrong-run"})
    with pytest.raises(CoreTraceError, match="do not join"):
        build_core_trace_record(
            inputs,
            replace(run, requests=(bad_request, *run.requests[1:])),
            query_ordinal=0,
        )

    bad_receipt = row.execution_receipt.model_copy(
        update={"request_sha256": "0" * 64}
    )
    bad_row = row.model_copy(update={"execution_receipt": bad_receipt})
    with pytest.raises(CoreTraceError, match="receipt and result do not join"):
        build_core_trace_record(
            inputs,
            replace(run, rows=(bad_row, *run.rows[1:])),
            query_ordinal=0,
        )

    bad_query = inputs.selected_queries[0].model_copy(
        update={"query_id": "query-tampered"}
    )
    with pytest.raises(CoreTraceError, match="do not join"):
        build_core_trace_record(
            replace(inputs, selected_queries=(bad_query, *inputs.selected_queries[1:])),
            run,
            query_ordinal=0,
        )
