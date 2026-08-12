from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from skillchain import config, llm
from skillchain.evaluation.core_fast import live_adapter as live_module
from skillchain.evaluation.core_fast.engine import CoreFastEngine
from skillchain.evaluation.core_fast.live_adapter import LiveCoreFastAdapter
from skillchain.evaluation.core_fast.models import (
    AssistantObservation,
    CallIntent,
    CallResult,
    Concurrency,
)
from skillchain.evaluation.core_fast.pacing import StartPacer
from skillchain.evaluation.core_fast.store import CallStore
from skillchain.evaluation.feedback_runtime import visual_feedback_response_format_v1
from skillchain.runners.assistant import CoreFastAssistantRunner
from skillchain.schemas import Query


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


@pytest.mark.parametrize(
    ("requests_per_second", "interval"),
    ((20.0, 0.05), (8.0, 0.125)),
)
def test_start_pacer_uses_measured_spacing_with_a_fake_clock(
    requests_per_second: float,
    interval: float,
) -> None:
    clock = _FakeClock()
    pacer = StartPacer(
        requests_per_second,
        clock=clock.monotonic,
        sleeper=clock.sleep,
    )

    for _ in range(4):
        pacer.wait()

    assert clock.sleeps == pytest.approx([interval, interval, interval])
    assert clock.now == pytest.approx(100.0 + 3 * interval)


def test_core_fast_capacity_model_is_bound_to_both_measured_profiles() -> None:
    capacity = Concurrency()

    assert (capacity.assistant, capacity.assistant_requests_per_second) == (60, 20.0)
    assert (capacity.feedback, capacity.feedback_requests_per_second) == (60, 8.0)


def test_repeated_multi_tool_calls_are_scored_not_misclassified_as_provider_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated valid tool calls stay in the population for GCS to reject."""

    adapter = LiveCoreFastAdapter(
        spec=SimpleNamespace(
            experiment_id="repeated-multi-test",
            concurrency=SimpleNamespace(assistant_requests_per_second=20.0),
        ),
        cwd=tmp_path,
        base_dir=tmp_path,
    )
    query = _query().model_copy(
        update={
            "canonical_intent": "multi_product",
            "canonical_capability": "product.multi_search",
            "acceptable_capabilities": ("product.multi_search",),
        }
    )
    first_expected = (SimpleNamespace(item_ref="item-001", label="gum"),)
    scorer_calls = (SimpleNamespace(call_index=1), SimpleNamespace(call_index=2))
    execution = SimpleNamespace(
        response=SimpleNamespace(
            response_text="answer",
            visible_cards=(),
            visible_tool_evidence=(),
            tool_trace=(),
            selected_capability="product.multi_search",
            skill_slug="static-product-multi-search",
            route_trace_sha256="a" * 64,
            registry_sha256="b" * 64,
            registry_runtime_sha256="c" * 64,
            backbone_provider="qwen",
            backbone_model=config.ASSISTANT_MODEL,
            backbone_request_id="request-id",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            latency_ms=1,
            error_code=None,
        ),
        receipt=SimpleNamespace(
            request_sha256="d" * 64,
            receipt_sha256="e" * 64,
        ),
        scorer_calls=scorer_calls,
    )
    monkeypatch.setattr(
        live_module,
        "expected_multi_items_from_call_v2",
        lambda call: first_expected if call.call_index == 1 else first_expected,
    )
    monkeypatch.setattr(
        live_module,
        "AssistantResult",
        lambda **kwargs: SimpleNamespace(
            model_dump=lambda **_dump_kwargs: {
                "query_id": kwargs["query_id"],
                "config": kwargs["config"],
            }
        ),
    )
    captured: dict[str, object] = {}

    def make_sidecar(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(live_module, "make_public_scorer_evidence_v2", make_sidecar)
    monkeypatch.setattr(
        live_module,
        "score_portfolio_gcs_v2",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(live_module, "build_gcs_population_v2", lambda _q: object())
    monkeypatch.setattr(live_module, "load_mvp_task_specification_v1", lambda: object())
    monkeypatch.setattr(live_module, "portfolio_gcs_oracles_v2", lambda: object())
    monkeypatch.setattr(adapter, "_query_sha256", lambda: "f" * 64)
    bank = SimpleNamespace(bank_sha256="0" * 64)

    _result, _score = adapter._assistant_result(
        query=query,
        config_name="llm_static",
        bank=bank,
        execution=execution,
    )

    assert captured["calls"] == scorer_calls
    assert captured["expected_multi_items"] == first_expected


def test_live_adapter_injects_one_global_pacer_into_cached_assistant_runners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = LiveCoreFastAdapter(
        spec=SimpleNamespace(
            concurrency=SimpleNamespace(assistant_requests_per_second=0.5),
            runtime=SimpleNamespace(
                assistant_contract="core-fast-deterministic-action-response-v1"
            ),
        ),
        cwd=tmp_path,
        base_dir=tmp_path,
    )
    pacer = _CountingPacer()
    adapter._assistant_start_pacer = pacer
    adapter._load_runtime = lambda: (
        SimpleNamespace(registry=object()),
        object(),
    )
    captured: list[dict[str, object]] = []
    runner = object()

    def _runner_factory(**kwargs: object) -> object:
        captured.append(kwargs)
        return runner

    monkeypatch.setattr(live_module, "CoreFastAssistantRunner", _runner_factory)
    monkeypatch.setattr(
        live_module,
        "require_core_fast_assistant_runner",
        lambda value: value,
    )
    monkeypatch.setattr(live_module.time, "monotonic", lambda: 42.0)
    bank = SimpleNamespace(bank_sha256="verified-bank")

    assert adapter._runner(bank) is runner
    assert adapter._runner(bank) is runner
    assert len(captured) == 1
    assert (
        captured[0]["deterministic_action_contract_version"]
        == "core-fast-deterministic-action-response-v1"
    )
    waiter = captured[0]["qwen_call_start_waiter"]
    assert callable(waiter)
    assert waiter("provider-call") == 42.0
    assert pacer.calls == 1


class _CountingPacer:
    def __init__(self) -> None:
        self.calls = 0

    def wait(self) -> None:
        self.calls += 1


class _CountingAdapter:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {"assistant": 0, "feedback": 0}

    def invoke(self, intent: CallIntent) -> CallResult:
        self.calls[intent.role] += 1
        return CallResult(
            call_id=intent.call_id,
            role=intent.role,
            status="success",
            requested_model=intent.requested_model,
            returned_model=intent.requested_model,
            raw_output="{}",
            schema_valid=True,
            output={},
        )


def test_cached_calls_do_not_consume_pacing_or_live_adapter_slots(
    tmp_path: Path,
) -> None:
    roles = {
        role: SimpleNamespace(
            requested_model=(
                config.FEEDBACK_JUDGE_MODEL
                if role == "feedback"
                else config.ASSISTANT_MODEL
            ),
            estimated_call_cost_cny=0.01,
        )
        for role in ("assistant", "feedback")
    }
    spec = SimpleNamespace(
        models=roles,
        limits=SimpleNamespace(
            external_cost_cny=10.0,
            max_feedback_calls=12,
            max_creator_calls=3,
        ),
    )
    adapter = _CountingAdapter()
    pacer = _CountingPacer()
    engine = object.__new__(CoreFastEngine)
    engine.spec = spec
    engine.calls = CallStore(tmp_path, spec)
    engine.adapter = adapter
    engine._feedback_start_pacer = pacer

    for role in ("feedback", "assistant"):
        first = engine._call(
            role=role,
            call_id=f"cached-{role}",
            purpose="cache boundary test",
            payload={},
        )
        second = engine._call(
            role=role,
            call_id=f"cached-{role}",
            purpose="cache boundary test",
            payload={},
        )
        assert second == first
        assert adapter.calls[role] == 1

    assert pacer.calls == 1


def _query() -> Query:
    return Query(
        schema_version=2,
        taxonomy_version="ecommerce-mvp-taxonomy-v0",
        task_spec_version="ecommerce-task-spec-v1",
        query_id="wire-query-001",
        asset_id="asset.wire-query-001",
        image_path="query_images/product.exact_match/wire-query-001.png",
        leakage_group_id="leak-wire-query-001",
        boundary_group_id=None,
        template_family="wire-test",
        generator_batch_id="wire-test-batch",
        text="Find the exact product.",
        turns=[{"role": "user", "content": "Find the exact product."}],
        canonical_intent="exact_match",
        canonical_capability="product.exact_match",
        acceptable_capabilities=["product.exact_match"],
        is_boundary=False,
        boundary_strategy=None,
        requires_card=True,
        split="opt_pool",
        label_status="auto",
        label_provenance=[
            {
                "decision_type": "constructed",
                "annotator_kind": "planner",
                "annotator_id": "core-fast-wire-test",
                "canonical_intent": "exact_match",
                "canonical_capability": "product.exact_match",
                "acceptable_capabilities": ["product.exact_match"],
            }
        ],
    )


def _openai_client(
    calls: list[dict[str, object]],
    *,
    content: str,
    finish_reason: str,
    tool_calls: list[object] | None = None,
) -> object:
    def _create(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            id="captured-request",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    message=SimpleNamespace(
                        content=content,
                        tool_calls=tool_calls,
                        reasoning_content=None,
                        refusal=None,
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=17,
                completion_tokens=5,
                total_tokens=22,
                completion_tokens_details=None,
            ),
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )


def test_core_fast_assistant_runner_emits_the_measured_qwen_wire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    tool_call = SimpleNamespace(
        id="tool-call-1",
        function=SimpleNamespace(
            name="image_product_search",
            arguments='{"asset_id":"asset.wire-query-001"}',
        ),
    )
    monkeypatch.setattr(
        llm,
        "_client",
        lambda _provider: _openai_client(
            calls,
            content="",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        ),
    )
    monkeypatch.setattr(llm, "_log_usage", lambda _response: None)
    starts: list[str] = []
    verified_assets: list[tuple[str, ...]] = []
    runner = object.__new__(CoreFastAssistantRunner)
    runner._asset_catalog = SimpleNamespace(
        verify_asset_ids=lambda values: verified_assets.append(tuple(values))
    )
    runner._qwen_call_start_waiter = lambda label: starts.append(label) or 0.0
    image = tmp_path / "query.png"
    image.write_bytes(b"not-a-real-provider-image")
    tool = {
        "type": "function",
        "function": {
            "name": "image_product_search",
            "description": "search",
            "parameters": {"type": "object", "additionalProperties": True},
        },
    }
    request = SimpleNamespace(
        query=SimpleNamespace(query_id="wire-query-001"),
        backbone=SimpleNamespace(
            provider="qwen",
            model=config.ASSISTANT_MODEL,
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            temperature=0.0,
            top_p=1.0,
            seed=None,
        ),
    )

    response = runner._chat(
        request,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        4096,
        image,
        "asset.wire-query-001",
        json_mode=False,
        tools=[tool],
        attach_image=True,
        timeout_seconds=180,
        failure_stage="action",
    )

    assert response.finish_reason == "tool_calls"
    assert starts == ["core-fast:wire-query-001:action"]
    assert verified_assets == [("asset.wire-query-001",)]
    assert len(calls) == 1
    wire = calls[0]
    assert wire["model"] == config.ASSISTANT_MODEL
    assert wire["max_tokens"] == 4096
    assert wire["temperature"] == 0.0
    assert wire["top_p"] == 1.0
    assert wire["tools"] == [tool]
    assert wire["tool_choice"] == "auto"
    assert wire["parallel_tool_calls"] is False
    assert wire["extra_body"] == {"enable_thinking": False}
    assert wire["timeout"] == 180
    assert "max_completion_tokens" not in wire
    assert "stream" not in wire
    image_part = wire["messages"][1]["content"][0]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


class _ParsedFeedback:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"policy_labeled": True}


class _AssistantResultStub:
    @classmethod
    def model_validate_json(cls, _content: bytes, *, strict: bool) -> object:
        assert strict
        return object()


def test_live_feedback_emits_exact_qwen38_capacity_probe_wire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        llm,
        "_client",
        lambda _provider: _openai_client(
            calls,
            content='{"feedback":"captured"}',
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr(live_module, "AssistantResult", _AssistantResultStub)
    monkeypatch.setattr(
        live_module,
        "build_feedback_packet_v3",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        live_module,
        "build_feedback_evaluator_prompt_v6",
        lambda _packet: object(),
    )
    expected_messages = [
        {"role": "system", "content": "feedback system"},
        {"role": "user", "content": "feedback user"},
    ]
    monkeypatch.setattr(
        live_module,
        "evaluator_wire_messages",
        lambda _prompt, *, image_bytes: expected_messages,
    )
    monkeypatch.setattr(
        live_module,
        "parse_visual_feedback_output_v3",
        lambda _text: _ParsedFeedback(),
    )
    policy_checks: list[object] = []
    monkeypatch.setattr(
        live_module,
        "require_policy_labeled_suggestions",
        lambda parsed: policy_checks.append(parsed),
    )
    adapter = LiveCoreFastAdapter(
        spec=SimpleNamespace(
            concurrency=SimpleNamespace(assistant_requests_per_second=0.5)
        ),
        cwd=tmp_path,
        base_dir=tmp_path,
    )
    adapter._load_runtime = lambda: (object(), object())
    adapter._rubric_value = lambda: object()
    adapter._feedback_gcs_contract = lambda _capability: object()
    image = tmp_path / "feedback.png"
    image.write_bytes(b"feedback-image")
    adapter._image_path = lambda _query: image
    query = _query()
    components = {
        "route_acceptable": True,
        "no_hard_error": True,
        "tool_contract_pass": True,
        "evidence_grounded": False,
        "output_contract_pass": False,
    }
    baseline = AssistantObservation(
        query_id=query.query_id,
        response_text="baseline",
        selected_capability=query.canonical_capability,
        route_trace_key="route",
        tool_trace_key="tool",
        replay_context={"assistant_result": {}},
        answer_mode="unresolved",
        gcs_components=components,
        gcs_reason_codes=("unsupported_claim",),
        gcs_score=0.0,
        hard_error=False,
        evidence_violation=True,
    )
    intent = CallIntent(
        call_id="feedback-wire",
        role="feedback",
        purpose="capture live Feedback wire",
        requested_model=config.FEEDBACK_JUDGE_MODEL,
        payload={
            "query": query.model_dump(mode="json"),
            "baseline": baseline.model_dump(mode="json"),
            "sample_role": "failure",
        },
    )

    result = adapter._invoke_feedback(intent)

    assert result.status == "success"
    assert result.returned_model == config.FEEDBACK_JUDGE_MODEL
    assert len(policy_checks) == 1
    assert len(calls) == 1
    wire = calls[0]
    assert wire["model"] == config.FEEDBACK_JUDGE_MODEL
    assert wire["messages"] == expected_messages
    assert wire["max_completion_tokens"] == 6144
    assert wire["response_format"] == visual_feedback_response_format_v1().model_dump(
        mode="json", by_alias=True
    )
    assert wire["stream"] is False
    assert wire["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 2048,
    }
    assert wire["timeout"] == 600
    for omitted in ("max_tokens", "temperature", "top_p", "seed", "stream_options"):
        assert omitted not in wire
