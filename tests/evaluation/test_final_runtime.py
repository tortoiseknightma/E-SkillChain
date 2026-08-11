from __future__ import annotations

import base64
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillchain import config, llm
from skillchain.evaluation import final_runtime as final_runtime_module
from skillchain.evaluation.evaluator_isolation import (
    make_active_portfolio_evaluator_isolation_lock,
)
from skillchain.evaluation.final_runtime import (
    CARD_REQUIREMENT_GUARD_POLICY_SHA256,
    CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    FINAL_JUDGE_ANSWER_MAX_TOKENS,
    FINAL_JUDGE_MAX_ATTEMPTS,
    FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS,
    FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS,
    FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE,
    FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_LIMIT,
    FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE,
    FINAL_JUDGE_RETRY_POLICY_SHA256,
    FINAL_JUDGE_RETRY_POLICY_SHA256_V3,
    FINAL_JUDGE_RETRY_POLICY_VERSION,
    FINAL_JUDGE_RETRY_POLICY_VERSION_V3,
    FINAL_JUDGE_THINKING_BUDGET,
    FINAL_JUDGE_TIMEOUT_SECONDS,
    FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
    FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
    FinalJudgeBudgetContext,
    FinalJudgeEvaluationResult,
    FinalJudgeRuntimeError,
    load_final_judge_evaluation_result,
    run_visual_final_judge as _run_visual_final_judge,
    write_final_judge_evaluation_result,
)
from skillchain.evaluation.evaluator_outputs import (
    FINAL_JUDGE_PARSER_POLICY_SHA256_V2,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V2,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
)
from skillchain.evaluation.packets import (
    EvaluationImage,
    FinalEvaluationPacket,
    RubricSnapshot,
    VisibleCard,
)
from skillchain.evaluation.portfolio_execution import (
    PortfolioBudgetError,
    PortfolioBudgetExceededError,
    initialize_portfolio_budget_ledger,
    load_portfolio_budget_ledger,
)
from skillchain.llm import LLMResponse, LLMUsage
from skillchain.schemas import ConversationTurn
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


IMAGE_BYTES = b"final-judge-image"
_ACTIVE_BUDGET_CONTEXT: FinalJudgeBudgetContext | None = None
_FAKE_BUDGET_EVENTS: list[tuple[str, object]] = []


def _make_budget_context(
    root: Path,
    *,
    phase_cap_cny: Decimal = Decimal("1000"),
    attempt_index: int = 1,
) -> FinalJudgeBudgetContext:
    ledger_root = root / "budget-ledger"
    matrix_run_id = f"final-runtime-{sha256_bytes(str(root).encode())[:16]}"
    initialize_portfolio_budget_ledger(
        ledger_root,
        matrix_run_id=matrix_run_id,
        phase_cap_cny=phase_cap_cny,
    )
    return FinalJudgeBudgetContext(
        ledger_root=ledger_root,
        matrix_run_id=matrix_run_id,
        shard_id="dev-mini-001",
        config="Full",
        query_id="dm-001",
        instance_sha256="a" * 64,
        request_sha256="b" * 64,
        attempt_index=attempt_index,
    )


@pytest.fixture(autouse=True)
def _provide_final_judge_budget_context(tmp_path, monkeypatch, request):
    global _ACTIVE_BUDGET_CONTEXT
    _FAKE_BUDGET_EVENTS.clear()
    _ACTIVE_BUDGET_CONTEXT = _make_budget_context(tmp_path)
    if request.node.name != "test_final_judge_budget_rejection_prevents_provider_call":

        def fake_reserve(_root, *, identity, final_judge_max_output_tokens):
            reservation = SimpleNamespace(
                reservation_sha256=identity.identity_sha256,
                identity=identity,
                pricing_profile=SimpleNamespace(
                    max_output_tokens=final_judge_max_output_tokens
                ),
            )
            _FAKE_BUDGET_EVENTS.append(("reserve", reservation))
            return reservation, True

        def fake_settle(_root, **kwargs):
            _FAKE_BUDGET_EVENTS.append(("settle", SimpleNamespace(**kwargs)))

        def fake_forfeit(_root, **kwargs):
            receipt = SimpleNamespace(
                **kwargs,
                forfeit_sha256=sha256_bytes(
                    canonical_json_bytes({"fixture_forfeit": kwargs})
                ),
            )
            _FAKE_BUDGET_EVENTS.append(("forfeit", receipt))
            return receipt, True

        monkeypatch.setattr(
            final_runtime_module, "reserve_portfolio_provider_call", fake_reserve
        )
        monkeypatch.setattr(
            final_runtime_module,
            "settle_portfolio_provider_call_success",
            fake_settle,
        )
        monkeypatch.setattr(
            final_runtime_module, "forfeit_portfolio_provider_call", fake_forfeit
        )
    try:
        yield
    finally:
        _ACTIVE_BUDGET_CONTEXT = None


def run_visual_final_judge(*args, budget_context=None, **kwargs):
    """Keep ordinary tests concise while exercising the required hard cap."""

    context = budget_context or _ACTIVE_BUDGET_CONTEXT
    assert context is not None
    return _run_visual_final_judge(
        *args,
        budget_context=context,
        **kwargs,
    )


def test_final_judge_token_reserves_are_frozen_and_bounded() -> None:
    assert FINAL_JUDGE_THINKING_BUDGET is None
    assert FINAL_JUDGE_ANSWER_MAX_TOKENS == 2_048
    assert FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS == 32_768
    assert FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE is None
    assert FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS == 2_048
    assert FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE is None
    assert FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_LIMIT is None


def test_visual_final_runner_rejects_answer_budget_drift_before_loading_image() -> None:
    with pytest.raises(FinalJudgeRuntimeError, match="max_tokens must remain frozen"):
        run_visual_final_judge(
            _packet(),
            make_active_portfolio_evaluator_isolation_lock(),
            remote_runtime=_runtime(),
            max_tokens=FINAL_JUDGE_ANSWER_MAX_TOKENS + 1,
            record_usage=False,
        )


def test_final_judge_budget_rejection_prevents_provider_call(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet()
    runtime = _runtime()
    context = _make_budget_context(
        tmp_path / "blocked-first-call",
        phase_cap_cny=Decimal("1"),
    )
    calls = 0

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )

    def fake_chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not be called without a reservation")

    monkeypatch.setattr(llm, "chat", fake_chat)

    with pytest.raises(PortfolioBudgetError, match="pricing and provider token"):
        run_visual_final_judge(
            packet,
            make_active_portfolio_evaluator_isolation_lock(),
            remote_runtime=runtime,
            budget_context=context,
            record_usage=False,
        )

    state = load_portfolio_budget_ledger(context.ledger_root)
    assert calls == 0
    assert state.reservations == ()
    assert state.settlements == ()


@pytest.mark.parametrize(
    ("initial_text", "retry_reason"),
    [
        ("", "empty_final_response"),
        ("```json\n{}\n```", "invalid_judge_json"),
    ],
)
def test_retryable_judge_answer_reserves_and_settles_each_captured_attempt(
    monkeypatch,
    tmp_path,
    initial_text: str,
    retry_reason: str,
) -> None:
    packet = _packet()
    runtime = _runtime()
    context = _make_budget_context(
        tmp_path / "two-attempts",
        phase_cap_cny=Decimal("5"),
        attempt_index=7,
    )
    responses = iter(
        (
            LLMResponse(
                provider="kimi",
                endpoint=config.PROVIDER_ENDPOINTS["kimi"],
                requested_model=config.PORTFOLIO_JUDGE_MODEL,
                response_model=config.PORTFOLIO_JUDGE_MODEL,
                request_id="req-budget-empty",
                text=initial_text,
                usage=LLMUsage(input_tokens=20, output_tokens=65),
                finish_reason="stop",
                latency_ms=3,
            ),
            LLMResponse(
                provider="kimi",
                endpoint=config.PROVIDER_ENDPOINTS["kimi"],
                requested_model=config.PORTFOLIO_JUDGE_MODEL,
                response_model=config.PORTFOLIO_JUDGE_MODEL,
                request_id="req-budget-scored",
                text=_submission(requires_card=False),
                usage=LLMUsage(input_tokens=30, output_tokens=12),
                finish_reason="stop",
                latency_ms=4,
            ),
        )
    )
    captured_responses: list[LLMResponse] = []

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )

    def fake_chat(*_args, **_kwargs):
        response = next(responses)
        captured_responses.append(response)
        return response

    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        budget_context=context,
        record_usage=False,
    )

    assert result.outcome.status == "scored"
    assert result.initial_empty_response is not None
    assert result.initial_empty_response.retry_reason == retry_reason
    reservations = [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "reserve"]
    settlements = [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "settle"]
    assert len(reservations) == len(settlements) == 2
    assert [item.identity.attempt_index for item in reservations] == [7, 7]
    assert [item.identity.call_index for item in reservations] == [1, 2]
    assert all(
        item.identity.stage == "final_judge"
        and item.identity.wire_request_sha256 == result.wire_sha256
        and item.pricing_profile.max_output_tokens is None
        for item in reservations
    )
    assert [
        (item.actual_input_tokens, item.actual_output_tokens) for item in settlements
    ] == [(20, 65), (30, 12)]
    assert [item.response_sha256 for item in settlements] == [
        sha256_bytes(canonical_json_bytes(response.model_dump(mode="json")))
        for response in captured_responses
    ]


@pytest.mark.parametrize(
    "initial_text",
    ["", "```json\n{}\n```"],
)
def test_second_judge_retry_budget_rejection_returns_no_score(
    monkeypatch,
    tmp_path,
    initial_text: str,
) -> None:
    packet = _packet()
    runtime = _runtime()
    context = _make_budget_context(
        tmp_path / "blocked-retry",
        phase_cap_cny=Decimal("1.713"),
    )
    calls = 0

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )

    def fake_chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-budget-only-first",
            text=initial_text,
            usage=LLMUsage(input_tokens=20, output_tokens=65),
            finish_reason="stop",
            latency_ms=3,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    original_reserve = final_runtime_module.reserve_portfolio_provider_call
    reservation_attempts = 0

    def reject_second_reservation(*args, **kwargs):
        nonlocal reservation_attempts
        reservation_attempts += 1
        if reservation_attempts == 2:
            raise PortfolioBudgetExceededError("fixture second reservation blocked")
        return original_reserve(*args, **kwargs)

    monkeypatch.setattr(
        final_runtime_module,
        "reserve_portfolio_provider_call",
        reject_second_reservation,
    )

    with pytest.raises(PortfolioBudgetExceededError):
        run_visual_final_judge(
            packet,
            make_active_portfolio_evaluator_isolation_lock(),
            remote_runtime=runtime,
            budget_context=context,
            record_usage=False,
        )

    assert calls == 1
    assert reservation_attempts == 2
    assert [kind for kind, _ in _FAKE_BUDGET_EVENTS] == ["reserve", "settle"]


def test_captured_response_is_settled_before_response_contract_error(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet()
    runtime = _runtime()
    context = _make_budget_context(
        tmp_path / "captured-contract-error",
        phase_cap_cny=Decimal("5"),
    )
    response = LLMResponse(
        provider="kimi",
        endpoint=config.PROVIDER_ENDPOINTS["kimi"],
        requested_model=config.PORTFOLIO_JUDGE_MODEL,
        response_model=config.PORTFOLIO_JUDGE_MODEL,
        request_id="req-budget-contract-error",
        text=_submission(requires_card=False),
        usage=LLMUsage(input_tokens=30, output_tokens=12),
        finish_reason="stop",
        latency_ms=4,
    )

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(llm, "chat", lambda *_, **__: response)
    monkeypatch.setattr(
        final_runtime_module,
        "parse_final_judge_output_v4",
        lambda *_, **__: (_ for _ in ()).throw(
            llm.LLMContractError("captured response violated a later contract")
        ),
    )

    with pytest.raises(llm.LLMContractError, match="later contract"):
        run_visual_final_judge(
            packet,
            make_active_portfolio_evaluator_isolation_lock(),
            remote_runtime=runtime,
            budget_context=context,
            record_usage=False,
        )

    settlements = [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "settle"]
    assert len(settlements) == 1
    assert settlements[0].provider_request_id == response.request_id
    assert settlements[0].response_sha256 == sha256_bytes(
        canonical_json_bytes(response.model_dump(mode="json"))
    )


def _hash_payload(payload: dict[str, object]) -> str:
    def jsonable(value):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    return sha256_bytes(canonical_json_bytes(jsonable(payload)))


def _resign_result(payload: dict[str, object]) -> dict[str, object]:
    unsigned = {key: value for key, value in payload.items() if key != "result_sha256"}
    payload["result_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    return payload


def _as_historical_kimi_result(payload: dict[str, object]) -> dict[str, object]:
    """Restore the former Kimi request identity before schema downgrades."""

    payload["provider"] = "kimi"
    payload["model"] = "kimi-k2.6"
    payload["endpoint"] = config.PROVIDER_ENDPOINTS["kimi"]
    payload["thinking_budget"] = 6_144
    payload["max_billable_output_tokens"] = 8_192
    payload["retry_policy_version"] = FINAL_JUDGE_RETRY_POLICY_VERSION_V3
    payload["retry_policy_sha256"] = FINAL_JUDGE_RETRY_POLICY_SHA256_V3
    payload["outcome"]["judge_provider"] = "kimi"
    payload["outcome"]["judge_model"] = "kimi-k2.6"
    payload.pop("transport_policy_version", None)
    payload.pop("transport_policy_sha256", None)
    payload.pop("requested_response_format", None)
    payload.pop("forfeited_reservation_sha256", None)
    payload.pop("budget_forfeit_sha256", None)
    return payload


def _packet(
    *,
    requires_card: bool = False,
    visible_card_count: int | None = None,
) -> FinalEvaluationPacket:
    rubric_text = "Score TCR, CCC when applicable, CQ, and CA."
    rubric = RubricSnapshot(
        rubric_id="portfolio-final-rubric-v1",
        rubric_version="1",
        content=rubric_text,
        content_sha256=hashlib.sha256(rubric_text.encode()).hexdigest(),
    )
    if visible_card_count is None:
        visible_card_count = 1 if requires_card else 0
    cards = tuple(
        VisibleCard(title=f"Card {index}", body="Visible grounded card.")
        for index in range(visible_card_count)
    )
    payload: dict[str, object] = {
        "schema_version": 3,
        "packet_kind": "final",
        "cache_namespace": "final-evaluator-v2",
        "evaluation_id": "e" * 64,
        "turns": (ConversationTurn(role="user", content="图中是什么商品？"),),
        "image": EvaluationImage(
            mime_type="image/png",
            sha256=hashlib.sha256(IMAGE_BYTES).hexdigest(),
        ),
        "response_text": "这是用户可见回答。",
        "cards": cards,
        "tool_evidence": (),
        "card_requirement": "required" if requires_card else "forbidden",
        "rubric": rubric,
    }
    return FinalEvaluationPacket.model_validate(
        {**payload, "packet_sha256": _hash_payload(payload)},
        strict=True,
    )


def _runtime():
    return SimpleNamespace(
        authorization=SimpleNamespace(authorization_id="portfolio-judge-v2"),
        authorization_file_sha256="b" * 64,
        receipt_file_sha256="c" * 64,
        receipt=SimpleNamespace(receipt_sha256="d" * 64),
        catalog=SimpleNamespace(catalog_sha256="a" * 64),
    )


def _submission(
    *,
    requires_card: bool,
    shape: str = "assessment_array",
) -> str:
    dimensions = [
        {"dimension": "CA", "score": 8},
        {"dimension": "CQ", "score": 14},
        {"dimension": "TCR", "score": 9},
    ]
    if requires_card:
        dimensions.insert(1, {"dimension": "CCC", "score": 7})
    if shape == "assessment_array":
        raw_dimensions = dimensions
    elif shape == "score_mapping":
        raw_dimensions = {item["dimension"]: item["score"] for item in dimensions}
    elif shape == "dimension_score_pairs":
        raw_dimensions = [[item["dimension"], item["score"]] for item in dimensions]
    else:
        raise AssertionError(f"unsupported test shape: {shape}")
    return json.dumps(
        {
            "schema_version": 1,
            "requires_card": requires_card,
            "dimensions": raw_dimensions,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _bare_submission(
    *,
    requires_card: bool,
    shape: str,
) -> str:
    wrapped = json.loads(_submission(requires_card=requires_card, shape=shape))
    return json.dumps(
        wrapped["dimensions"],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    ("requires_card", "expected_j"),
    [(False, 77.5), (True, 76.0)],
)
def test_visual_final_runner_compiles_scores_and_receipt(
    monkeypatch,
    tmp_path,
    requires_card: bool,
    expected_j: float,
) -> None:
    packet = _packet(requires_card=requires_card)
    runtime = _runtime()
    calls: list[tuple[str, list[dict], dict]] = []
    image_loads: list[tuple[object, str, object]] = []

    def fake_load(remote_runtime, *, processor, image):
        image_loads.append((remote_runtime, processor, image))
        return remote_runtime, IMAGE_BYTES

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id=f"req-final-{requires_card}",
            text=_submission(requires_card=requires_card),
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        )

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        fake_load,
    )
    monkeypatch.setattr(llm, "chat", fake_chat)
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert image_loads == [(runtime, "aifast-gemini-judge", packet.image)]
    assert result.outcome.status == "scored"
    assert result.outcome.scores.j_project == expected_j
    assert result.outcome.scores.evaluation_id == packet.evaluation_id
    assert result.parsed_submission is not None
    assert result.schema_version == 10
    assert result.cache_namespace == "final-evaluator-v11"
    assert result.max_tokens == FINAL_JUDGE_ANSWER_MAX_TOKENS
    assert result.thinking_budget == FINAL_JUDGE_THINKING_BUDGET
    assert result.max_billable_input_tokens == FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS
    assert result.max_billable_output_tokens == FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS
    assert result.attempts == 1
    assert result.max_attempts == FINAL_JUDGE_MAX_ATTEMPTS
    assert result.retry_policy_version == FINAL_JUDGE_RETRY_POLICY_VERSION
    assert result.retry_policy_sha256 == FINAL_JUDGE_RETRY_POLICY_SHA256
    assert result.transport_policy_version == FINAL_JUDGE_TRANSPORT_POLICY_VERSION
    assert result.transport_policy_sha256 == FINAL_JUDGE_TRANSPORT_POLICY_SHA256
    assert result.requested_response_format == "json_object"
    assert result.initial_empty_response is None
    assert result.forfeited_reservation_sha256 is None
    assert result.budget_forfeit_sha256 is None
    assert result.captured_response_count == 1
    assert result.aggregate_usage == LLMUsage(input_tokens=30, output_tokens=12)
    assert result.parser_policy_version == FINAL_JUDGE_PARSER_POLICY_VERSION_V4
    assert result.parser_policy_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V4
    assert result.visible_card_count == (1 if requires_card else 0)
    assert (
        result.card_requirement_guard_policy_version
        == CARD_REQUIREMENT_GUARD_POLICY_VERSION
    )
    assert (
        result.card_requirement_guard_policy_sha256
        == CARD_REQUIREMENT_GUARD_POLICY_SHA256
    )
    assert result.card_requirement_guard_adjusted is False
    assert result.raw_dimensions_shape == "assessment_array"
    assert result.canonical_submission_sha256 is not None
    assert result.formal_eligible is False
    assert result.result_sha256
    provider, messages, kwargs = calls[0]
    assert provider == "gemini"
    assert packet.evaluation_id not in messages[1]["content"][1]["text"]
    assert '"output_contract"' in messages[1]["content"][1]["text"]
    data_url = messages[1]["content"][0]["image_url"]["url"]
    assert base64.b64decode(data_url.split(",", 1)[1]) == IMAGE_BYTES
    assert "thinking" not in kwargs
    assert "thinking_budget" not in kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert kwargs["max_attempts"] == 1
    assert kwargs["timeout_seconds"] == FINAL_JUDGE_TIMEOUT_SECONDS
    assert kwargs["json_mode"] is True
    assert "tools" not in kwargs
    receipt_path = write_final_judge_evaluation_result(
        tmp_path / f"final-result-{requires_card}.json",
        result,
    )
    assert b"content_base64" not in receipt_path.read_bytes()
    assert b"data:image" not in receipt_path.read_bytes()
    assert (
        load_final_judge_evaluation_result(
            receipt_path,
            expected_result_sha256=result.result_sha256,
        )
        == result
    )

    legacy_v8 = result.model_dump(mode="json")
    legacy_v8["schema_version"] = 8
    legacy_v8["cache_namespace"] = "final-evaluator-v9"
    legacy_v8["provider"] = "kimi"
    legacy_v8["model"] = "kimi-k2.6"
    legacy_v8["endpoint"] = config.PROVIDER_ENDPOINTS["kimi"]
    legacy_v8["thinking_budget"] = 6_144
    legacy_v8["max_billable_output_tokens"] = 8_192
    legacy_v8["retry_policy_version"] = FINAL_JUDGE_RETRY_POLICY_VERSION_V3
    legacy_v8["retry_policy_sha256"] = FINAL_JUDGE_RETRY_POLICY_SHA256_V3
    legacy_v8["outcome"]["judge_provider"] = "kimi"
    legacy_v8["outcome"]["judge_model"] = "kimi-k2.6"
    legacy_v8.pop("transport_policy_version", None)
    legacy_v8.pop("transport_policy_sha256", None)
    legacy_v8.pop("requested_response_format", None)
    legacy_v8.pop("forfeited_reservation_sha256", None)
    legacy_v8.pop("budget_forfeit_sha256", None)
    reparsed_v8 = FinalJudgeEvaluationResult.model_validate_json(
        canonical_json_bytes(_resign_result(legacy_v8)),
        strict=True,
    )
    assert reparsed_v8.schema_version == 8
    assert reparsed_v8.cache_namespace == "final-evaluator-v9"

    legacy_v9 = _as_historical_kimi_result(result.model_dump(mode="json"))
    legacy_v9["schema_version"] = 9
    legacy_v9["cache_namespace"] = "final-evaluator-v10"
    reparsed_v9 = FinalJudgeEvaluationResult.model_validate_json(
        canonical_json_bytes(_resign_result(legacy_v9)),
        strict=True,
    )
    assert reparsed_v9.schema_version == 9
    assert reparsed_v9.provider == "kimi"
    assert reparsed_v9.model == "kimi-k2.6"

    tampered = result.model_dump(mode="json")
    tampered["outcome"]["scores"]["j_project"] = 0.0
    with pytest.raises(ValueError):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )

    endpoint_tamper = result.model_dump(mode="json")
    endpoint_tamper["endpoint"] = config.PROVIDER_ENDPOINTS["kimi"]
    with pytest.raises(ValueError, match="endpoint differs"):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(endpoint_tamper)),
            strict=True,
        )

    for token_kind, ceiling in (
        ("input_tokens", FINAL_JUDGE_MAX_BILLABLE_INPUT_TOKENS),
        ("output_tokens", FINAL_JUDGE_MAX_BILLABLE_OUTPUT_TOKENS),
    ):
        over_limit = result.model_dump(mode="json")
        over_limit["usage"][token_kind] = ceiling + 1
        with pytest.raises(ValueError, match=f"billable {token_kind.split('_')[0]}"):
            FinalJudgeEvaluationResult.model_validate_json(
                canonical_json_bytes(_resign_result(over_limit)),
                strict=True,
            )


def test_visual_final_runner_accepts_dm_109_complete_json_fence_without_retry(
    monkeypatch,
) -> None:
    packet = _packet()
    runtime = _runtime()
    response_text = '```json\n{"CA":8,"CQ":14,"TCR":9}\n```'
    calls = 0

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )

    def fake_chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-dm-109-fenced",
            text=response_text,
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert calls == 1
    assert result.outcome.status == "scored"
    assert result.attempts == 1
    assert result.initial_empty_response is None
    assert result.raw_response_text == response_text
    assert result.raw_dimensions_shape == "bare_score_mapping"
    assert result.parser_policy_version == FINAL_JUDGE_PARSER_POLICY_VERSION_V4
    assert result.parser_policy_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V4


def test_required_empty_cards_preserve_submission_and_force_compiled_ccc_zero(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet(requires_card=True, visible_card_count=0)
    runtime = _runtime()
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-required-empty-cards",
            text=_submission(requires_card=True),
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        ),
    )

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert result.parsed_submission is not None
    raw_ccc = next(
        item.score
        for item in result.parsed_submission.dimensions
        if item.dimension == "CCC"
    )
    compiled_ccc = next(
        item.score
        for item in result.outcome.scores.dimensions
        if item.dimension == "CCC"
    )
    assert raw_ccc == 7
    assert compiled_ccc == 0
    assert result.outcome.scores.j_project == 62.0
    assert result.visible_card_count == 0
    assert result.card_requirement_guard_adjusted is True

    receipt = write_final_judge_evaluation_result(
        tmp_path / "required-empty-cards.json",
        result,
    )
    assert (
        load_final_judge_evaluation_result(
            receipt,
            expected_result_sha256=result.result_sha256,
        )
        == result
    )

    tampered = result.model_dump(mode="json")
    tampered["card_requirement_guard_adjusted"] = False
    with pytest.raises(ValueError, match="card-guard adjustment"):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )

    tampered = result.model_dump(mode="json")
    tampered["visible_card_count"] = 1
    with pytest.raises(ValueError, match="scores differ from raw response"):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )


def test_forbidden_visible_cards_preserve_submission_and_force_compiled_ca_zero(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet(requires_card=False, visible_card_count=1)
    runtime = _runtime()
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-forbidden-visible-cards",
            text=_submission(requires_card=False),
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        ),
    )

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert result.parsed_submission is not None
    raw_ca = next(
        item.score
        for item in result.parsed_submission.dimensions
        if item.dimension == "CA"
    )
    compiled_ca = next(
        item.score
        for item in result.outcome.scores.dimensions
        if item.dimension == "CA"
    )
    assert raw_ca == 8
    assert compiled_ca == 0
    assert result.outcome.scores.j_project == 57.5
    assert result.visible_card_count == 1
    assert result.card_requirement_guard_adjusted is True

    receipt = write_final_judge_evaluation_result(
        tmp_path / "forbidden-visible-cards.json",
        result,
    )
    assert (
        load_final_judge_evaluation_result(
            receipt,
            expected_result_sha256=result.result_sha256,
        )
        == result
    )

    tampered = result.model_dump(mode="json")
    tampered["visible_card_count"] = 0
    with pytest.raises(ValueError, match="scores differ from raw response"):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )


@pytest.mark.parametrize(
    ("shape", "expected_shape"),
    [
        ("score_mapping", "score_mapping"),
        ("dimension_score_pairs", "dimension_score_pairs"),
    ],
)
def test_visual_final_runner_normalizes_equivalent_score_shapes(
    monkeypatch,
    shape: str,
    expected_shape: str,
) -> None:
    packet = _packet(requires_card=True)
    runtime = _runtime()
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id=f"req-final-{shape}",
            text=_submission(requires_card=True, shape=shape),
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        ),
    )

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert result.outcome.status == "scored"
    assert result.outcome.scores.j_project == 76.0
    assert result.raw_dimensions_shape == expected_shape
    assert tuple(item.dimension for item in result.parsed_submission.dimensions) == (
        "CA",
        "CCC",
        "CQ",
        "TCR",
    )


@pytest.mark.parametrize(
    ("shape", "expected_shape"),
    [
        ("assessment_array", "bare_assessment_array"),
        ("score_mapping", "bare_score_mapping"),
        ("dimension_score_pairs", "bare_dimension_score_pairs"),
    ],
)
def test_visual_final_runner_scores_bare_equivalent_shapes_once(
    monkeypatch,
    shape: str,
    expected_shape: str,
) -> None:
    packet = _packet(requires_card=True)
    runtime = _runtime()
    calls = 0

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )

    def fake_chat(*args, **kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id=f"req-final-bare-{shape}",
            text=_bare_submission(requires_card=True, shape=shape),
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert calls == 1
    assert result.outcome.status == "scored"
    assert result.outcome.scores.j_project == 76.0
    assert result.raw_dimensions_shape == expected_shape
    assert result.canonical_submission_sha256 is not None


def test_empty_final_response_retries_once_with_identical_wire_and_scores(
    monkeypatch,
) -> None:
    packet = _packet(requires_card=False)
    runtime = _runtime()
    private_reasoning = b"private reasoning must not be persisted"
    responses = iter(
        (
            LLMResponse(
                provider="kimi",
                endpoint=config.PROVIDER_ENDPOINTS["kimi"],
                requested_model=config.PORTFOLIO_JUDGE_MODEL,
                response_model=config.PORTFOLIO_JUDGE_MODEL,
                request_id="req-empty-first",
                text="",
                usage=LLMUsage(input_tokens=20, output_tokens=65),
                finish_reason="stop",
                latency_ms=3,
                reasoning_present=True,
                reasoning_tokens=65,
                reasoning_bytes=len(private_reasoning),
                reasoning_sha256=sha256_bytes(private_reasoning),
            ),
            LLMResponse(
                provider="kimi",
                endpoint=config.PROVIDER_ENDPOINTS["kimi"],
                requested_model=config.PORTFOLIO_JUDGE_MODEL,
                response_model=config.PORTFOLIO_JUDGE_MODEL,
                request_id="req-valid-second",
                text=_submission(requires_card=False),
                usage=LLMUsage(input_tokens=30, output_tokens=12),
                finish_reason="stop",
                latency_ms=4,
            ),
        )
    )
    calls: list[tuple[bytes, dict]] = []

    def fake_chat(_provider, messages, **kwargs):
        calls.append((canonical_json_bytes(messages), kwargs))
        return next(responses)

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert result.outcome.status == "scored"
    assert result.attempts == 2
    assert result.max_attempts == 2
    assert result.captured_response_count == 2
    assert result.aggregate_usage == LLMUsage(input_tokens=50, output_tokens=77)
    assert result.initial_empty_response is not None
    assert result.initial_empty_response.request_id == "req-empty-first"
    assert result.initial_empty_response.raw_response_text == ""
    assert result.initial_empty_response.reasoning_tokens == 65
    assert result.initial_empty_response.reasoning_sha256 == sha256_bytes(
        private_reasoning
    )
    serialized = canonical_json_bytes(result.model_dump(mode="json"))
    assert private_reasoning not in serialized
    assert b"reasoning_content" not in serialized

    tampered = result.model_dump(mode="json")
    tampered["initial_empty_response"]["raw_response_bytes"] = 1
    with pytest.raises(ValueError, match="commitment mismatch"):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )


def test_reasoning_only_length_response_is_not_retried(monkeypatch) -> None:
    packet = _packet()
    runtime = _runtime()
    reasoning = b"reasoning consumed the output budget"
    calls = 0

    def fake_chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-reasoning-length",
            text="",
            usage=LLMUsage(input_tokens=21, output_tokens=2048),
            finish_reason="length",
            latency_ms=5,
            reasoning_present=True,
            reasoning_tokens=2048,
            reasoning_bytes=len(reasoning),
            reasoning_sha256=sha256_bytes(reasoning),
        )

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert calls == 1
    assert result.outcome.status == "parse_error"
    assert result.outcome.error_code == "reasoning_budget_exhausted"
    assert result.attempts == 1
    assert result.initial_empty_response is None
    assert result.captured_response_count == 1
    assert result.aggregate_usage == LLMUsage(input_tokens=21, output_tokens=2048)


def test_two_empty_final_responses_exhaust_exactly_one_retry(monkeypatch) -> None:
    packet = _packet()
    runtime = _runtime()
    calls = 0

    def fake_chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id=f"req-empty-{calls}",
            text="",
            usage=LLMUsage(input_tokens=10 + calls, output_tokens=2 + calls),
            finish_reason="stop",
            latency_ms=2,
        )

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert calls == 2
    assert result.outcome.status == "parse_error"
    assert result.outcome.error_code == "empty_final_response"
    assert result.attempts == 2
    assert result.captured_response_count == 2
    assert result.aggregate_usage == LLMUsage(input_tokens=23, output_tokens=7)

    tampered = result.model_dump(mode="json")
    tampered["outcome"]["error_code"] = "invalid_judge_json"
    with pytest.raises(ValueError, match="wrong error code"):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )


def test_empty_retry_then_provider_error_preserves_first_usage_and_stops(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet()
    runtime = _runtime()
    context = _make_budget_context(tmp_path / "second-call-forfeit")
    calls = 0

    def fake_chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise llm.LLMContractError("second request failed before a response")
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-empty-before-provider-error",
            text="",
            usage=LLMUsage(input_tokens=17, output_tokens=4),
            finish_reason="stop",
            latency_ms=2,
        )

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        budget_context=context,
        record_usage=False,
    )

    assert calls == 2
    assert result.outcome.status == "provider_error"
    assert result.attempts == 2
    assert result.request_id is None
    assert result.initial_empty_response is not None
    assert result.captured_response_count == 1
    assert result.aggregate_usage == LLMUsage(input_tokens=17, output_tokens=4)
    reservations = [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "reserve"]
    settlements = [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "settle"]
    forfeits = [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "forfeit"]
    assert len(reservations) == 2
    assert len(settlements) == 1
    assert len(forfeits) == 1
    assert forfeits[0].reservation_sha256 == reservations[1].reservation_sha256
    assert forfeits[0].reason == "provider_call_ended_without_captured_response"
    assert result.forfeited_reservation_sha256 == forfeits[0].reservation_sha256
    assert result.budget_forfeit_sha256 == forfeits[0].forfeit_sha256


@pytest.mark.parametrize(
    "response_text",
    [
        "```json\n{}\n```",
        '{"schema_version":1,"requires_card":false,"dimensions":[]}',
        '{"schema_version":1,"requires_card":false,"dimensions":['
        '{"dimension":"CA","score":11},{"dimension":"CQ","score":14},'
        '{"dimension":"TCR","score":9}]}',
    ],
)
def test_visual_final_runner_retries_then_retains_repeated_parse_errors_as_zero(
    monkeypatch,
    response_text: str,
) -> None:
    packet = _packet()
    runtime = _runtime()
    calls = 0

    def fake_chat(*args, **kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-invalid",
            text=response_text,
            usage=LLMUsage(input_tokens=20, output_tokens=8),
            finish_reason="stop",
            latency_ms=3,
        )

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(llm, "chat", fake_chat)
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert calls == 2
    assert result.outcome.status == "parse_error"
    assert result.outcome.error_code == "invalid_judge_json"
    assert result.outcome.scores.j_project == 0.0
    assert all(item.score == 0 for item in result.outcome.scores.dimensions)
    assert result.raw_response_text == response_text
    assert result.initial_empty_response is not None
    assert result.initial_empty_response.retry_reason == "invalid_judge_json"
    assert result.initial_empty_response.raw_response_text == response_text
    assert result.aggregate_usage == LLMUsage(input_tokens=40, output_tokens=16)


def test_visual_final_runner_recovers_from_one_invalid_judge_answer(
    monkeypatch,
) -> None:
    packet = _packet()
    runtime = _runtime()
    responses = iter(
        (
            "```json\n{}\n```",
            _submission(requires_card=False),
        )
    )
    calls = 0

    def fake_chat(*args, **kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id=f"req-final-contract-{calls}",
            text=next(responses),
            usage=LLMUsage(input_tokens=20, output_tokens=8),
            finish_reason="stop",
            latency_ms=3,
        )

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(llm, "chat", fake_chat)

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert calls == 2
    assert result.outcome.status == "scored"
    assert result.attempts == 2
    assert result.initial_empty_response is not None
    assert result.initial_empty_response.retry_reason == "invalid_judge_json"
    assert result.initial_empty_response.raw_response_text == "```json\n{}\n```"
    assert result.aggregate_usage == LLMUsage(input_tokens=40, output_tokens=16)

    tampered = result.model_dump(mode="json")
    valid_initial = _submission(requires_card=False)
    tampered["initial_empty_response"]["raw_response_text"] = valid_initial
    tampered["initial_empty_response"]["raw_response_sha256"] = sha256_bytes(
        valid_initial.encode("utf-8")
    )
    tampered["initial_empty_response"]["raw_response_bytes"] = len(
        valid_initial.encode("utf-8")
    )
    with pytest.raises(
        ValueError,
        match="invalid-JSON retry receipt contains a valid Judge response",
    ):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )


def test_visual_final_runner_retains_provider_contract_error(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet()
    runtime = _runtime()
    context = _make_budget_context(tmp_path / "first-call-forfeit")

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: (_ for _ in ()).throw(
            llm.LLMContractError("provider contract failed")
        ),
    )
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        budget_context=context,
        record_usage=False,
    )

    assert result.outcome.status == "provider_error"
    assert result.outcome.scores.j_project == 0.0
    assert result.request_id is None
    reservations = [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "reserve"]
    forfeits = [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "forfeit"]
    assert len(reservations) == len(forfeits) == 1
    assert not [value for kind, value in _FAKE_BUDGET_EVENTS if kind == "settle"]
    assert forfeits[0].reservation_sha256 == reservations[0].reservation_sha256
    assert forfeits[0].reason == "provider_call_ended_without_captured_response"
    assert result.forfeited_reservation_sha256 == forfeits[0].reservation_sha256
    assert result.budget_forfeit_sha256 == forfeits[0].forfeit_sha256

    tampered = result.model_dump(mode="json")
    tampered.pop("budget_forfeit_sha256")
    with pytest.raises(ValueError, match="binding must be complete"):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )


def test_visual_final_runner_retains_wrapped_timeout(monkeypatch) -> None:
    packet = _packet()
    runtime = _runtime()
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: (_ for _ in ()).throw(
            llm.LLMTimeoutError("provider request timed out")
        ),
    )

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert result.outcome.status == "timeout"
    assert result.outcome.error_code == "timeout"
    assert result.outcome.scores.j_project == 0.0
    assert result.request_id is None


def test_final_parse_error_receipt_cannot_hide_valid_raw_response(
    monkeypatch,
) -> None:
    packet = _packet()
    runtime = _runtime()

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-invalid-for-tamper",
            text="{}",
            usage=LLMUsage(input_tokens=20, output_tokens=8),
            finish_reason="stop",
            latency_ms=3,
        ),
    )
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    tampered = result.model_dump(mode="json")
    valid_response = _bare_submission(
        requires_card=False,
        shape="score_mapping",
    )
    tampered["raw_response_text"] = valid_response
    tampered["raw_response_sha256"] = sha256_bytes(valid_response.encode("utf-8"))
    tampered["raw_response_bytes"] = len(valid_response.encode("utf-8"))
    tampered["outcome"]["raw_response_sha256"] = tampered["raw_response_sha256"]
    with pytest.raises(
        ValueError,
        match="parse-error final-Judge result contains a valid response",
    ):
        FinalJudgeEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )


def test_legacy_v1_parse_error_receipt_keeps_forward_shape_invalid(
    monkeypatch,
) -> None:
    packet = _packet()
    runtime = _runtime()
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-legacy-shape",
            text="{}",
            usage=LLMUsage(input_tokens=20, output_tokens=8),
            finish_reason="stop",
            latency_ms=3,
        ),
    )
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )
    legacy = _as_historical_kimi_result(result.model_dump(mode="json"))
    raw = _submission(requires_card=False, shape="score_mapping")
    legacy["schema_version"] = 1
    legacy["cache_namespace"] = "final-evaluator-v2"
    for field in (
        "parser_policy_version",
        "parser_policy_sha256",
        "raw_dimensions_shape",
        "canonical_submission_sha256",
        "visible_card_count",
        "card_requirement_guard_policy_version",
        "card_requirement_guard_policy_sha256",
        "card_requirement_guard_adjusted",
        "retry_policy_version",
        "retry_policy_sha256",
        "initial_empty_response",
        "reasoning_present",
        "reasoning_tokens",
        "reasoning_bytes",
        "reasoning_sha256",
        "thinking_budget",
        "max_billable_input_tokens",
        "max_billable_output_tokens",
    ):
        legacy.pop(field, None)
    legacy["attempts"] = 1
    legacy["max_attempts"] = 1
    legacy["outcome"]["attempts"] = 1
    legacy["outcome"]["max_attempts"] = 1
    legacy["raw_response_text"] = raw
    legacy["raw_response_sha256"] = sha256_bytes(raw.encode("utf-8"))
    legacy["raw_response_bytes"] = len(raw.encode("utf-8"))
    legacy["outcome"]["raw_response_sha256"] = legacy["raw_response_sha256"]
    content = canonical_json_bytes(_resign_result(legacy))

    reparsed = FinalJudgeEvaluationResult.model_validate_json(
        content,
        strict=True,
    )

    assert reparsed.schema_version == 1
    assert reparsed.outcome.status == "parse_error"
    assert canonical_json_bytes(reparsed.model_dump(mode="json")) == content


def test_legacy_v2_scored_receipt_keeps_byte_and_hash_compatibility(
    monkeypatch,
) -> None:
    packet = _packet(requires_card=True)
    runtime = _runtime()
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-legacy-v2",
            text=_submission(requires_card=True, shape="score_mapping"),
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        ),
    )
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )
    legacy = _as_historical_kimi_result(result.model_dump(mode="json"))
    legacy["schema_version"] = 2
    legacy["cache_namespace"] = "final-evaluator-v3"
    legacy["parser_policy_version"] = FINAL_JUDGE_PARSER_POLICY_VERSION_V2
    legacy["parser_policy_sha256"] = FINAL_JUDGE_PARSER_POLICY_SHA256_V2
    for field in (
        "visible_card_count",
        "card_requirement_guard_policy_version",
        "card_requirement_guard_policy_sha256",
        "card_requirement_guard_adjusted",
        "retry_policy_version",
        "retry_policy_sha256",
        "initial_empty_response",
        "reasoning_present",
        "reasoning_tokens",
        "reasoning_bytes",
        "reasoning_sha256",
        "thinking_budget",
        "max_billable_input_tokens",
        "max_billable_output_tokens",
    ):
        legacy.pop(field, None)
    legacy["attempts"] = 1
    legacy["max_attempts"] = 1
    legacy["outcome"]["attempts"] = 1
    legacy["outcome"]["max_attempts"] = 1
    content = canonical_json_bytes(_resign_result(legacy))
    expected_result_sha256 = legacy["result_sha256"]

    reparsed = FinalJudgeEvaluationResult.model_validate_json(
        content,
        strict=True,
    )

    assert reparsed.schema_version == 2
    assert reparsed.cache_namespace == "final-evaluator-v3"
    assert reparsed.result_sha256 == expected_result_sha256
    assert canonical_json_bytes(reparsed.model_dump(mode="json")) == content


@pytest.mark.parametrize(
    ("schema_version", "cache_namespace"),
    [(3, "final-evaluator-v4"), (4, "final-evaluator-v5")],
)
def test_legacy_v3_v4_receipts_keep_canonical_byte_compatibility(
    monkeypatch,
    schema_version: int,
    cache_namespace: str,
) -> None:
    packet = _packet(requires_card=True)
    runtime = _runtime()
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id=f"req-final-legacy-v{schema_version}",
            text=_submission(requires_card=True),
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        ),
    )
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )
    legacy = _as_historical_kimi_result(result.model_dump(mode="json"))
    legacy["schema_version"] = schema_version
    legacy["cache_namespace"] = cache_namespace
    legacy["parser_policy_version"] = FINAL_JUDGE_PARSER_POLICY_VERSION_V3
    legacy["parser_policy_sha256"] = FINAL_JUDGE_PARSER_POLICY_SHA256_V3
    for field in (
        "retry_policy_version",
        "retry_policy_sha256",
        "initial_empty_response",
        "reasoning_present",
        "reasoning_tokens",
        "reasoning_bytes",
        "reasoning_sha256",
        "thinking_budget",
        "max_billable_input_tokens",
        "max_billable_output_tokens",
    ):
        legacy.pop(field, None)
    if schema_version == 3:
        for field in (
            "visible_card_count",
            "card_requirement_guard_policy_version",
            "card_requirement_guard_policy_sha256",
            "card_requirement_guard_adjusted",
        ):
            legacy.pop(field, None)
    legacy["attempts"] = 1
    legacy["max_attempts"] = 1
    legacy["outcome"]["attempts"] = 1
    legacy["outcome"]["max_attempts"] = 1
    content = canonical_json_bytes(_resign_result(legacy))

    reparsed = FinalJudgeEvaluationResult.model_validate_json(
        content,
        strict=True,
    )

    assert reparsed.schema_version == schema_version
    assert reparsed.cache_namespace == cache_namespace
    assert canonical_json_bytes(reparsed.model_dump(mode="json")) == content


def test_legacy_v5_receipt_keeps_canonical_byte_compatibility(monkeypatch) -> None:
    packet = _packet(requires_card=True)
    runtime = _runtime()
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-legacy-v5",
            text=_submission(requires_card=True),
            usage=LLMUsage(input_tokens=30, output_tokens=12),
            finish_reason="stop",
            latency_ms=5,
        ),
    )
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )
    legacy = _as_historical_kimi_result(result.model_dump(mode="json"))
    legacy["schema_version"] = 5
    legacy["cache_namespace"] = "final-evaluator-v6"
    legacy["parser_policy_version"] = FINAL_JUDGE_PARSER_POLICY_VERSION_V3
    legacy["parser_policy_sha256"] = FINAL_JUDGE_PARSER_POLICY_SHA256_V3
    legacy["retry_policy_version"] = (
        final_runtime_module._FINAL_JUDGE_RETRY_POLICY_VERSION_V1
    )
    legacy["retry_policy_sha256"] = (
        final_runtime_module._FINAL_JUDGE_RETRY_POLICY_SHA256_V1
    )
    for field in (
        "thinking_budget",
        "max_billable_input_tokens",
        "max_billable_output_tokens",
    ):
        legacy.pop(field)
    content = canonical_json_bytes(_resign_result(legacy))

    reparsed = FinalJudgeEvaluationResult.model_validate_json(
        content,
        strict=True,
    )

    assert reparsed.schema_version == 5
    assert reparsed.cache_namespace == "final-evaluator-v6"
    assert canonical_json_bytes(reparsed.model_dump(mode="json")) == content


@pytest.mark.parametrize(
    ("schema_version", "cache_namespace", "retry_version", "retry_sha256"),
    [
        (
            5,
            "final-evaluator-v6",
            final_runtime_module._FINAL_JUDGE_RETRY_POLICY_VERSION_V1,
            final_runtime_module._FINAL_JUDGE_RETRY_POLICY_SHA256_V1,
        ),
        (
            6,
            "final-evaluator-v7",
            final_runtime_module.FINAL_JUDGE_RETRY_POLICY_VERSION_V2,
            final_runtime_module.FINAL_JUDGE_RETRY_POLICY_SHA256_V2,
        ),
        (
            7,
            "final-evaluator-v8",
            FINAL_JUDGE_RETRY_POLICY_VERSION_V3,
            FINAL_JUDGE_RETRY_POLICY_SHA256_V3,
        ),
    ],
)
def test_legacy_empty_retry_receipts_keep_canonical_byte_compatibility(
    monkeypatch,
    schema_version: int,
    cache_namespace: str,
    retry_version: str,
    retry_sha256: str,
) -> None:
    packet = _packet(requires_card=True)
    runtime = _runtime()
    responses = iter(
        (
            LLMResponse(
                provider="kimi",
                endpoint=config.PROVIDER_ENDPOINTS["kimi"],
                requested_model=config.PORTFOLIO_JUDGE_MODEL,
                response_model=config.PORTFOLIO_JUDGE_MODEL,
                request_id=f"req-final-legacy-v{schema_version}-empty",
                text="",
                usage=LLMUsage(input_tokens=30, output_tokens=8),
                finish_reason="stop",
                latency_ms=5,
            ),
            LLMResponse(
                provider="kimi",
                endpoint=config.PROVIDER_ENDPOINTS["kimi"],
                requested_model=config.PORTFOLIO_JUDGE_MODEL,
                response_model=config.PORTFOLIO_JUDGE_MODEL,
                request_id=f"req-final-legacy-v{schema_version}-scored",
                text=_submission(requires_card=True),
                usage=LLMUsage(input_tokens=30, output_tokens=12),
                finish_reason="stop",
                latency_ms=5,
            ),
        )
    )
    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(llm, "chat", lambda *_, **__: next(responses))

    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )
    legacy = _as_historical_kimi_result(result.model_dump(mode="json"))
    assert "retry_reason" not in legacy["initial_empty_response"]
    legacy["schema_version"] = schema_version
    legacy["cache_namespace"] = cache_namespace
    legacy["parser_policy_version"] = FINAL_JUDGE_PARSER_POLICY_VERSION_V3
    legacy["parser_policy_sha256"] = FINAL_JUDGE_PARSER_POLICY_SHA256_V3
    legacy["retry_policy_version"] = retry_version
    legacy["retry_policy_sha256"] = retry_sha256
    if schema_version == 5:
        for field in (
            "thinking_budget",
            "max_billable_input_tokens",
            "max_billable_output_tokens",
        ):
            legacy.pop(field)
    content = canonical_json_bytes(_resign_result(legacy))

    reparsed = FinalJudgeEvaluationResult.model_validate_json(
        content,
        strict=True,
    )

    assert reparsed.schema_version == schema_version
    assert reparsed.initial_empty_response is not None
    assert reparsed.initial_empty_response.retry_reason == "empty_final_response"
    assert canonical_json_bytes(reparsed.model_dump(mode="json")) == content


def test_v2_parser_policy_hash_is_frozen() -> None:
    assert FINAL_JUDGE_PARSER_POLICY_SHA256_V2 == (
        "d151456396bd1bef560de116dc3eb83e94cfcb3fe686ac3ec79fb429e89d2509"
    )


def test_card_requirement_guard_policy_hash_is_frozen() -> None:
    assert CARD_REQUIREMENT_GUARD_POLICY_SHA256 == (
        "18c2567c4948d9a5b7f5db2c3fee635d5c0302abbb1e07b62e7231e3a6bf4e04"
    )


def test_judge_retry_policy_hashes_are_frozen() -> None:
    assert FINAL_JUDGE_RETRY_POLICY_SHA256_V3 == (
        "71e4941e3d6bece1e8562d5ed1c367d05cd981b2b06cf8f04e313f881f900af5"
    )
    assert FINAL_JUDGE_RETRY_POLICY_SHA256 == (
        "c0abe7790bcaaada1db25a118b46f6d7a3e47f246453fbb954361b44c85ab203"
    )
    assert final_runtime_module.FINAL_JUDGE_RETRY_POLICY_SHA256_V2 == (
        "f592d37b5f7c11f6d88fdc5b0b31eb69b6a519f2ce39db3f6f635ac52ae80cc1"
    )
    assert FINAL_JUDGE_TRANSPORT_POLICY_SHA256 == (
        "72c8e66b3ffb4aa149af8706b5fb9709953a24bb3881599f332c1b86949e4005"
    )


def test_visual_final_runner_redacts_input_image_echo(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet()
    runtime = _runtime()
    data_url = "data:image/png;base64," + base64.b64encode(IMAGE_BYTES).decode("ascii")

    monkeypatch.setattr(
        "skillchain.evaluation.final_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, IMAGE_BYTES),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="kimi",
            endpoint=config.PROVIDER_ENDPOINTS["kimi"],
            requested_model=config.PORTFOLIO_JUDGE_MODEL,
            response_model=config.PORTFOLIO_JUDGE_MODEL,
            request_id="req-final-image-echo",
            text=data_url,
            usage=LLMUsage(input_tokens=20, output_tokens=20),
            finish_reason="stop",
            latency_ms=3,
        ),
    )
    result = run_visual_final_judge(
        packet,
        make_active_portfolio_evaluator_isolation_lock(),
        remote_runtime=runtime,
        record_usage=False,
    )

    assert result.outcome.status == "parse_error"
    assert result.outcome.error_code == "input_image_echo"
    assert result.response_redaction_reason == "input_image_echo"
    assert result.raw_response_text is None
    assert result.raw_response_sha256 == sha256_bytes(data_url.encode("utf-8"))
    receipt = write_final_judge_evaluation_result(
        tmp_path / "redacted-final.json",
        result,
    ).read_bytes()
    assert data_url.encode("utf-8") not in receipt
    assert base64.b64encode(IMAGE_BYTES) not in receipt
