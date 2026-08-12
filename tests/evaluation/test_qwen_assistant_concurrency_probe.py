from decimal import Decimal

import pytest

from skillchain import config
from skillchain.evaluation.core_fast.live_adapter import _assistant_cost
from skillchain.evaluation.core_fast.pacing import StartPacer

from scripts.probe_qwen_assistant_concurrency import (
    DEFAULT_REQUESTS_PER_SECOND,
    _cost,
    _messages,
    _summarize,
    _variant,
    ProbeResult,
)


def _result(*, ordinal: int, error: str | None = None) -> ProbeResult:
    return ProbeResult(
        ordinal=ordinal,
        level=60,
        variant=_variant(ordinal),
        ok=error is None,
        error_kind=error,
        status_code=429 if error == "rate_limit" else None,
        latency_ms=1_000,
        input_tokens=1_400 if error != "rate_limit" else 0,
        output_tokens=30 if error != "rate_limit" else 0,
        finish_reason="stop" if ordinal % 2 == 0 else "tool_calls",
        response_sha256="a" * 64,
        request_id_sha256="b" * 64,
        cost_cny="0.000304" if error != "rate_limit" else "0",
    )


def test_probe_alternates_production_action_variants() -> None:
    assert [_variant(index) for index in range(1, 5)] == [
        "tool_selection",
        "final_response",
        "tool_selection",
        "final_response",
    ]
    assert len(_messages(1, "tool_selection")) == 2
    assert [item["role"] for item in _messages(2, "final_response")] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


def test_cost_uses_qwen37_flash_under_32k_tier() -> None:
    assert DEFAULT_REQUESTS_PER_SECOND == 20.0
    assert _cost(1_400, 30) == Decimal("0.000304")
    assert _assistant_cost(1_400, 30) == 0.000304


def test_assistant_start_pacer_smooths_provider_calls() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    pacer = StartPacer(20.0, clock=lambda: now[0], sleeper=sleep)
    for _ in range(4):
        pacer.wait()

    assert sleeps == pytest.approx([0.05, 0.05, 0.05])


def test_active_assistant_capacity_constants_bind_the_measured_profile() -> None:
    assert config.ASSISTANT_VALIDATED_CONCURRENCY == 60
    assert config.ASSISTANT_REQUESTS_PER_SECOND == 20.0
    assert config.ASSISTANT_ACCEPTABLE_ERROR_RATE == 0.02
    assert config.ASSISTANT_SERVICE_ERROR_RATE == 0.0


def test_summary_separates_contract_and_service_failures() -> None:
    summary = _summarize(
        [
            _result(ordinal=1),
            _result(ordinal=2, error="contract"),
            _result(ordinal=3, error="rate_limit"),
        ],
        peak_inflight=3,
    )

    assert summary["failures"] == 2
    assert summary["service_failures"] == 1
    assert summary["variant_counts"]["tool_selection"] == {
        "calls": 2,
        "successes": 1,
    }
    assert summary["peak_inflight_observed"] == 3
