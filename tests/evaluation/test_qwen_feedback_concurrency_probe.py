from decimal import Decimal

from scripts.probe_qwen_feedback_concurrency import (
    ProbeResult,
    _cost,
    _highest_passing_level,
    _summarize,
)
from skillchain import config


def _result(*, ordinal: int, level: int, error: str | None = None) -> ProbeResult:
    return ProbeResult(
        ordinal=ordinal,
        level=level,
        ok=error is None,
        error_kind=error,
        status_code=429 if error == "rate_limit" else None,
        latency_ms=1_000,
        input_tokens=100 if error not in {"rate_limit", "timeout"} else 0,
        output_tokens=50 if error not in {"rate_limit", "timeout"} else 0,
        finish_reason="stop" if error not in {"rate_limit", "timeout"} else None,
        reasoning_present=True if error not in {"rate_limit", "timeout"} else None,
        response_sha256="a" * 64 if error not in {"rate_limit", "timeout"} else None,
        request_id_sha256="b" * 64 if error not in {"rate_limit", "timeout"} else None,
        cost_cny="0.000600" if error not in {"rate_limit", "timeout"} else "0",
    )


def test_forward_capacity_constants_bind_the_measured_240_call_profile() -> None:
    assert config.FEEDBACK_JUDGE_MAX_CONCURRENCY == 240
    assert config.FEEDBACK_JUDGE_REQUESTS_PER_SECOND == 8.0
    assert config.FEEDBACK_JUDGE_ACCEPTABLE_ERROR_RATE == 0.02
    assert config.FEEDBACK_JUDGE_SERVICE_ERROR_RATE == 0.0


def test_summary_separates_payload_and_service_failures() -> None:
    summary = _summarize(
        240,
        [
            _result(ordinal=1, level=240),
            _result(ordinal=2, level=240, error="parse"),
            _result(ordinal=3, level=240, error="rate_limit"),
        ],
    )

    assert summary["failures"] == 2
    assert summary["service_failures"] == 1
    assert summary["observed_error_rate"] == 2 / 3
    assert summary["observed_service_error_rate"] == 1 / 3


def test_confirmation_recomputes_the_highest_combined_passing_level() -> None:
    results = [
        *[_result(ordinal=index, level=8) for index in range(1, 9)],
        *[_result(ordinal=index, level=16) for index in range(9, 88)],
        _result(ordinal=88, level=16, error="parse"),
    ]

    assert (
        _highest_passing_level(
            results,
            error_rate_limit=0.01,
            service_error_rate_limit=0.0,
        )
        == 8
    )


def test_cost_uses_frozen_qwen37_rates() -> None:
    assert _cost(3_857, 2_000) == Decimal("0.023714")
