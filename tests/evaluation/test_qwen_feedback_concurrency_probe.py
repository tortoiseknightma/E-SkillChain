from decimal import Decimal
import json

from scripts.probe_qwen_feedback_concurrency import (
    PROBE_PROFILES,
    ProbeResult,
    _cost,
    _highest_passing_level,
    _prompt,
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
    assert config.LEGACY_QWEN37_FEEDBACK_MAX_CONCURRENCY == 240
    assert config.LEGACY_QWEN37_FEEDBACK_REQUESTS_PER_SECOND == 8.0
    assert config.LEGACY_QWEN37_FEEDBACK_ACCEPTABLE_ERROR_RATE == 0.02
    assert config.LEGACY_QWEN37_FEEDBACK_SERVICE_ERROR_RATE == 0.0


def test_active_capacity_constants_bind_the_measured_60_call_profile() -> None:
    assert config.FEEDBACK_JUDGE_VALIDATED_CONCURRENCY == 60
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
    assert _cost(3_857, 2_000, PROBE_PROFILES["legacy-qwen37"]) == Decimal(
        "0.023714"
    )


def test_active_qwen38_profile_uses_production_wire_and_rates() -> None:
    profile = PROBE_PROFILES["active-qwen38"]

    assert profile.model == config.FEEDBACK_JUDGE_MODEL == "qwen3.8-max"
    assert profile.max_completion_tokens == 6_144
    assert profile.official_snapshot_rpm == 30_000
    assert profile.official_snapshot_tpm == 5_000_000
    assert _cost(3_857, 2_000, profile) == Decimal("0.118284")


def test_active_qwen38_prompt_requires_policy_labeled_suggestions() -> None:
    messages = _prompt(1, PROBE_PROFILES["active-qwen38"])
    payload = json.loads(messages[1]["content"])

    assert payload["schema_version"] == 3
    assert payload["cache_namespace"] == "feedback-evaluator-v10"
    assert payload["gcs_diagnostics"]["gcs"] == 0
    assert payload["output_contract"]["skill_suggestions_item_schema"][
        "required_prefix_exactly_one_of"
    ] == [
        "[policy_compatible] ",
        "[requires_new_evidence] ",
        "[rejected] ",
    ]
