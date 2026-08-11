import json

from scripts.probe_gemini_judge_concurrency import (
    AttemptResult,
    TaskResult,
    _prompt,
    _summarize,
)
from skillchain import config
from skillchain.evaluation.final_runtime import FINAL_JUDGE_ANSWER_MAX_TOKENS


def _attempt(*, attempt: int, ok: bool, error: str | None = None) -> AttemptResult:
    return AttemptResult(
        attempt=attempt,
        ok=ok,
        error_kind=error,
        status_code=429 if error == "rate_limit" else None,
        latency_ms=1_000,
        input_tokens=500,
        output_tokens=50,
        finish_reason="stop",
        reasoning_present=False,
        response_sha256="a" * 64,
        request_id_sha256="b" * 64,
    )


def test_active_gemini_profile_matches_final_judge_wire() -> None:
    assert config.PORTFOLIO_JUDGE_PROVIDER == "gemini"
    assert config.PORTFOLIO_JUDGE_MODEL == "gemini-3.6-flash"
    assert FINAL_JUDGE_ANSWER_MAX_TOKENS == 2_048


def test_prompt_embeds_exact_no_card_output_contract() -> None:
    messages = _prompt("frozen rubric")
    payload = json.loads(messages[1]["content"])

    assert payload["card_requirement"] == "forbidden"
    assert payload["output_contract"]["requires_card"] is False
    assert payload["output_contract"]["dimension_order"] == ["CA", "CQ", "TCR"]


def test_summary_separates_recovered_retry_and_service_failure() -> None:
    results = [
        TaskResult(
            ordinal=1,
            ok=True,
            error_kind=None,
            latency_ms=1_000,
            attempts=(_attempt(attempt=1, ok=True),),
        ),
        TaskResult(
            ordinal=2,
            ok=True,
            error_kind=None,
            latency_ms=2_000,
            attempts=(
                _attempt(attempt=1, ok=False, error="parse"),
                _attempt(attempt=2, ok=True),
            ),
        ),
        TaskResult(
            ordinal=3,
            ok=False,
            error_kind="rate_limit",
            latency_ms=1_000,
            attempts=(_attempt(attempt=1, ok=False, error="rate_limit"),),
        ),
    ]

    summary = _summarize(results)

    assert summary["successes"] == 2
    assert summary["failures"] == 1
    assert summary["provider_attempts"] == 4
    assert summary["retry_count"] == 1
    assert summary["first_attempt_failures"] == 2
    assert summary["service_failures"] == 1
