"""Measure safe in-flight concurrency for the frozen Qwen3.7 Feedback wire.

This is an explicit paid integration probe.  It uses the production model,
thinking controls, strict JSON Schema, parser, and representative local images,
but it does not consume or mutate an evaluation authorization.  Only response
hashes and request metrics are persisted; model text and credentials are not.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from threading import Lock
import time
from typing import Any

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, RateLimitError

from skillchain import config, llm
from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    parse_visual_feedback_output_v3,
)
from skillchain.evaluation.feedback_runtime import visual_feedback_response_format_v1
from skillchain.evaluation.evaluator_outputs import feedback_output_contract_v4
from skillchain.evaluation.packets import (
    _FEEDBACK_SYSTEM_PROMPT_V5,
    visual_feedback_prompt_output_identity_v5,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS,
    QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS,
)
from skillchain.synthesis.store import atomic_create_file, canonical_json_bytes


DEFAULT_LEVELS = (2, 4, 8, 16, 32, 64)
DEFAULT_ERROR_RATE_LIMIT = config.LEGACY_QWEN37_FEEDBACK_ACCEPTABLE_ERROR_RATE
DEFAULT_SERVICE_ERROR_RATE_LIMIT = config.LEGACY_QWEN37_FEEDBACK_SERVICE_ERROR_RATE
DEFAULT_COST_CAP_CNY = Decimal("9.500000")
OFFICIAL_SNAPSHOT_RPM = 600
OFFICIAL_SNAPSHOT_TPM = 1_000_000


@dataclass(frozen=True)
class ProbeResult:
    ordinal: int
    level: int
    ok: bool
    error_kind: str | None
    status_code: int | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    reasoning_present: bool | None
    response_sha256: str | None
    request_id_sha256: str | None
    cost_cny: str

    def payload(self) -> dict[str, object]:
        return self.__dict__.copy()


class _StartPacer:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1.0 / requests_per_second
        self._lock = Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self._interval
        if delay:
            time.sleep(delay)


class _ConcurrencyTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._active = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)

    def exit(self) -> None:
        with self._lock:
            self._active -= 1


def _load_dashscope_credential(env_file: Path) -> str:
    values = dotenv_values(env_file)
    source_name = "DASHSCOPE_API_KEY"
    value = values.get(source_name)
    if not value:
        # Historical local environments used KIMI_API_KEY for the same
        # DashScope account before the repository standardized the name.
        source_name = "KIMI_API_KEY"
        value = values.get(source_name)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"{env_file} does not configure DASHSCOPE_API_KEY or KIMI_API_KEY"
        )
    os.environ["DASHSCOPE_API_KEY"] = value.strip()
    return source_name


def _prompt(ordinal: int) -> list[dict[str, str]]:
    payload = {
        "schema_version": 2,
        "packet_kind": "feedback",
        "cache_namespace": "feedback-evaluator-v9",
        "query_id": f"capacity-probe-{ordinal:04d}",
        "turns": [
            {
                "role": "user",
                "content": "Identify the product and answer only from visible evidence.",
            }
        ],
        "canonical_capability": "product.exact_match",
        "acceptable_capabilities": ["product.exact_match"],
        "response_text": (
            "This image shows a blue household product, so I recommend buying it."
        ),
        "cards": [],
        "tool_evidence": [],
        "tool_trace": [],
        "rubric": (
            "Check task completion, image grounding, required evidence, response "
            "clarity, routing, and tool use."
        ),
        "output_contract": feedback_output_contract_v4(),
        "response_identity": visual_feedback_prompt_output_identity_v5(),
    }
    return [
        {
            "role": "system",
            "content": _FEEDBACK_SYSTEM_PROMPT_V5,
        },
        {
            "role": "user",
            "content": canonical_json_bytes(payload).decode("utf-8").removesuffix("\n"),
        },
    ]


def _cost(input_tokens: int, output_tokens: int) -> Decimal:
    return (
        Decimal(input_tokens)
        * Decimal(QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS)
        + Decimal(output_tokens)
        * Decimal(QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS)
    ) / Decimal(1_000_000)


def _error_kind(error: BaseException) -> tuple[str, int | None]:
    if isinstance(error, RateLimitError):
        return "rate_limit", 429
    if isinstance(error, llm.LLMTimeoutError):
        return "timeout", None
    if isinstance(error, APIConnectionError):
        return "connection", None
    if isinstance(error, APIStatusError):
        return "api_status", error.status_code
    if isinstance(error, EvaluatorOutputParseError):
        return "parse", None
    if isinstance(error, llm.LLMContractError):
        return "contract", None
    return type(error).__name__, getattr(error, "status_code", None)


def _one_call(
    *,
    ordinal: int,
    level: int,
    image: Path,
    pacer: _StartPacer,
    tracker: _ConcurrencyTracker,
) -> ProbeResult:
    pacer.wait()
    tracker.enter()
    try:
        return _one_call_active(ordinal=ordinal, level=level, image=image)
    finally:
        tracker.exit()


def _one_call_active(*, ordinal: int, level: int, image: Path) -> ProbeResult:
    started = time.perf_counter()
    try:
        response = llm.chat(
            "qwen",
            _prompt(ordinal),
            model=config.LEGACY_QWEN37_FEEDBACK_JUDGE_MODEL,
            images=[str(image)],
            thinking=True,
            thinking_budget=2048,
            max_tokens=None,
            max_completion_tokens=4096,
            response_format=visual_feedback_response_format_v1(),
            max_attempts=1,
            record_usage=False,
            timeout_seconds=600,
        )
    except Exception as error:  # retain every worker outcome in the receipt
        error_kind, status_code = _error_kind(error)
        return ProbeResult(
            ordinal=ordinal,
            level=level,
            ok=False,
            error_kind=error_kind,
            status_code=status_code,
            latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
            input_tokens=0,
            output_tokens=0,
            finish_reason=None,
            reasoning_present=None,
            response_sha256=None,
            request_id_sha256=None,
            cost_cny="0",
        )
    try:
        if response.finish_reason != "stop" or response.tool_calls:
            raise llm.LLMContractError(
                "capacity probe response did not finish with plain stop text"
            )
        parse_visual_feedback_output_v3(response.text)
    except Exception as error:
        error_kind, status_code = _error_kind(error)
        return ProbeResult(
            ordinal=ordinal,
            level=level,
            ok=False,
            error_kind=error_kind,
            status_code=status_code,
            latency_ms=response.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.finish_reason,
            reasoning_present=response.reasoning_present,
            response_sha256=hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
            request_id_sha256=hashlib.sha256(
                response.request_id.encode("utf-8")
            ).hexdigest(),
            cost_cny=format(
                _cost(response.usage.input_tokens, response.usage.output_tokens), "f"
            ),
        )
    return ProbeResult(
        ordinal=ordinal,
        level=level,
        ok=True,
        error_kind=None,
        status_code=None,
        latency_ms=response.latency_ms,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        finish_reason=response.finish_reason,
        reasoning_present=response.reasoning_present,
        response_sha256=hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
        request_id_sha256=hashlib.sha256(response.request_id.encode("utf-8")).hexdigest(),
        cost_cny=format(
            _cost(response.usage.input_tokens, response.usage.output_tokens), "f"
        ),
    )
def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _summarize(level: int, results: list[ProbeResult]) -> dict[str, object]:
    successes = [item for item in results if item.ok]
    latencies = [item.latency_ms for item in successes]
    errors: dict[str, int] = {}
    for item in results:
        if item.error_kind:
            errors[item.error_kind] = errors.get(item.error_kind, 0) + 1
    total = len(results)
    failures = total - len(successes)
    service_failures = sum(
        item.error_kind in {"rate_limit", "timeout", "connection", "api_status"}
        for item in results
    )
    return {
        "level": level,
        "calls": total,
        "successes": len(successes),
        "failures": failures,
        "observed_error_rate": failures / total,
        "service_failures": service_failures,
        "observed_service_error_rate": service_failures / total,
        "errors": errors,
        "latency_ms": {
            "median": round(median(latencies)) if latencies else 0,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0),
        },
        "input_tokens": sum(item.input_tokens for item in results),
        "output_tokens": sum(item.output_tokens for item in results),
        "cost_cny": format(sum(Decimal(item.cost_cny) for item in results), "f"),
    }


def _summary_passes(
    summary: dict[str, object],
    *,
    error_rate_limit: float,
    service_error_rate_limit: float,
) -> bool:
    return (
        float(summary["observed_error_rate"]) <= error_rate_limit
        and float(summary["observed_service_error_rate"])
        <= service_error_rate_limit
    )


def _highest_passing_level(
    results: list[ProbeResult],
    *,
    error_rate_limit: float,
    service_error_rate_limit: float,
) -> int:
    passing = []
    for level in sorted({item.level for item in results}):
        summary = _summarize(level, [item for item in results if item.level == level])
        if _summary_passes(
            summary,
            error_rate_limit=error_rate_limit,
            service_error_rate_limit=service_error_rate_limit,
        ):
            passing.append(level)
    return max(passing, default=0)


def _run_level(
    *,
    level: int,
    call_count: int,
    first_ordinal: int,
    images: tuple[Path, ...],
    requests_per_second: float,
) -> tuple[list[ProbeResult], int]:
    pacer = _StartPacer(requests_per_second)
    tracker = _ConcurrencyTracker()
    results: list[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=level) as executor:
        futures = [
            executor.submit(
                _one_call,
                ordinal=first_ordinal + offset,
                level=level,
                image=images[offset % len(images)],
                pacer=pacer,
                tracker=tracker,
            )
            for offset in range(call_count)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.ordinal), tracker.peak


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(payload)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite capacity receipt: {path}")
    atomic_create_file(path, content)


def _parse_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from error
    if not levels or any(item < 1 for item in levels) or tuple(sorted(set(levels))) != levels:
        raise argparse.ArgumentTypeError("levels must be positive, unique, and increasing")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=_parse_levels, default=DEFAULT_LEVELS)
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=config.LEGACY_QWEN37_FEEDBACK_REQUESTS_PER_SECOND,
    )
    parser.add_argument("--error-rate-limit", type=float, default=DEFAULT_ERROR_RATE_LIMIT)
    parser.add_argument(
        "--service-error-rate-limit",
        type=float,
        default=DEFAULT_SERVICE_ERROR_RATE_LIMIT,
    )
    parser.add_argument("--cost-cap-cny", type=Decimal, default=DEFAULT_COST_CAP_CNY)
    parser.add_argument(
        "--confirmation-calls",
        type=int,
        default=64,
        help="extra calls at the highest passing level",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.error_rate_limit < 1:
        raise ValueError("error-rate-limit must be in [0, 1)")
    if not 0 <= args.service_error_rate_limit < 1:
        raise ValueError("service-error-rate-limit must be in [0, 1)")
    if args.requests_per_second <= 0:
        raise ValueError("requests-per-second must be positive")
    if args.confirmation_calls < 0:
        raise ValueError("confirmation-calls must be non-negative")
    images = tuple(path.resolve(strict=True) for path in args.image)
    credential_source = _load_dashscope_credential(args.env_file.resolve(strict=True))

    started_at = datetime.now(timezone.utc)
    all_results: list[ProbeResult] = []
    summaries: list[dict[str, object]] = []
    highest_passing = 0
    stop_reason = "levels_exhausted"
    next_ordinal = 1

    for level in args.levels:
        average_cost = (
            sum(Decimal(item.cost_cny) for item in all_results) / len(all_results)
            if all_results
            else Decimal("0.020")
        )
        current_cost = sum(Decimal(item.cost_cny) for item in all_results)
        if current_cost + average_cost * level > args.cost_cap_cny:
            stop_reason = "projected_cost_cap"
            break
        results, peak_inflight = _run_level(
            level=level,
            call_count=level,
            first_ordinal=next_ordinal,
            images=images,
            requests_per_second=args.requests_per_second,
        )
        next_ordinal += len(results)
        all_results.extend(results)
        summary = _summarize(level, results)
        summary["peak_inflight_observed"] = peak_inflight
        summaries.append(summary)
        if not _summary_passes(
            summary,
            error_rate_limit=args.error_rate_limit,
            service_error_rate_limit=args.service_error_rate_limit,
        ):
            stop_reason = "error_rate_limit"
            break
        highest_passing = level

    if highest_passing and args.confirmation_calls:
        current_cost = sum(Decimal(item.cost_cny) for item in all_results)
        average_cost = current_cost / len(all_results)
        affordable = max(
            0,
            int((args.cost_cap_cny - current_cost) / max(average_cost, Decimal("0.000001"))),
        )
        confirmation_calls = min(args.confirmation_calls, affordable)
        if confirmation_calls:
            confirmation, peak_inflight = _run_level(
                level=highest_passing,
                call_count=confirmation_calls,
                first_ordinal=next_ordinal,
                images=images,
                requests_per_second=args.requests_per_second,
            )
            all_results.extend(confirmation)
            summary = _summarize(highest_passing, confirmation)
            summary["confirmation"] = True
            summary["peak_inflight_observed"] = peak_inflight
            summaries.append(summary)
            combined = [item for item in all_results if item.level == highest_passing]
            if not _summary_passes(
                _summarize(highest_passing, combined),
                error_rate_limit=args.error_rate_limit,
                service_error_rate_limit=args.service_error_rate_limit,
            ):
                stop_reason = "confirmation_error_rate_limit"
                highest_passing = _highest_passing_level(
                    all_results,
                    error_rate_limit=args.error_rate_limit,
                    service_error_rate_limit=args.service_error_rate_limit,
                )

    completed_at = datetime.now(timezone.utc)
    total_cost = sum(Decimal(item.cost_cny) for item in all_results)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen37-feedback-concurrency-probe",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "provider": "qwen",
        "model": config.LEGACY_QWEN37_FEEDBACK_JUDGE_MODEL,
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "credential_source_name": credential_source,
        "credential_value_persisted": False,
        "production_wire": {
            "enable_thinking": True,
            "thinking_budget": 2048,
            "max_completion_tokens": 4096,
            "response_format": "strict_json_schema",
            "max_attempts": 1,
        },
        "official_snapshot_limits": {
            "requests_per_minute": OFFICIAL_SNAPSHOT_RPM,
            "tokens_per_minute": OFFICIAL_SNAPSHOT_TPM,
        },
        "probe_policy": {
            "levels": list(args.levels),
            "requests_per_second": args.requests_per_second,
            "observed_error_rate_limit": args.error_rate_limit,
            "observed_service_error_rate_limit": args.service_error_rate_limit,
            "cost_cap_cny": format(args.cost_cap_cny, "f"),
            "confirmation_calls_requested": args.confirmation_calls,
        },
        "image_sha256s": [hashlib.sha256(path.read_bytes()).hexdigest() for path in images],
        "level_summaries": summaries,
        "highest_passing_concurrency": highest_passing,
        "stop_reason": stop_reason,
        "calls": len(all_results),
        "successes": sum(item.ok for item in all_results),
        "failures": sum(not item.ok for item in all_results),
        "total_cost_cny": format(total_cost, "f"),
        "results": [item.payload() for item in all_results],
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_receipt(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in (
        "highest_passing_concurrency",
        "stop_reason",
        "calls",
        "successes",
        "failures",
        "total_cost_cny",
        "level_summaries",
        "receipt_sha256",
    )}, ensure_ascii=False, indent=2))
    return 0 if highest_passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
