"""Measure task concurrency for the active AIFast Gemini final-Judge wire.

This paid integration probe sends representative local images through the
production Gemini model, JSON-object response mode, output parser, and bounded
retry rule.  It does not consume an experiment authorization.  Response text,
request IDs, image bytes, and credentials are never persisted; only hashes and
request metrics are written to the receipt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
from statistics import median
from threading import Event, Lock
import time
from typing import Any

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, RateLimitError

from skillchain import config, llm
from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    final_output_contract,
    parse_final_judge_output_v4,
)
from skillchain.evaluation.final_runtime import (
    FINAL_JUDGE_ANSWER_MAX_TOKENS,
    FINAL_JUDGE_MAX_ATTEMPTS,
    FINAL_JUDGE_TIMEOUT_SECONDS,
)
from skillchain.evaluation.packets import _FINAL_SYSTEM_PROMPT
from skillchain.evaluation.visual_runtime import contains_encoded_image_echo
from skillchain.synthesis.store import atomic_create_file, canonical_json_bytes


DEFAULT_TASKS = 8
DEFAULT_CONCURRENCY = 8
DEFAULT_ERROR_RATE_LIMIT = 0.0
DEFAULT_SERVICE_ERROR_RATE_LIMIT = 0.0


@dataclass(frozen=True)
class AttemptResult:
    attempt: int
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

    def payload(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class TaskResult:
    ordinal: int
    ok: bool
    error_kind: str | None
    latency_ms: int
    attempts: tuple[AttemptResult, ...]

    def payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "ok": self.ok,
            "error_kind": self.error_kind,
            "latency_ms": self.latency_ms,
            "attempts": [item.payload() for item in self.attempts],
        }


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


def _load_gemini_credential(env_file: Path) -> str:
    values = dotenv_values(env_file)
    value = values.get("GEMINI_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{env_file} does not configure GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = value.strip()
    endpoint = values.get("AIFAST_BASE_URL")
    if isinstance(endpoint, str) and endpoint.strip():
        if endpoint.strip() != config.PROVIDER_ENDPOINTS["gemini"]:
            raise RuntimeError(
                "AIFAST_BASE_URL from .env differs from the imported runtime endpoint"
            )
    return "GEMINI_API_KEY"


def _load_rubric(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    content = payload.get("content")
    expected = payload.get("content_sha256")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("rubric content must be nonblank text")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected:
        raise ValueError("rubric content SHA-256 mismatch")
    return content


def _prompt(rubric: str) -> list[dict[str, str]]:
    payload = {
        "turns": [
            {
                "role": "user",
                "content": "Identify the product using only visible evidence.",
            }
        ],
        "response_text": (
            "The image appears to show a household product, but the exact brand "
            "cannot be established from the visible evidence alone."
        ),
        "cards": [],
        "tool_evidence": [],
        "card_requirement": "forbidden",
        "rubric": rubric,
        "output_contract": final_output_contract(requires_card=False),
    }
    return [
        {"role": "system", "content": _FINAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": canonical_json_bytes(payload)
            .decode("utf-8")
            .removesuffix("\n"),
        },
    ]


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


def _call_once(
    *, ordinal: int, attempt: int, image: Path, rubric: str
) -> tuple[AttemptResult, bool]:
    started = time.perf_counter()
    try:
        response = llm.chat(
            config.PORTFOLIO_JUDGE_PROVIDER,
            _prompt(rubric),
            model=config.PORTFOLIO_JUDGE_MODEL,
            images=[str(image)],
            max_tokens=FINAL_JUDGE_ANSWER_MAX_TOKENS,
            json_mode=True,
            max_attempts=1,
            record_usage=False,
            timeout_seconds=FINAL_JUDGE_TIMEOUT_SECONDS,
        )
    except Exception as error:
        error_kind, status_code = _error_kind(error)
        return (
            AttemptResult(
                attempt=attempt,
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
            ),
            False,
        )

    response_sha256 = hashlib.sha256(response.text.encode("utf-8")).hexdigest()
    request_id_sha256 = hashlib.sha256(
        response.request_id.encode("utf-8")
    ).hexdigest()
    image_bytes = image.read_bytes()
    mime_type = mimetypes.guess_type(image.name)[0]
    if mime_type is None or contains_encoded_image_echo(
        response_text=response.text,
        tool_argument_texts=(call.arguments_json for call in response.tool_calls),
        image_bytes=image_bytes,
        mime_type=mime_type,
    ):
        return (
            AttemptResult(
                attempt=attempt,
                ok=False,
                error_kind="input_image_echo",
                status_code=None,
                latency_ms=response.latency_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                finish_reason=response.finish_reason,
                reasoning_present=response.reasoning_present,
                response_sha256=response_sha256,
                request_id_sha256=request_id_sha256,
            ),
            False,
        )
    try:
        if response.finish_reason != "stop" or response.tool_calls:
            raise EvaluatorOutputParseError(
                "final-Judge response did not stop as one text answer"
            )
        parse_final_judge_output_v4(
            response.text,
            expected_requires_card=False,
        )
    except Exception as error:
        error_kind, status_code = _error_kind(error)
        retryable = (
            attempt == 1
            and response.finish_reason == "stop"
            and not response.tool_calls
            and error_kind == "parse"
        )
        return (
            AttemptResult(
                attempt=attempt,
                ok=False,
                error_kind=error_kind,
                status_code=status_code,
                latency_ms=response.latency_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                finish_reason=response.finish_reason,
                reasoning_present=response.reasoning_present,
                response_sha256=response_sha256,
                request_id_sha256=request_id_sha256,
            ),
            retryable,
        )

    return (
        AttemptResult(
            attempt=attempt,
            ok=True,
            error_kind=None,
            status_code=None,
            latency_ms=response.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.finish_reason,
            reasoning_present=response.reasoning_present,
            response_sha256=response_sha256,
            request_id_sha256=request_id_sha256,
        ),
        False,
    )


def _one_task(
    *,
    ordinal: int,
    image: Path,
    rubric: str,
    start_gate: Event,
    tracker: _ConcurrencyTracker,
) -> TaskResult:
    tracker.enter()
    started = time.perf_counter()
    attempts: list[AttemptResult] = []
    try:
        start_gate.wait()
        for attempt_index in range(1, FINAL_JUDGE_MAX_ATTEMPTS + 1):
            attempt, retryable = _call_once(
                ordinal=ordinal,
                attempt=attempt_index,
                image=image,
                rubric=rubric,
            )
            attempts.append(attempt)
            if attempt.ok:
                return TaskResult(
                    ordinal=ordinal,
                    ok=True,
                    error_kind=None,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    attempts=tuple(attempts),
                )
            if not retryable:
                break
        return TaskResult(
            ordinal=ordinal,
            ok=False,
            error_kind=attempts[-1].error_kind,
            latency_ms=round((time.perf_counter() - started) * 1000),
            attempts=tuple(attempts),
        )
    finally:
        tracker.exit()


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _summarize(results: list[TaskResult]) -> dict[str, object]:
    successes = [item for item in results if item.ok]
    errors: dict[str, int] = {}
    for item in results:
        if item.error_kind:
            errors[item.error_kind] = errors.get(item.error_kind, 0) + 1
    attempts = [attempt for item in results for attempt in item.attempts]
    first_attempt_failures = sum(not item.attempts[0].ok for item in results)
    service_kinds = {"rate_limit", "timeout", "connection", "api_status"}
    service_failures = sum(item.error_kind in service_kinds for item in results)
    latencies = [item.latency_ms for item in successes]
    return {
        "tasks": len(results),
        "successes": len(successes),
        "failures": len(results) - len(successes),
        "observed_error_rate": (len(results) - len(successes)) / len(results),
        "service_failures": service_failures,
        "observed_service_error_rate": service_failures / len(results),
        "errors": errors,
        "provider_attempts": len(attempts),
        "retry_count": len(attempts) - len(results),
        "first_attempt_failures": first_attempt_failures,
        "first_attempt_error_rate": first_attempt_failures / len(results),
        "latency_ms": {
            "median": round(median(latencies)) if latencies else 0,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0),
        },
        "input_tokens": sum(item.input_tokens for item in attempts),
        "output_tokens": sum(item.output_tokens for item in attempts),
    }


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite capacity receipt: {path}")
    atomic_create_file(path, canonical_json_bytes(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path("specs/evaluation/portfolio-final-rubric-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--error-rate-limit", type=float, default=DEFAULT_ERROR_RATE_LIMIT
    )
    parser.add_argument(
        "--service-error-rate-limit",
        type=float,
        default=DEFAULT_SERVICE_ERROR_RATE_LIMIT,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.tasks < 1 or args.concurrency < 1 or args.concurrency > args.tasks:
        raise ValueError("tasks/concurrency must be positive and concurrency <= tasks")
    if not 0 <= args.error_rate_limit < 1:
        raise ValueError("error-rate-limit must be in [0, 1)")
    if not 0 <= args.service_error_rate_limit < 1:
        raise ValueError("service-error-rate-limit must be in [0, 1)")

    images = tuple(path.resolve(strict=True) for path in args.image)
    rubric_path = args.rubric.resolve(strict=True)
    rubric = _load_rubric(rubric_path)
    credential_source = _load_gemini_credential(args.env_file.resolve(strict=True))

    started_at = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    start_gate = Event()
    tracker = _ConcurrencyTracker()
    results: list[TaskResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                _one_task,
                ordinal=index + 1,
                image=images[index % len(images)],
                rubric=rubric,
                start_gate=start_gate,
                tracker=tracker,
            )
            for index in range(args.tasks)
        ]
        start_gate.set()
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item.ordinal)
    wall_ms = round((time.perf_counter() - wall_started) * 1000)
    completed_at = datetime.now(timezone.utc)

    summary = _summarize(results)
    passed = (
        float(summary["observed_error_rate"]) <= args.error_rate_limit
        and float(summary["observed_service_error_rate"])
        <= args.service_error_rate_limit
        and tracker.peak == args.concurrency
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gemini-final-judge-concurrency-probe",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_ms": wall_ms,
        "provider": config.PORTFOLIO_JUDGE_PROVIDER,
        "model": config.PORTFOLIO_JUDGE_MODEL,
        "endpoint": config.PROVIDER_ENDPOINTS[config.PORTFOLIO_JUDGE_PROVIDER],
        "gateway_identity_provider_attested": False,
        "scope": "exploratory_capacity_only",
        "formal_eligible": False,
        "production_authorization_consumed": False,
        "credential_source_name": credential_source,
        "credential_value_persisted": False,
        "production_wire": {
            "response_format": "json_object",
            "max_tokens": FINAL_JUDGE_ANSWER_MAX_TOKENS,
            "thinking_controls": "omitted",
            "sampling_controls": "omitted",
            "provider_internal_max_attempts": 1,
            "task_max_attempts": FINAL_JUDGE_MAX_ATTEMPTS,
            "timeout_seconds": FINAL_JUDGE_TIMEOUT_SECONDS,
            "parser": "parse_final_judge_output_v4",
        },
        "probe_policy": {
            "tasks": args.tasks,
            "max_workers": args.concurrency,
            "observed_error_rate_limit": args.error_rate_limit,
            "observed_service_error_rate_limit": args.service_error_rate_limit,
        },
        "pricing": {
            "status": "unavailable",
            "reason": "AIFast Gemini price is not frozen in the repository",
        },
        "rubric_sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest(),
        "image_sha256s": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in images
        ],
        "peak_inflight_observed": tracker.peak,
        "retry_path_exercised": int(summary["retry_count"]) > 0,
        "passed": passed,
        "summary": summary,
        "results": [item.payload() for item in results],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    _write_receipt(args.output, receipt)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "passed",
                    "wall_ms",
                    "peak_inflight_observed",
                    "summary",
                    "receipt_sha256",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
