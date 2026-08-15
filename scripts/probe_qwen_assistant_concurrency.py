"""Measure safe concurrency for the frozen production Qwen Assistant wire.

This paid integration probe alternates the two dominant Assistant action-call
shapes: image-grounded tool selection and a final answer after typed tool
evidence.  It records metrics and hashes only, never model text or credentials.
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
from typing import Any, Literal

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, RateLimitError

from skillchain import config, llm
from skillchain.synthesis.store import atomic_create_file, canonical_json_bytes
from skillchain.tools.portfolio_runtime import PORTFOLIO_SYSTEM_PROMPT


ASSISTANT_INPUT_CNY_PER_MILLION_TOKENS = Decimal("0.2")
ASSISTANT_OUTPUT_CNY_PER_MILLION_TOKENS = Decimal("2.0")
OFFICIAL_SNAPSHOT_RPM = 600
OFFICIAL_SNAPSHOT_TPM = 1_000_000
# The active Qwen3.5 profile stays below the official 600 RPM ceiling.
DEFAULT_REQUESTS_PER_SECOND = 8.0
DEFAULT_ERROR_RATE_LIMIT = 0.02
DEFAULT_SERVICE_ERROR_RATE_LIMIT = 0.0
DEFAULT_COST_CAP_CNY = Decimal("1.000000")

_ACTION_PROTOCOL = (
    "\n\nUse only the function tools supplied with this request. When a tool is "
    "needed, call exactly one function and let the runner return its real result. "
    "Do not invent or quote a tool result before receiving it. When you have "
    "enough evidence, answer the user directly in plain text. A successful tool "
    "message may include a runner-authored final_response_contract. Treat it as "
    "mandatory runtime response policy, not as retrieved evidence. The final "
    "answer must not expose hidden routing metadata, local paths, or raw runtime "
    "identifiers. Copy only public evidence_reference and product_id handles."
)
_FINAL_CONTRACT = {
    "required_sections": ["answer", "product_cards", "uncertainty"],
    "supported_rules": [
        "Include evidence_reference, product_id, and title for every returned candidate."
    ],
    "fallback_rule": "When candidates is empty, state no supported match.",
}
_TOOL = {
    "type": "function",
    "function": {
        "name": "image_product_search",
        "description": "Search the audited product gallery using the authoritative image.",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "minLength": 1},
            },
            "required": ["asset_id"],
            "additionalProperties": False,
        },
    },
}


class AssistantProbeContractError(ValueError):
    """A captured response is unusable by the production Assistant contract."""


@dataclass(frozen=True)
class ProbeResult:
    ordinal: int
    level: int
    variant: Literal["tool_selection", "final_response"]
    ok: bool
    error_kind: str | None
    status_code: int | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
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
        source_name = "KIMI_API_KEY"
        value = values.get(source_name)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"{env_file} does not configure DASHSCOPE_API_KEY or KIMI_API_KEY"
        )
    os.environ["DASHSCOPE_API_KEY"] = value.strip()
    return source_name


def _variant(ordinal: int) -> Literal["tool_selection", "final_response"]:
    return "tool_selection" if ordinal % 2 else "final_response"


def _messages(
    ordinal: int, variant: Literal["tool_selection", "final_response"]
) -> list[dict[str, Any]]:
    user = canonical_json_bytes(
        {
            "query_id": f"assistant-capacity-{ordinal:04d}",
            "asset_id": "query_asset",
            "image_path": "query_image",
            "turns": [
                {
                    "role": "user",
                    "content": "Find the exact catalog product shown in this image.",
                }
            ],
        }
    ).decode("utf-8")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PORTFOLIO_SYSTEM_PROMPT + _ACTION_PROTOCOL},
        {"role": "user", "content": user},
    ]
    if variant == "final_response":
        call_id = f"capacity-tool-{ordinal:04d}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "image_product_search",
                                "arguments": '{"asset_id":"query_asset"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": canonical_json_bytes(
                        {
                            "support_status": "supported",
                            "candidates": [
                                {
                                    "evidence_reference": "tool-call-1-candidate-1",
                                    "product_id": "public-product-001",
                                    "title": "Blue household product",
                                }
                            ],
                            "final_response_contract": _FINAL_CONTRACT,
                        }
                    ).decode("utf-8"),
                },
            ]
        )
    return messages


def _cost(input_tokens: int, output_tokens: int) -> Decimal:
    return (
        Decimal(input_tokens) * ASSISTANT_INPUT_CNY_PER_MILLION_TOKENS
        + Decimal(output_tokens) * ASSISTANT_OUTPUT_CNY_PER_MILLION_TOKENS
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
    if isinstance(error, (AssistantProbeContractError, llm.LLMContractError)):
        return "contract", None
    return type(error).__name__, getattr(error, "status_code", None)


def _response_hash(response: llm.LLMResponse) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "text": response.text,
                "tool_calls": [
                    item.model_dump(mode="json") for item in response.tool_calls
                ],
            }
        )
    ).hexdigest()


def _validate_response(
    response: llm.LLMResponse,
    variant: Literal["tool_selection", "final_response"],
) -> None:
    if (
        response.provider != "qwen"
        or response.requested_model != config.ASSISTANT_MODEL
        or response.response_model != config.ASSISTANT_MODEL
        or response.reasoning_present
        or response.usage.input_tokens <= 0
        or response.usage.output_tokens <= 0
    ):
        raise AssistantProbeContractError("Assistant response identity drifted")
    if variant == "tool_selection":
        if response.finish_reason != "tool_calls" or len(response.tool_calls) != 1:
            raise AssistantProbeContractError("tool-selection response shape drifted")
        call = response.tool_calls[0]
        try:
            arguments = json.loads(call.arguments_json)
        except json.JSONDecodeError as error:
            raise AssistantProbeContractError(
                "tool arguments are invalid JSON"
            ) from error
        if call.name != "image_product_search" or arguments != {
            "asset_id": "query_asset"
        }:
            raise AssistantProbeContractError("tool-selection arguments drifted")
        return
    if (
        response.finish_reason != "stop"
        or response.tool_calls
        or not response.text.strip()
    ):
        raise AssistantProbeContractError("final Assistant response shape drifted")
    # The probe's synthetic tool evidence is not an evaluation sample. Capacity
    # success therefore uses the production transport contract (plain nonblank
    # stop text); semantic section scoring remains an evaluation concern.


def _one_call(
    *,
    ordinal: int,
    level: int,
    image: Path,
    pacer: _StartPacer,
    tracker: _ConcurrencyTracker,
) -> ProbeResult:
    variant = _variant(ordinal)
    pacer.wait()
    tracker.enter()
    started = time.perf_counter()
    response: llm.LLMResponse | None = None
    try:
        response = llm.chat(
            "qwen",
            _messages(ordinal, variant),
            model=config.ASSISTANT_MODEL,
            images=[str(image)],
            temperature=0.0,
            top_p=1.0,
            seed=None,
            max_tokens=4096,
            tools=[_TOOL],
            tool_choice="auto",
            parallel_tool_calls=False,
            thinking=False,
            max_attempts=1,
            record_usage=False,
            timeout_seconds=180,
        )
        _validate_response(response, variant)
    except Exception as error:
        error_kind, status_code = _error_kind(error)
        return ProbeResult(
            ordinal=ordinal,
            level=level,
            variant=variant,
            ok=False,
            error_kind=error_kind,
            status_code=status_code,
            latency_ms=(
                max(0, round((time.perf_counter() - started) * 1000))
                if response is None
                else response.latency_ms
            ),
            input_tokens=0 if response is None else response.usage.input_tokens,
            output_tokens=0 if response is None else response.usage.output_tokens,
            finish_reason=None if response is None else response.finish_reason,
            response_sha256=None if response is None else _response_hash(response),
            request_id_sha256=(
                None
                if response is None
                else hashlib.sha256(response.request_id.encode("utf-8")).hexdigest()
            ),
            cost_cny=(
                "0"
                if response is None
                else format(
                    _cost(response.usage.input_tokens, response.usage.output_tokens),
                    "f",
                )
            ),
        )
    finally:
        tracker.exit()
    return ProbeResult(
        ordinal=ordinal,
        level=level,
        variant=variant,
        ok=True,
        error_kind=None,
        status_code=None,
        latency_ms=response.latency_ms,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        finish_reason=response.finish_reason,
        response_sha256=_response_hash(response),
        request_id_sha256=hashlib.sha256(
            response.request_id.encode("utf-8")
        ).hexdigest(),
        cost_cny=format(
            _cost(response.usage.input_tokens, response.usage.output_tokens), "f"
        ),
    )


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _summarize(results: list[ProbeResult], peak_inflight: int) -> dict[str, object]:
    successful = [item for item in results if item.ok]
    failures = len(results) - len(successful)
    service_failures = sum(
        item.error_kind in {"rate_limit", "timeout", "connection", "api_status"}
        for item in results
    )
    errors: dict[str, int] = {}
    for item in results:
        if item.error_kind:
            errors[item.error_kind] = errors.get(item.error_kind, 0) + 1
    latencies = [item.latency_ms for item in successful]
    return {
        "level": results[0].level,
        "calls": len(results),
        "successes": len(successful),
        "failures": failures,
        "observed_error_rate": failures / len(results),
        "service_failures": service_failures,
        "observed_service_error_rate": service_failures / len(results),
        "errors": errors,
        "variant_counts": {
            variant: {
                "calls": sum(item.variant == variant for item in results),
                "successes": sum(
                    item.variant == variant and item.ok for item in results
                ),
            }
            for variant in ("tool_selection", "final_response")
        },
        "latency_ms": {
            "median": round(median(latencies)) if latencies else 0,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0),
        },
        "input_tokens": sum(item.input_tokens for item in results),
        "output_tokens": sum(item.output_tokens for item in results),
        "cost_cny": format(sum(Decimal(item.cost_cny) for item in results), "f"),
        "peak_inflight_observed": peak_inflight,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--requests-per-second", type=float, default=DEFAULT_REQUESTS_PER_SECOND
    )
    parser.add_argument(
        "--error-rate-limit", type=float, default=DEFAULT_ERROR_RATE_LIMIT
    )
    parser.add_argument(
        "--service-error-rate-limit",
        type=float,
        default=DEFAULT_SERVICE_ERROR_RATE_LIMIT,
    )
    parser.add_argument("--cost-cap-cny", type=Decimal, default=DEFAULT_COST_CAP_CNY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.calls < 1 or args.max_workers < 1:
        raise ValueError("calls and max-workers must be positive")
    if args.requests_per_second <= 0:
        raise ValueError("requests-per-second must be positive")
    if not 0 <= args.error_rate_limit < 1 or not 0 <= args.service_error_rate_limit < 1:
        raise ValueError("error limits must be in [0, 1)")
    if args.cost_cap_cny <= 0:
        raise ValueError("cost-cap-cny must be positive")
    images = tuple(path.resolve(strict=True) for path in args.image)
    credential_source = _load_dashscope_credential(args.env_file.resolve(strict=True))

    started_at = datetime.now(timezone.utc)
    pacer = _StartPacer(args.requests_per_second)
    tracker = _ConcurrencyTracker()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                _one_call,
                ordinal=ordinal,
                level=args.max_workers,
                image=images[(ordinal - 1) % len(images)],
                pacer=pacer,
                tracker=tracker,
            )
            for ordinal in range(1, args.calls + 1)
        ]
        results = sorted(
            (future.result() for future in as_completed(futures)),
            key=lambda item: item.ordinal,
        )
    completed_at = datetime.now(timezone.utc)
    summary = _summarize(results, tracker.peak)
    if Decimal(str(summary["cost_cny"])) > args.cost_cap_cny:
        raise RuntimeError("Assistant probe exceeded its cost cap")
    passes = (
        float(summary["observed_error_rate"]) <= args.error_rate_limit
        and float(summary["observed_service_error_rate"])
        <= args.service_error_rate_limit
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen-assistant-concurrency-probe",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "provider": "qwen",
        "model": config.ASSISTANT_MODEL,
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "credential_source_name": credential_source,
        "credential_value_persisted": False,
        "production_wire": {
            "enable_thinking": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": None,
            "max_tokens": 4096,
            "function_calling": True,
            "parallel_tool_calls": False,
            "max_attempts": 1,
            "timeout_seconds": 180,
            "variants": ["tool_selection", "final_response"],
        },
        "official_snapshot_limits": {
            "requests_per_minute": OFFICIAL_SNAPSHOT_RPM,
            "tokens_per_minute": OFFICIAL_SNAPSHOT_TPM,
        },
        "probe_policy": {
            "calls": args.calls,
            "max_workers": args.max_workers,
            "requests_per_second": args.requests_per_second,
            "observed_error_rate_limit": args.error_rate_limit,
            "observed_service_error_rate_limit": args.service_error_rate_limit,
            "cost_cap_cny": format(args.cost_cap_cny, "f"),
        },
        "image_sha256s": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in images
        ],
        "summary": summary,
        "passes": passes,
        "calls": len(results),
        "successes": sum(item.ok for item in results),
        "failures": sum(not item.ok for item in results),
        "total_cost_cny": summary["cost_cny"],
        "results": [item.payload() for item in results],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Assistant receipt: {args.output}")
    atomic_create_file(args.output, canonical_json_bytes(receipt))
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "passes",
                    "calls",
                    "successes",
                    "failures",
                    "total_cost_cny",
                    "summary",
                    "receipt_sha256",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
