"""可审计的统一 LLM 调用入口。

默认单元测试不会访问网络。真实 provider 冒烟测试必须显式使用
``pytest -m integration``。每个成功调用返回 :class:`LLMResponse`，并把
provider/model/endpoint/request、usage、finish reason 与总延迟追加到 usage log。
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import anthropic
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skillchain import config

Provider = Literal["qwen", "deepseek", "kimi", "gemini", "claude"]
ReasoningEffort = Literal["max"]


class LLMContractError(RuntimeError):
    """Raised when a provider response violates the auditable runtime contract."""


class LLMTimeoutError(TimeoutError):
    """Raised when one provider request exceeds its caller-owned deadline."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMUsage(_FrozenModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMToolCall(_FrozenModel):
    call_id: str
    name: str
    arguments_json: str

    @field_validator("call_id", "name")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tool call identity must not be blank")
        return value


class LLMJsonSchemaSpec(_FrozenModel):
    """One provider-facing strict JSON-Schema output contract."""

    name: str
    strict: Literal[True] = True
    schema_: dict[str, Any] = Field(alias="schema", serialization_alias="schema")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value):
            raise ValueError("JSON-Schema response name must be canonical")
        return value

    @field_validator("schema_")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object" or not isinstance(
            value.get("properties"), dict
        ):
            raise ValueError("JSON-Schema response root must be an object")
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise ValueError("JSON-Schema response contract exceeds 64 KiB")
        return json.loads(encoded)


class LLMJsonSchemaResponseFormat(_FrozenModel):
    """Typed OpenAI-compatible ``response_format=json_schema`` request."""

    type: Literal["json_schema"] = "json_schema"
    json_schema: LLMJsonSchemaSpec


class LLMResponse(_FrozenModel):
    """Provider-independent response with explicit request/runtime identity."""

    schema_version: Literal[1] = 1
    provider: Provider
    endpoint: str
    requested_model: str
    response_model: str
    request_id: str
    text: str
    tool_calls: tuple[LLMToolCall, ...] = ()
    usage: LLMUsage
    finish_reason: str
    latency_ms: int = Field(ge=0)
    reasoning_present: bool = Field(
        default=False,
        exclude_if=lambda value: value is False,
    )
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    reasoning_bytes: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    reasoning_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )

    @field_validator(
        "endpoint", "requested_model", "response_model", "request_id", "finish_reason"
    )
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("LLM response identity fields must not be blank")
        return value

    @model_validator(mode="after")
    def validate_model_identity(self) -> LLMResponse:
        if self.response_model != self.requested_model:
            raise ValueError(
                "provider response model does not match requested model: "
                f"{self.response_model!r} != {self.requested_model!r}"
            )
        expected_endpoint = config.PROVIDER_ENDPOINTS[self.provider]
        if self.endpoint != expected_endpoint:
            raise ValueError(
                "provider endpoint identity mismatch: "
                f"{self.endpoint!r} != {expected_endpoint!r}"
            )
        reasoning_observed = self.reasoning_bytes > 0 or (
            self.reasoning_tokens is not None and self.reasoning_tokens > 0
        )
        if self.reasoning_present is not reasoning_observed:
            raise ValueError(
                "reasoning_present differs from the retained reasoning metadata"
            )
        if (self.reasoning_bytes > 0) is (self.reasoning_sha256 is None):
            raise ValueError(
                "reasoning bytes and SHA-256 must be present or absent together"
            )
        return self


_clients: dict[str, object] = {}


def _validated_provider_model(provider: str, model: str | None) -> tuple[Provider, str]:
    if provider not in config.PROVIDER_ENDPOINTS:
        raise ValueError(f"未知 provider: {provider}")
    typed_provider: Provider = provider  # type: ignore[assignment]
    selected_model = model or config.PROVIDER_DEFAULT_MODELS[provider]
    if selected_model != selected_model.strip() or not selected_model:
        raise ValueError("model identity must be non-blank and trimmed")
    prefixes = config.PROVIDER_MODEL_PREFIXES[provider]
    if not selected_model.lower().startswith(prefixes):
        raise ValueError(
            f"model {selected_model!r} does not belong to provider {provider!r}; "
            f"expected prefix in {prefixes!r}"
        )
    return typed_provider, selected_model


def _client(provider: Provider) -> Any:
    if provider in _clients:
        return _clients[provider]
    key_var = config.PROVIDER_API_KEY_ENV[provider]
    api_key = os.environ.get(key_var)
    if not api_key:
        raise RuntimeError(f"环境变量 {key_var} 未配置（见 .env.example）")
    endpoint = config.PROVIDER_ENDPOINTS[provider]
    if provider == "claude":
        client: object = anthropic.Anthropic(api_key=api_key, base_url=endpoint)
    else:
        # The repository owns retries at the call-contract layer.  Disable the
        # OpenAI SDK's independent default retries so a one-attempt formal
        # authoring authorization cannot fan out into multiple HTTP attempts.
        client = OpenAI(base_url=endpoint, api_key=api_key, max_retries=0)
    _clients[provider] = client
    return client


def _encode_image(path: str) -> tuple[str, str]:
    media_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    with Path(path).open("rb") as file:
        return media_type, base64.standard_b64encode(file.read()).decode("ascii")


def _log_usage(response: LLMResponse) -> None:
    config.USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": response.schema_version,
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": response.provider,
        "endpoint": response.endpoint,
        "requested_model": response.requested_model,
        "response_model": response.response_model,
        "request_id": response.request_id,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "finish_reason": response.finish_reason,
        "latency_ms": response.latency_ms,
        "tool_call_count": len(response.tool_calls),
        "reasoning_present": response.reasoning_present,
        "reasoning_tokens": response.reasoning_tokens,
        "reasoning_bytes": response.reasoning_bytes,
        "reasoning_sha256": response.reasoning_sha256,
    }
    with config.USAGE_LOG.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _attach_images_openai(
    messages: list[dict[str, Any]], images: list[str]
) -> list[dict[str, Any]]:
    parts = [
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        for media_type, data in (_encode_image(path) for path in images)
    ]
    attached = [dict(message) for message in messages]
    for index in range(len(attached) - 1, -1, -1):
        message = attached[index]
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"] = parts + [{"type": "text", "text": message["content"]}]
            return attached
    raise ValueError(
        "images= requires a user message with plain-text content; "
        "supply prebuilt multimodal content without images= instead"
    )


def _attach_images_claude(
    messages: list[dict[str, Any]], images: list[str]
) -> list[dict[str, Any]]:
    parts = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }
        for media_type, data in (_encode_image(path) for path in images)
    ]
    last = messages[-1]
    return messages[:-1] + [
        {
            "role": last["role"],
            "content": parts + [{"type": "text", "text": last["content"]}],
        }
    ]


def _contains_embedded_multimodal_content(value: object) -> bool:
    if isinstance(value, list):
        return any(_contains_embedded_multimodal_content(item) for item in value)
    if not isinstance(value, dict):
        return False
    if set(value) & {"image", "image_url", "input_image", "source"}:
        return True
    item_type = value.get("type")
    if isinstance(item_type, str) and item_type.casefold() in {
        "image",
        "image_url",
        "input_image",
    }:
        return True
    return any(_contains_embedded_multimodal_content(item) for item in value.values())


def _extract_openai_reasoning_metadata(
    message: object,
    usage: object | None,
) -> dict[str, object]:
    """Commit to provider reasoning without retaining chain-of-thought text."""

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is None:
        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            reasoning_content = model_extra.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise LLMContractError("provider reasoning_content must be text when present")
    reasoning_bytes_value = (
        b"" if reasoning_content is None else reasoning_content.encode("utf-8")
    )

    completion_details = getattr(usage, "completion_tokens_details", None)
    if isinstance(completion_details, dict):
        reasoning_tokens = completion_details.get("reasoning_tokens")
    else:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
    if reasoning_tokens is not None and (
        isinstance(reasoning_tokens, bool)
        or not isinstance(reasoning_tokens, int)
        or reasoning_tokens < 0
    ):
        raise LLMContractError(
            "provider reasoning token count must be a non-negative integer"
        )

    reasoning_present = bool(reasoning_bytes_value) or bool(reasoning_tokens)
    return {
        "reasoning_present": reasoning_present,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_bytes": len(reasoning_bytes_value),
        "reasoning_sha256": (
            hashlib.sha256(reasoning_bytes_value).hexdigest()
            if reasoning_bytes_value
            else None
        ),
    }


def chat(
    provider: str,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    images: list[str] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = 2048,
    max_completion_tokens: int | None = None,
    json_mode: bool = False,
    response_format: LLMJsonSchemaResponseFormat | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
    parallel_tool_calls: bool | None = None,
    thinking: bool = False,
    thinking_budget: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    max_attempts: int = 3,
    record_usage: bool = True,
    timeout_seconds: float | None = None,
) -> LLMResponse:
    """Call one configured provider and return a fully identified response."""

    if provider == "claude_code":
        raise RuntimeError(
            "Creator/Refiner 由 Claude Code 会话承担（人机协作批处理），不经 API 调用。"
            "各 stage 模块应将任务写为 prompt 文件供会话处理，而非调用 llm.chat()。"
        )
    typed_provider, selected_model = _validated_provider_model(provider, model)
    if not messages:
        raise ValueError("messages must not be empty")
    if typed_provider == "deepseek" and (
        images or _contains_embedded_multimodal_content(messages)
    ):
        raise ValueError("deepseek 无视觉能力，不能传 images")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive when supplied")
    if max_completion_tokens is not None and max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be positive when supplied")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when supplied")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative")
    if typed_provider == "claude" and seed is not None:
        raise ValueError("claude endpoint does not expose deterministic seed control")
    if typed_provider == "qwen" and selected_model == config.FEEDBACK_JUDGE_MODEL:
        if reasoning_effort is not None:
            raise ValueError("Qwen3.7 Plus Feedback does not expose reasoning_effort")
        if thinking is not config.FEEDBACK_JUDGE_THINKING:
            raise ValueError("Qwen3.7 Plus Feedback requires enable_thinking=true")
        if thinking_budget != config.FEEDBACK_JUDGE_THINKING_BUDGET:
            raise ValueError("Qwen3.7 Plus Feedback requires thinking_budget=2048")
        if max_tokens is not None:
            raise ValueError("Qwen3.7 Plus Feedback requires max_tokens to be omitted")
        if max_completion_tokens != config.FEEDBACK_JUDGE_MAX_COMPLETION_TOKENS:
            raise ValueError(
                "Qwen3.7 Plus Feedback requires max_completion_tokens=4096"
            )
        if timeout_seconds != config.FEEDBACK_JUDGE_TIMEOUT_SECONDS:
            raise ValueError("Qwen3.7 Plus Feedback requires timeout_seconds=600")
        if temperature is not None or top_p is not None:
            raise ValueError("Qwen3.7 Plus Feedback sampling controls must be omitted")
        if seed is not None:
            raise ValueError("Qwen3.7 Plus Feedback contract does not expose seed")
        if json_mode or response_format is None:
            raise ValueError(
                "Qwen3.7 Plus Feedback requires typed response_format=json_schema"
            )
    elif typed_provider == "kimi":
        if max_tokens is None or max_completion_tokens is not None:
            raise ValueError(
                "historical Kimi Feedback requires max_tokens and omits "
                "max_completion_tokens"
            )
        if selected_model != config.LEGACY_KIMI_FEEDBACK_JUDGE_MODEL:
            raise ValueError(
                "historical Kimi Feedback adapter requires exact model "
                f"{config.LEGACY_KIMI_FEEDBACK_JUDGE_MODEL!r}"
            )
        if reasoning_effort is not None:
            raise ValueError(
                "kimi-k2.6 uses enable_thinking; reasoning_effort is unsupported"
            )
        if thinking is not config.LEGACY_KIMI_FEEDBACK_JUDGE_THINKING:
            raise ValueError("kimi-k2.6 Feedback requires enable_thinking=false")
        if thinking_budget is not None:
            raise ValueError(
                "non-thinking kimi-k2.6 Feedback does not accept thinking_budget"
            )
        if temperature != config.LEGACY_KIMI_FEEDBACK_JUDGE_TEMPERATURE:
            raise ValueError(
                "kimi-k2.6 Feedback temperature must be "
                f"{config.LEGACY_KIMI_FEEDBACK_JUDGE_TEMPERATURE}"
            )
        if top_p != config.LEGACY_KIMI_FEEDBACK_JUDGE_TOP_P:
            raise ValueError("kimi-k2.6 Feedback top_p must be 0.95")
        if seed is not None:
            raise ValueError("kimi-k2.6 Feedback contract does not expose seed")
        if json_mode:
            raise ValueError("kimi-k2.6 does not support structured JSON mode")
    elif typed_provider == "gemini":
        if max_tokens is None or max_completion_tokens is not None:
            raise ValueError(
                "Gemini adapter requires max_tokens and omits max_completion_tokens"
            )
        if thinking_budget is not None:
            raise ValueError("thinking_budget is only supported by the Kimi adapter")
        if selected_model != config.PORTFOLIO_JUDGE_MODEL:
            raise ValueError(
                "Gemini final-Judge adapter requires exact model "
                f"{config.PORTFOLIO_JUDGE_MODEL!r}"
            )
        if thinking:
            raise ValueError(
                "AIFast Gemini final-Judge thinking control is unavailable; "
                "use the disclosed gateway default"
            )
        if reasoning_effort is not None:
            raise ValueError(
                "AIFast Gemini final-Judge route does not expose reasoning_effort"
            )
        if temperature is not None or top_p is not None:
            raise ValueError(
                "Gemini 3.6 requests must omit deprecated temperature/top_p"
            )
        if seed is not None:
            raise ValueError("AIFast Gemini final-Judge contract does not expose seed")
    else:
        if max_tokens is None or max_completion_tokens is not None:
            raise ValueError(
                "this adapter requires max_tokens and does not support "
                "max_completion_tokens"
            )
        if thinking_budget is not None:
            raise ValueError("thinking_budget is only supported by the Kimi adapter")
        if reasoning_effort is not None:
            raise ValueError("reasoning_effort is unsupported by this adapter")
    if tools is not None and not tools:
        raise ValueError("tools must be non-empty when supplied")
    if json_mode and response_format is not None:
        raise ValueError("json_mode and response_format are mutually exclusive")
    if response_format is not None and not (
        typed_provider == "qwen" and selected_model == config.FEEDBACK_JUDGE_MODEL
    ):
        raise ValueError(
            "typed JSON-Schema response format is limited to Qwen3.7 Feedback"
        )
    if (json_mode or response_format is not None) and tools is not None:
        raise ValueError("structured response format and tools are mutually exclusive")
    if tool_choice is not None and tools is None:
        raise ValueError("tool_choice requires tools")
    if parallel_tool_calls is not None and tools is None:
        raise ValueError("parallel_tool_calls requires tools")
    if typed_provider == "claude" and tools is not None:
        raise ValueError(
            "OpenAI-compatible tool definitions are unsupported by the claude adapter"
        )

    started_at = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = _chat_once(
                typed_provider,
                messages,
                selected_model,
                images,
                temperature,
                top_p,
                seed,
                max_tokens,
                max_completion_tokens,
                json_mode,
                response_format,
                tools,
                tool_choice,
                parallel_tool_calls,
                thinking,
                thinking_budget,
                reasoning_effort,
                timeout_seconds,
            )
            response = response.model_copy(
                update={
                    "latency_ms": max(
                        0, round((time.perf_counter() - started_at) * 1000)
                    )
                }
            )
            if record_usage:
                _log_usage(response)
            return response
        except (APITimeoutError, anthropic.APITimeoutError) as error:
            raise LLMTimeoutError("LLM request exceeded its timeout") from error
        except (
            RateLimitError,
            APIConnectionError,
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
        ) as error:
            last_error = error
        except (APIStatusError, anthropic.APIStatusError) as error:
            if getattr(error, "status_code", 0) < 500:
                raise
            last_error = error
        if attempt + 1 < max_attempts:
            delay = min(2**attempt + random.uniform(0, 1), 30)
            time.sleep(delay)
    if last_error is None:  # defensive; max_attempts validation makes this unreachable
        raise LLMContractError("LLM call failed without a captured provider error")
    raise last_error


def _chat_once(
    provider: Provider,
    messages: list[dict[str, Any]],
    model: str,
    images: list[str] | None,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
    max_tokens: int | None,
    max_completion_tokens: int | None,
    json_mode: bool,
    response_format: LLMJsonSchemaResponseFormat | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: dict[str, Any] | str | None,
    parallel_tool_calls: bool | None,
    thinking: bool = False,
    thinking_budget: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    timeout_seconds: float | None = None,
) -> LLMResponse:
    client = _client(provider)
    endpoint = config.PROVIDER_ENDPOINTS[provider]
    if provider == "claude":
        if max_tokens is None:
            raise LLMContractError("Claude requires max_tokens")
        system = (
            "\n".join(
                str(message["content"])
                for message in messages
                if message["role"] == "system"
            )
            or anthropic.NOT_GIVEN
        )
        turns = [message for message in messages if message["role"] != "system"]
        if images:
            turns = _attach_images_claude(turns, images)
        claude_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": turns,
        }
        if temperature is not None:
            claude_kwargs["temperature"] = temperature
        if top_p is not None:
            claude_kwargs["top_p"] = top_p
        if timeout_seconds is not None:
            claude_kwargs["timeout"] = timeout_seconds
        raw_response = client.messages.create(
            **claude_kwargs,
        )
        tool_calls = tuple(
            LLMToolCall(
                call_id=block.id,
                name=block.name,
                arguments_json=json.dumps(
                    block.input,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for block in raw_response.content
            if block.type == "tool_use"
        )
        return LLMResponse(
            provider=provider,
            endpoint=endpoint,
            requested_model=model,
            response_model=raw_response.model,
            request_id=raw_response.id,
            text="".join(
                block.text for block in raw_response.content if block.type == "text"
            ),
            tool_calls=tool_calls,
            usage=LLMUsage(
                input_tokens=raw_response.usage.input_tokens,
                output_tokens=raw_response.usage.output_tokens,
            ),
            finish_reason=raw_response.stop_reason,
            latency_ms=0,
        )

    openai_messages = messages
    if images:
        openai_messages = _attach_images_openai(messages, images)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if seed is not None:
        kwargs["seed"] = seed
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    elif response_format is not None:
        kwargs["response_format"] = response_format.model_dump(
            mode="json", by_alias=True
        )
    if provider == "qwen" and model == config.FEEDBACK_JUDGE_MODEL:
        kwargs["stream"] = False
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = parallel_tool_calls
    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds
    if provider in {"qwen", "deepseek"}:
        kwargs["extra_body"] = {"enable_thinking": thinking}
        if thinking_budget is not None:
            kwargs["extra_body"]["thinking_budget"] = thinking_budget
    elif provider == "kimi":
        kwargs["extra_body"] = {"enable_thinking": thinking}
    raw_response = client.chat.completions.create(**kwargs)
    if len(raw_response.choices) != 1:
        raise LLMContractError("provider response must contain exactly one choice")
    choice = raw_response.choices[0]
    message = choice.message
    tool_calls = tuple(
        LLMToolCall(
            call_id=call.id,
            name=call.function.name,
            arguments_json=call.function.arguments,
        )
        for call in (message.tool_calls or ())
    )
    usage = raw_response.usage
    if provider == "qwen" and model == config.FEEDBACK_JUDGE_MODEL:
        prompt_tokens = None if usage is None else usage.prompt_tokens
        completion_tokens = None if usage is None else usage.completion_tokens
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or type(completion_tokens) is not int
            or completion_tokens < 0
        ):
            raise LLMContractError(
                "Qwen3.7 Feedback requires captured positive input and "
                "non-negative output token usage"
            )
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is not None and (
            type(total_tokens) is not int
            or total_tokens != prompt_tokens + completion_tokens
        ):
            raise LLMContractError(
                "Qwen3.7 Feedback provider token usage is inconsistent"
            )
    try:
        return LLMResponse(
            provider=provider,
            endpoint=endpoint,
            requested_model=model,
            response_model=raw_response.model,
            request_id=raw_response.id,
            text=message.content or "",
            tool_calls=tool_calls,
            usage=LLMUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason,
            latency_ms=0,
            **_extract_openai_reasoning_metadata(message, usage),
        )
    except ValueError as error:
        if provider == "qwen" and model == config.FEEDBACK_JUDGE_MODEL:
            raise LLMContractError(
                "Qwen3.7 Feedback response identity violates the runtime contract"
            ) from error
        raise


__all__ = [
    "LLMContractError",
    "LLMResponse",
    "LLMJsonSchemaResponseFormat",
    "LLMJsonSchemaSpec",
    "LLMToolCall",
    "LLMTimeoutError",
    "LLMUsage",
    "chat",
]
