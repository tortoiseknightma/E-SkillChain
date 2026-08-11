"""Unit contract tests plus explicit opt-in provider integration smoke tests."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
from openai import APITimeoutError
import pytest
from PIL import Image
from pydantic import ValidationError

from skillchain import config
from skillchain import llm
from skillchain.evaluation.evaluator_outputs import (
    parse_final_judge_output,
    parse_visual_feedback_output,
)
from skillchain.llm import LLMResponse, chat
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


def _requires(var: str):
    return pytest.mark.skipif(
        not os.environ.get(var),
        reason=f"{var} 未配置（复制 .env.example 为 .env 并填入密钥）",
    )


@pytest.fixture(autouse=True)
def clear_clients():
    llm._clients.clear()
    yield
    llm._clients.clear()


@pytest.fixture
def red_image(tmp_path):
    path = tmp_path / "red.png"
    Image.new("RGB", (64, 64), (220, 20, 20)).save(path)
    return str(path)


@pytest.mark.integration
@_requires("DASHSCOPE_API_KEY")
def test_qwen_vision_hello(red_image):
    response = chat(
        "qwen",
        [{"role": "user", "content": "这张图片的主色调是什么颜色？用一个词回答。"}],
        images=[red_image],
        max_tokens=64,
    )
    assert "红" in response.text, f"Qwen 视觉冒烟异常，回复：{response.text!r}"


@pytest.mark.integration
@_requires("DASHSCOPE_API_KEY")
def test_kimi_k26_visual_feedback_json(red_image):
    prompt = (
        "Inspect the image. Return exactly one JSON object with no Markdown: "
        '{"schema_version":1,"summary":"a nonblank summary naming the visible '
        'main color","rule_violations":[],"ideal_response_gaps":[],'
        '"skill_suggestions":[]}. Return no other keys or text.'
    )
    response = chat(
        "kimi",
        [{"role": "user", "content": prompt}],
        model=config.FEEDBACK_JUDGE_MODEL,
        images=[red_image],
        temperature=config.FEEDBACK_JUDGE_TEMPERATURE,
        top_p=config.FEEDBACK_JUDGE_TOP_P,
        thinking=config.FEEDBACK_JUDGE_THINKING,
        max_tokens=256,
        max_attempts=1,
    )
    parsed = parse_visual_feedback_output(response.text)
    assert "red" in parsed.summary.casefold() or "红" in parsed.summary


@pytest.mark.integration
@_requires("GEMINI_API_KEY")
def test_gemini_36_visual_judge_json(red_image):
    prompt = (
        "Inspect the image and the candidate claim 'the image is blue'. "
        "Return exactly one JSON object with no Markdown or prose: "
        '{"schema_version":1,"requires_card":false,"dimensions":['
        '{"dimension":"CA","score":0},{"dimension":"CQ","score":10},'
        '{"dimension":"TCR","score":0}]}. Return no other keys.'
    )
    response = chat(
        "gemini",
        [{"role": "user", "content": prompt}],
        model=config.PORTFOLIO_JUDGE_MODEL,
        images=[red_image],
        temperature=config.PORTFOLIO_JUDGE_TEMPERATURE,
        top_p=config.PORTFOLIO_JUDGE_TOP_P,
        thinking=config.PORTFOLIO_JUDGE_THINKING,
        json_mode=True,
        max_tokens=256,
        max_attempts=1,
    )
    parsed = parse_final_judge_output(
        response.text,
        expected_requires_card=False,
    )
    assert [item.dimension for item in parsed.dimensions] == ["CA", "CQ", "TCR"]


def test_active_portfolio_role_selection_is_self_hashed_and_matches_runtime():
    historical = config.ROOT / "specs" / "authoring" / "model-role-selection-v7.json"
    path = config.ROOT / "specs" / "authoring" / "model-role-selection-v8.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {
        key: value for key, value in payload.items() if key != "selection_sha256"
    }

    assert sha256_bytes(historical.read_bytes()) == (
        "fbfbe9437731052743b3025962a22e4d3cd6432c2212b7cc418b2118e9fd2ba4"
    )
    assert sha256_bytes(path.read_bytes()) == (
        "31ebdbe0af7d844ce3bf88bb0ab17d0c05640082618a4c24e1f022e38868efe8"
    )
    assert payload["selection_sha256"] == sha256_bytes(canonical_json_bytes(unsigned))
    feedback = payload["feedback_evaluator"]
    assert (feedback["provider"], feedback["model"]) == (
        config.FEEDBACK_JUDGE_PROVIDER,
        config.FEEDBACK_JUDGE_MODEL,
    )
    assert feedback["cache_namespace"] == "feedback-evaluator-v9"
    assert feedback["endpoint_configuration"] == "DASHSCOPE_BASE_URL"
    assert feedback["prompt_policy_version"] == (
        "visual-feedback-response-schema-v1-prompt-v5"
    )
    assert feedback["prompt_policy_sha256"] == (
        "024c1a8831b3497a904272cd9b9c2352fbbb97b633755388536da8475cbdfffb"
    )
    assert feedback["transport_policy_version"] == (
        "visual-feedback-qwen-dashscope-json-schema-v5"
    )
    assert feedback["transport_policy_sha256"] == (
        "092f36be5eb08e4fd58edc888fe273bee2b4295059048dbd9715d472e534f20c"
    )
    assert feedback["requested_response_format"] == "json_schema"
    assert feedback["requested_json_schema_strict"] is True
    assert feedback["max_tokens"] is None
    assert feedback["max_completion_tokens"] == 4096
    assert feedback["max_completion_tokens_documented_upper_tolerance_tokens"] == 10
    assert feedback["input_token_reservation_ceiling_per_call"] == 20_000
    assert feedback["output_token_reservation_ceiling_per_call"] == 4106
    assert feedback["per_call_reservation_cny"] == "0.072848000000"
    assert feedback["worst_case_reservation_cny"] == "3.496704000000"
    final = payload["offline_judge"]
    assert (final["provider"], final["model"]) == (
        config.PORTFOLIO_JUDGE_PROVIDER,
        config.PORTFOLIO_JUDGE_MODEL,
    )
    assert final["cache_namespace"] == "final-evaluator-v11"
    assert final["endpoint_configuration"] == "AIFAST_BASE_URL"
    assert final["transport_policy_version"] == (
        "final-judge-json-object-response-format-v1"
    )
    assert final["requested_response_format"] == "json_object"
    assert payload["evaluator_isolation"]["shared_model_runtime"] is False


@pytest.mark.integration
@_requires("DASHSCOPE_API_KEY")
def test_deepseek_hello():
    response = chat(
        "deepseek",
        [{"role": "user", "content": "回复两个字：收到"}],
        temperature=0.0,
        max_tokens=16,
    )
    assert response.text.strip(), "DeepSeek 返回为空"


def test_openai_response_is_structured_and_usage_is_identified(tmp_path, monkeypatch):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="recipe_lookup", arguments='{"dish":"汤"}'),
        )
        message = SimpleNamespace(content="完成", tool_calls=[tool_call])
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        return SimpleNamespace(
            id="req-qwen-1",
            model=config.BACKBONE_MODEL,
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    llm._clients["qwen"] = fake_client
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    response = chat(
        "qwen",
        [{"role": "user", "content": "test"}],
        temperature=0.0,
        top_p=0.9,
        seed=7,
        json_mode=True,
    )

    assert isinstance(response, LLMResponse)
    assert response.provider == "qwen"
    assert response.endpoint == config.DASHSCOPE_BASE_URL
    assert response.requested_model == config.BACKBONE_MODEL
    assert response.response_model == config.BACKBONE_MODEL
    assert response.request_id == "req-qwen-1"
    assert response.text == "完成"
    assert response.finish_reason == "tool_calls"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 15
    assert response.tool_calls[0].name == "recipe_lookup"
    assert response.tool_calls[0].arguments_json == '{"dish":"汤"}'
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert calls[0]["top_p"] == 0.9
    assert calls[0]["seed"] == 7

    records = config.USAGE_LOG.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    logged = json.loads(records[0])
    assert logged["provider"] == "qwen"
    assert logged["request_id"] == "req-qwen-1"
    assert logged["requested_model"] == config.BACKBONE_MODEL
    assert logged["response_model"] == config.BACKBONE_MODEL
    assert logged["total_tokens"] == 15
    assert logged["tool_call_count"] == 1


def test_openai_reasoning_is_committed_without_persisting_chain_of_thought(
    tmp_path, monkeypatch
):
    reasoning = "private chain of thought that must never be persisted"

    def create(**_kwargs):
        message = SimpleNamespace(
            content="",
            reasoning_content=reasoning,
            tool_calls=[],
        )
        return SimpleNamespace(
            id="req-reasoning-metadata",
            model=config.BACKBONE_MODEL,
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=9,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
            ),
        )

    llm._clients["qwen"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    response = chat(
        "qwen",
        [{"role": "user", "content": "test"}],
        record_usage=True,
    )

    encoded = reasoning.encode("utf-8")
    assert response.reasoning_present is True
    assert response.reasoning_tokens == 7
    assert response.reasoning_bytes == len(encoded)
    assert response.reasoning_sha256 == sha256_bytes(encoded)
    serialized = canonical_json_bytes(response.model_dump(mode="json"))
    usage_log = config.USAGE_LOG.read_bytes()
    assert reasoning.encode("utf-8") not in serialized
    assert reasoning.encode("utf-8") not in usage_log
    assert b"reasoning_content" not in serialized
    assert json.loads(usage_log)["reasoning_sha256"] == sha256_bytes(encoded)


def test_llm_response_without_reasoning_keeps_legacy_canonical_shape():
    response = LLMResponse(
        provider="qwen",
        endpoint=config.PROVIDER_ENDPOINTS["qwen"],
        requested_model=config.BACKBONE_MODEL,
        response_model=config.BACKBONE_MODEL,
        request_id="req-no-reasoning",
        text="done",
        usage={"input_tokens": 1, "output_tokens": 1},
        finish_reason="stop",
        latency_ms=1,
    )

    dumped = response.model_dump(mode="json")
    assert not any(key.startswith("reasoning_") for key in dumped)


def test_openai_response_rejects_multiple_choices(monkeypatch):
    choice = SimpleNamespace(
        message=SimpleNamespace(content="{}", tool_calls=[]),
        finish_reason="stop",
    )
    llm._clients["qwen"] = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    id="req-multiple-choices",
                    model=config.BACKBONE_MODEL,
                    choices=[choice, choice],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )
            )
        )
    )

    with pytest.raises(llm.LLMContractError, match="exactly one choice"):
        chat(
            "qwen",
            [{"role": "user", "content": "test"}],
            record_usage=False,
        )


def test_qwen_client_disables_sdk_level_retries(monkeypatch):
    constructor_calls: list[dict[str, object]] = []
    fake_client = object()

    def fake_openai(**kwargs):
        constructor_calls.append(kwargs)
        return fake_client

    monkeypatch.setenv(config.PROVIDER_API_KEY_ENV["qwen"], "test-only-key")
    monkeypatch.setattr(llm, "OpenAI", fake_openai)

    assert llm._client("qwen") is fake_client
    assert constructor_calls == [
        {
            "base_url": config.PROVIDER_ENDPOINTS["qwen"],
            "api_key": "test-only-key",
            "max_retries": 0,
        }
    ]


def test_kimi_client_disables_sdk_level_retries(monkeypatch):
    constructor_calls: list[dict[str, object]] = []
    fake_client = object()

    def fake_openai(**kwargs):
        constructor_calls.append(kwargs)
        return fake_client

    monkeypatch.setenv(config.PROVIDER_API_KEY_ENV["kimi"], "test-only-key")
    monkeypatch.setattr(llm, "OpenAI", fake_openai)

    assert llm._client("kimi") is fake_client
    assert constructor_calls == [
        {
            "base_url": config.PROVIDER_ENDPOINTS["kimi"],
            "api_key": "test-only-key",
            "max_retries": 0,
        }
    ]


def test_gemini_client_disables_sdk_level_retries(monkeypatch):
    constructor_calls: list[dict[str, object]] = []
    fake_client = object()

    def fake_openai(**kwargs):
        constructor_calls.append(kwargs)
        return fake_client

    monkeypatch.setenv(config.PROVIDER_API_KEY_ENV["gemini"], "test-only-key")
    monkeypatch.setattr(llm, "OpenAI", fake_openai)

    assert llm._client("gemini") is fake_client
    assert constructor_calls == [
        {
            "base_url": config.PROVIDER_ENDPOINTS["gemini"],
            "api_key": "test-only-key",
            "max_retries": 0,
        }
    ]


def test_chat_can_delegate_usage_persistence_to_parent_runner(monkeypatch):
    response = LLMResponse(
        provider="qwen",
        endpoint=config.PROVIDER_ENDPOINTS["qwen"],
        requested_model=config.BACKBONE_MODEL,
        response_model=config.BACKBONE_MODEL,
        request_id="req-parent-owned-usage",
        text="{}",
        usage={"input_tokens": 4, "output_tokens": 2},
        finish_reason="stop",
        latency_ms=0,
    )
    monkeypatch.setattr(llm, "_chat_once", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(
        llm,
        "_log_usage",
        lambda _response: (_ for _ in ()).throw(
            AssertionError("parent-owned usage must not touch the worker filesystem")
        ),
    )

    assert (
        chat(
            "qwen",
            [{"role": "user", "content": "test"}],
            record_usage=False,
        ).request_id
        == "req-parent-owned-usage"
    )


def test_chat_wraps_provider_timeout_for_callers(monkeypatch):
    timeout = APITimeoutError(httpx.Request("POST", "https://example.invalid"))
    monkeypatch.setattr(
        llm,
        "_chat_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(timeout),
    )

    with pytest.raises(llm.LLMTimeoutError, match="exceeded its timeout"):
        chat(
            "qwen",
            [{"role": "user", "content": "test"}],
            max_attempts=1,
            record_usage=False,
        )


def test_openai_compatible_tool_schema_and_forced_choice_are_forwarded(
    tmp_path, monkeypatch
):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        tool_call = SimpleNamespace(
            id="submission-1",
            function=SimpleNamespace(
                name="submit_payload",
                arguments='{\n  "value": 1\n}',
            ),
        )
        return SimpleNamespace(
            id="req-structured-1",
            model=config.BACKBONE_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None, tool_calls=[tool_call]),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        )

    llm._clients["qwen"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")
    tool = {
        "type": "function",
        "function": {
            "name": "submit_payload",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    choice = {
        "type": "function",
        "function": {"name": "submit_payload"},
    }

    response = chat(
        "qwen",
        [{"role": "user", "content": "submit"}],
        tools=[tool],
        tool_choice=choice,
        parallel_tool_calls=False,
    )

    assert response.finish_reason == "tool_calls"
    assert response.text == ""
    assert response.tool_calls[0].arguments_json == '{\n  "value": 1\n}'
    assert calls[0]["tools"] == [tool]
    assert calls[0]["tool_choice"] == choice
    assert calls[0]["parallel_tool_calls"] is False
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert "response_format" not in calls[0]


def test_openai_timeout_and_images_target_plain_user_before_tool_history(
    tmp_path, monkeypatch, red_image
):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="req-native-history",
            model=config.BACKBONE_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )

    llm._clients["qwen"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")
    messages = [
        {"role": "system", "content": "Use the tools."},
        {"role": "user", "content": "Inspect this image."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "recipe_lookup",
                        "arguments": '{"dish":"soup"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"status":"success"}',
        },
    ]
    tool = {
        "type": "function",
        "function": {
            "name": "recipe_lookup",
            "description": "Look up a recipe.",
            "parameters": {"type": "object"},
        },
    }

    response = chat(
        "qwen",
        messages,
        images=[red_image],
        tools=[tool],
        tool_choice="auto",
        parallel_tool_calls=False,
        timeout_seconds=12.5,
    )

    assert response.text == "done"
    assert calls[0]["timeout"] == 12.5
    image_part, text_part = calls[0]["messages"][1]["content"]
    assert image_part["type"] == "image_url"
    assert text_part == {"type": "text", "text": "Inspect this image."}
    assert calls[0]["messages"][2:] == messages[2:]


def test_deepseek_identity_and_default_thinking_guard(tmp_path, monkeypatch):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="req-deepseek-1",
            model=config.LABEL_TEXT_REVIEW_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    llm._clients["deepseek"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    response = chat("deepseek", [{"role": "user", "content": "test"}])

    assert response.provider == "deepseek"
    assert response.usage.total_tokens == 0
    assert calls[0]["extra_body"] == {"enable_thinking": False}


def test_qwen37_feedback_sends_strict_json_schema_and_thinking_contract(
    tmp_path, monkeypatch, red_image
):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="req-qwen37-1",
            model=config.FEEDBACK_JUDGE_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )

    llm._clients["qwen"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    response_format = llm.LLMJsonSchemaResponseFormat.model_validate(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "feedback_test_v1",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"schema_version": {"type": "integer"}},
                    "required": ["schema_version"],
                },
            },
        },
        strict=True,
    )
    response = chat(
        "qwen",
        [{"role": "user", "content": "feedback"}],
        model=config.FEEDBACK_JUDGE_MODEL,
        images=[red_image],
        temperature=config.FEEDBACK_JUDGE_TEMPERATURE,
        top_p=config.FEEDBACK_JUDGE_TOP_P,
        thinking=config.FEEDBACK_JUDGE_THINKING,
        thinking_budget=config.FEEDBACK_JUDGE_THINKING_BUDGET,
        max_tokens=config.FEEDBACK_JUDGE_MAX_TOKENS,
        max_completion_tokens=config.FEEDBACK_JUDGE_MAX_COMPLETION_TOKENS,
        response_format=response_format,
        max_attempts=1,
        timeout_seconds=config.FEEDBACK_JUDGE_TIMEOUT_SECONDS,
    )

    assert response.provider == "qwen"
    assert response.response_model == config.FEEDBACK_JUDGE_MODEL
    assert calls[0]["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 2048,
    }
    assert "max_tokens" not in calls[0]
    assert calls[0]["max_completion_tokens"] == 4096
    assert calls[0]["timeout"] == 600
    assert "temperature" not in calls[0]
    assert "top_p" not in calls[0]
    assert "seed" not in calls[0]
    image_part, text_part = calls[0]["messages"][-1]["content"]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert calls[0]["response_format"] == response_format.model_dump(
        mode="json", by_alias=True
    )
    assert calls[0]["stream"] is False
    assert "stream_options" not in calls[0]
    assert text_part == {"type": "text", "text": "feedback"}


@pytest.mark.parametrize(
    ("usage", "response_model"),
    (
        (None, config.FEEDBACK_JUDGE_MODEL),
        (
            SimpleNamespace(prompt_tokens=0, completion_tokens=0),
            config.FEEDBACK_JUDGE_MODEL,
        ),
        (
            SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=19),
            config.FEEDBACK_JUDGE_MODEL,
        ),
        (SimpleNamespace(prompt_tokens=11, completion_tokens=7), "qwen-wrong-snapshot"),
    ),
)
def test_qwen37_feedback_rejects_missing_or_invalid_provider_usage(
    tmp_path, monkeypatch, usage, response_model
):
    def create(**_kwargs):
        return SimpleNamespace(
            id="req-qwen37-invalid-usage",
            model=response_model,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    llm._clients["qwen"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")
    response_format = llm.LLMJsonSchemaResponseFormat.model_validate(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "feedback_usage_test_v1",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"schema_version": {"type": "integer"}},
                    "required": ["schema_version"],
                },
            },
        },
        strict=True,
    )

    with pytest.raises(llm.LLMContractError, match="Qwen3.7 Feedback"):
        chat(
            "qwen",
            [{"role": "user", "content": "feedback"}],
            model=config.FEEDBACK_JUDGE_MODEL,
            thinking=True,
            thinking_budget=2048,
            max_tokens=None,
            max_completion_tokens=4096,
            response_format=response_format,
            max_attempts=1,
            timeout_seconds=600,
        )


def test_historical_kimi_feedback_adapter_remains_available(tmp_path, monkeypatch):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="req-kimi-historical-1",
            model=config.LEGACY_KIMI_FEEDBACK_JUDGE_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )

    llm._clients["kimi"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    response = chat(
        "kimi",
        [{"role": "user", "content": "historical feedback"}],
        model=config.LEGACY_KIMI_FEEDBACK_JUDGE_MODEL,
        temperature=config.LEGACY_KIMI_FEEDBACK_JUDGE_TEMPERATURE,
        top_p=config.LEGACY_KIMI_FEEDBACK_JUDGE_TOP_P,
        thinking=config.LEGACY_KIMI_FEEDBACK_JUDGE_THINKING,
        max_attempts=1,
    )

    assert response.provider == "kimi"
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert calls[0]["temperature"] == 0.6
    assert calls[0]["top_p"] == 0.95
    assert "response_format" not in calls[0]


def test_gemini_36_judge_sends_json_mode_without_deprecated_sampling(
    tmp_path, monkeypatch, red_image
):
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="req-gemini-1",
            model=config.PORTFOLIO_JUDGE_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
        )

    llm._clients["gemini"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    response = chat(
        "gemini",
        [{"role": "user", "content": "judge"}],
        model=config.PORTFOLIO_JUDGE_MODEL,
        images=[red_image],
        temperature=None,
        top_p=None,
        thinking=False,
        json_mode=True,
        max_attempts=1,
    )

    assert response.provider == "gemini"
    assert response.response_model == config.PORTFOLIO_JUDGE_MODEL
    assert "extra_body" not in calls[0]
    assert "temperature" not in calls[0]
    assert "top_p" not in calls[0]
    assert "seed" not in calls[0]
    assert calls[0]["response_format"] == {"type": "json_object"}
    image_part, text_part = calls[0]["messages"][-1]["content"]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert text_part == {"type": "text", "text": "judge"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thinking": True},
        {"temperature": 0.6},
        {"top_p": 0.95},
        {"seed": 1},
        {"reasoning_effort": "max"},
    ],
)
def test_gemini_36_rejects_unavailable_or_deprecated_controls(kwargs):
    with pytest.raises(ValueError):
        chat(
            "gemini",
            [{"role": "user", "content": "judge"}],
            model=config.PORTFOLIO_JUDGE_MODEL,
            max_attempts=1,
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"thinking": True},
        {
            "thinking": False,
            "thinking_budget": 1,
            "reasoning_effort": "max",
            "temperature": 0.6,
            "top_p": 0.95,
        },
        {
            "thinking": False,
            "temperature": 0.0,
            "top_p": 0.95,
        },
        {
            "thinking": False,
            "temperature": 0.6,
            "top_p": 1.0,
        },
        {
            "thinking": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": 1,
        },
        {
            "thinking": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "json_mode": True,
        },
        {
            "thinking": False,
            "thinking_budget": 1,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    ],
)
def test_kimi_k26_rejects_unsupported_contract_before_client(kwargs):
    with pytest.raises(ValueError):
        chat(
            "kimi",
            [{"role": "user", "content": "feedback"}],
            model=config.LEGACY_KIMI_FEEDBACK_JUDGE_MODEL,
            max_attempts=1,
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_format": None},
        {"json_mode": True},
        {"thinking": False},
        {"thinking_budget": None},
        {"thinking_budget": 1024},
        {"max_tokens": 2048},
        {"max_completion_tokens": None},
        {"max_completion_tokens": 2048},
        {"timeout_seconds": None},
        {"json_mode": True, "temperature": 0.2},
        {"json_mode": True, "top_p": 0.95},
        {"json_mode": True, "seed": 1},
        {"json_mode": True, "reasoning_effort": "max"},
    ],
)
def test_qwen37_feedback_rejects_unfrozen_contract_before_client(kwargs):
    controls = {
        "response_format": llm.LLMJsonSchemaResponseFormat.model_validate(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "feedback_test_v1",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            strict=True,
        ),
        "thinking": True,
        "thinking_budget": 2048,
        "max_tokens": None,
        "max_completion_tokens": 4096,
        "timeout_seconds": 600,
    }
    controls.update(kwargs)
    with pytest.raises(ValueError):
        chat(
            "qwen",
            [{"role": "user", "content": "feedback"}],
            model=config.FEEDBACK_JUDGE_MODEL,
            max_attempts=1,
            **controls,
        )


def test_thinking_budget_is_rejected_outside_kimi_adapter():
    with pytest.raises(ValueError, match="only supported by the Kimi"):
        chat(
            "qwen",
            [{"role": "user", "content": "test"}],
            thinking_budget=1,
            max_attempts=1,
        )


def test_claude_response_preserves_text_tool_and_actual_identity(tmp_path, monkeypatch):
    content = [
        SimpleNamespace(type="text", text="先查工具"),
        SimpleNamespace(
            type="tool_use",
            id="toolu-1",
            name="encyclopedia_lookup",
            input={"entity": "银杏"},
        ),
    ]
    raw = SimpleNamespace(
        id="req-claude-1",
        model=config.REFINER_MODEL,
        content=content,
        usage=SimpleNamespace(input_tokens=8, output_tokens=5),
        stop_reason="tool_use",
    )
    llm._clients["claude"] = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **_: raw)
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    response = chat("claude", [{"role": "user", "content": "test"}])

    assert response.text == "先查工具"
    assert response.tool_calls[0].call_id == "toolu-1"
    assert response.tool_calls[0].arguments_json == '{"entity":"银杏"}'
    assert response.finish_reason == "tool_use"


def test_response_model_mismatch_fails_closed(tmp_path, monkeypatch):
    raw = SimpleNamespace(
        id="req-mismatch",
        model="qwen-unexpected",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="x", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    llm._clients["qwen"] = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: raw))
    )
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")

    with pytest.raises(ValidationError, match="does not match requested model"):
        chat("qwen", [{"role": "user", "content": "test"}])
    assert not config.USAGE_LOG.exists()


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("deepseek", "glm-4"),
        ("qwen", "deepseek-v4"),
        ("kimi", "kimi/kimi-k3"),
        ("gemini", "qwen3-vl-flash"),
        ("claude", "qwen3-vl-plus"),
    ],
)
def test_provider_model_family_mismatch_fails_before_client(provider, model):
    with pytest.raises(ValueError, match="does not belong"):
        chat(provider, [{"role": "user", "content": "test"}], model=model)


def test_invalid_request_contract_fails_before_client():
    with pytest.raises(ValueError, match="messages"):
        chat("qwen", [])
    with pytest.raises(ValueError, match="max_attempts"):
        chat("qwen", [{"role": "user", "content": "test"}], max_attempts=0)
    with pytest.raises(ValueError, match="max_tokens"):
        chat("qwen", [{"role": "user", "content": "test"}], max_tokens=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        chat("qwen", [{"role": "user", "content": "test"}], timeout_seconds=0)
    with pytest.raises(ValueError, match="top_p"):
        chat("qwen", [{"role": "user", "content": "test"}], top_p=0)
    with pytest.raises(ValueError, match="seed"):
        chat("qwen", [{"role": "user", "content": "test"}], seed=-1)
    with pytest.raises(ValueError, match="seed control"):
        chat("claude", [{"role": "user", "content": "test"}], seed=1)
    with pytest.raises(ValueError, match="non-empty"):
        chat("qwen", [{"role": "user", "content": "test"}], tools=[])
    with pytest.raises(ValueError, match="mutually exclusive"):
        chat(
            "qwen",
            [{"role": "user", "content": "test"}],
            json_mode=True,
            tools=[{"type": "function"}],
        )
    with pytest.raises(ValueError, match="requires tools"):
        chat(
            "qwen",
            [{"role": "user", "content": "test"}],
            tool_choice="required",
        )
    with pytest.raises(ValueError, match="parallel_tool_calls requires tools"):
        chat(
            "qwen",
            [{"role": "user", "content": "test"}],
            parallel_tool_calls=False,
        )
    with pytest.raises(ValueError, match="claude adapter"):
        chat(
            "claude",
            [{"role": "user", "content": "test"}],
            tools=[{"type": "function"}],
        )
    with pytest.raises(ValueError, match="视觉"):
        chat(
            "deepseek",
            [{"role": "user", "content": "test"}],
            images=["not-read-before-validation.jpg"],
        )


def test_claude_code_provider_guard():
    with pytest.raises(RuntimeError, match="Claude Code"):
        chat("claude_code", [{"role": "user", "content": "hi"}])
