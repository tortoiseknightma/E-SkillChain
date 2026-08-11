from __future__ import annotations

import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from skillchain import config, llm
from skillchain.evaluation.evaluator_isolation import (
    make_evaluator_isolation_lock,
    make_feedback_evaluator_identity,
    make_final_evaluator_identity,
)
from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    parse_visual_feedback_output_v3,
)
from skillchain.evaluation.feedback_runtime import (
    FeedbackEvaluationResult,
    FeedbackRuntimeError,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V2,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7,
    VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
    _make_result,
    load_feedback_evaluation_result,
    redact_feedback_result_for_creator_privacy,
    run_visual_feedback,
    visual_feedback_json_schema_v1,
    visual_feedback_response_format_v1,
    write_feedback_evaluation_result,
)
from skillchain.evaluation.packets import (
    EvaluationImage,
    FeedbackGCSComponentBitsV1,
    FeedbackGCSContractV1,
    FeedbackGCSDiagnosticsV1,
    FeedbackPacket,
    FeedbackPacketV3,
    RubricSnapshot,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
)
from skillchain.llm import LLMResponse, LLMUsage
from skillchain.schemas import ConversationTurn
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes
from skillchain.evaluation.visual_runtime import EvaluatorImageLoadError


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


def _packet() -> FeedbackPacket:
    image_bytes = b"visual-feedback-image"
    rubric_text = "Diagnose the visible response against the requested task."
    rubric = RubricSnapshot(
        rubric_id="feedback-rubric-v2",
        rubric_version="2",
        content=rubric_text,
        content_sha256=hashlib.sha256(rubric_text.encode()).hexdigest(),
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "packet_kind": "feedback",
        "cache_namespace": "feedback-evaluator-v2",
        "query_id": "query-001",
        "turns": (ConversationTurn(role="user", content="识别图中商品"),),
        "image": EvaluationImage(
            mime_type="image/png",
            sha256=hashlib.sha256(image_bytes).hexdigest(),
        ),
        "canonical_capability": "product.exact_match",
        "acceptable_capabilities": ("product.exact_match",),
        "response_text": "这是可见回答",
        "cards": (),
        "tool_evidence": (),
        "tool_trace": (),
        "rubric": rubric,
    }
    return FeedbackPacket.model_validate(
        {**payload, "packet_sha256": _hash_payload(payload)},
        strict=True,
    )


def _packet_v3() -> FeedbackPacketV3:
    legacy = _packet()
    payload = {
        **legacy.model_dump(
            mode="python",
            exclude={"schema_version", "cache_namespace", "packet_sha256"},
        ),
        "schema_version": 3,
        "cache_namespace": "feedback-evaluator-v10",
        "gcs_diagnostics": FeedbackGCSDiagnosticsV1(
            answer_mode="fallback",
            gcs=0,
            components=FeedbackGCSComponentBitsV1(
                route_acceptable=1,
                no_hard_error=1,
                tool_contract_pass=1,
                evidence_grounded=0,
                output_contract_pass=1,
            ),
            reason_codes=("material_fact_uncited",),
        ),
        "gcs_contract": FeedbackGCSContractV1(
            policy_sha256="a" * 64,
            required_sections=("answer", "evidence", "uncertainty"),
            fallback_markers=("not enough evidence",),
            preferred_fallback_marker="not enough evidence",
            card_requirement="forbidden",
            legal_tool_sequences=(
                ("encyclopedia_lookup",),
                ("object_detect", "encyclopedia_lookup"),
            ),
        ),
        "canonical_capability": "knowledge.visual_encyclopedia",
        "acceptable_capabilities": ("knowledge.visual_encyclopedia",),
    }
    return FeedbackPacketV3.model_validate(
        {**payload, "packet_sha256": _hash_payload(payload)}, strict=True
    )


def _lock():
    feedback = make_feedback_evaluator_identity(
        provider=config.FEEDBACK_JUDGE_PROVIDER,
        model=config.FEEDBACK_JUDGE_MODEL,
        model_family=config.FEEDBACK_JUDGE_MODEL,
        endpoint=config.PROVIDER_ENDPOINTS[config.FEEDBACK_JUDGE_PROVIDER],
    )
    final = make_final_evaluator_identity(
        provider=config.PORTFOLIO_JUDGE_PROVIDER,
        model=config.PORTFOLIO_JUDGE_MODEL,
        model_family="gemini-3.6-flash",
        endpoint=config.PROVIDER_ENDPOINTS["gemini"],
    )
    return make_evaluator_isolation_lock(feedback, final)


@pytest.mark.parametrize("failure_kind", ["copied_schema_version", "five_evidence"])
def test_parser_v3_stays_strict_after_prompt_v5(failure_kind: str) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "summary": "Strict parser regression.",
        "rule_violations": [],
        "ideal_response_gaps": [],
        "skill_suggestions": [],
    }
    if failure_kind == "copied_schema_version":
        payload["schema_version"] = 2
    else:
        payload["rule_violations"] = [
            {
                "dimension": "CQ",
                "severity": "medium",
                "grounded_in_image": False,
                "description": "The response omitted a requested detail.",
                "evidence": [f"evidence-{index}" for index in range(5)],
            }
        ]
    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v3(
            canonical_json_bytes(payload).decode("utf-8").strip()
        )


def test_qwen_provider_schema_has_exact_visual_feedback_shape() -> None:
    schema = visual_feedback_json_schema_v1()
    expected_root = {
        "schema_version",
        "summary",
        "rule_violations",
        "ideal_response_gaps",
        "skill_suggestions",
    }
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == expected_root
    assert set(schema["required"]) == expected_root
    finding = schema["properties"]["rule_violations"]["items"]
    expected_finding = {
        "dimension",
        "severity",
        "grounded_in_image",
        "description",
        "evidence",
    }
    assert finding["additionalProperties"] is False
    assert set(finding["properties"]) == expected_finding
    assert set(finding["required"]) == expected_finding


def test_visual_feedback_runner_sends_real_image_url_array(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet()
    calls: list[tuple[str, list[dict], dict]] = []
    catalog_sha256 = "a" * 64
    authorization_file_sha256 = "b" * 64
    receipt_file_sha256 = "c" * 64
    receipt_sha256 = "d" * 64
    runtime = SimpleNamespace(
        authorization=SimpleNamespace(authorization_id="portfolio-visual-feedback-v2"),
        authorization_file_sha256=authorization_file_sha256,
        receipt_file_sha256=receipt_file_sha256,
        receipt=SimpleNamespace(receipt_sha256=receipt_sha256),
        catalog=SimpleNamespace(catalog_sha256=catalog_sha256),
    )
    preflight_calls: list[tuple[object, str, object]] = []

    def fake_load(remote_runtime, *, processor, image, query_id):
        assert query_id == packet.query_id
        preflight_calls.append((remote_runtime, processor, image))
        return remote_runtime, b"visual-feedback-image"

    def fake_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        payload = {
            "schema_version": 1,
            "summary": "图像与回答不一致。",
            "rule_violations": [
                {
                    "dimension": "TCR",
                    "severity": "high",
                    "grounded_in_image": True,
                    "description": "未核对图片主体。",
                    "evidence": ["图片主体与回答不一致。"],
                }
            ],
            "ideal_response_gaps": [],
            "skill_suggestions": ["先核对视觉主体再回答。"],
        }
        payload["summary"] = " leading and trailing "
        payload["rule_violations"][0]["description"] = " finding text "
        return LLMResponse(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            requested_model=config.FEEDBACK_JUDGE_MODEL,
            response_model=config.FEEDBACK_JUDGE_MODEL,
            request_id="req-feedback-001",
            text=canonical_json_bytes(payload).decode().strip(),
            usage=LLMUsage(input_tokens=20, output_tokens=8),
            finish_reason="stop",
            latency_ms=3,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(
        "skillchain.evaluation.feedback_runtime.load_verified_evaluator_image",
        fake_load,
    )
    result = run_visual_feedback(
        packet,
        _lock(),
        remote_runtime=runtime,
        max_completion_tokens=4096,
        record_usage=False,
    )

    assert result.query_id == packet.query_id
    assert result.image_sha256 == packet.image.sha256
    assert result.asset_catalog_sha256 == catalog_sha256
    assert result.remote_authorization_id == "portfolio-visual-feedback-v2"
    assert result.remote_authorization_file_sha256 == authorization_file_sha256
    assert result.remote_receipt_file_sha256 == receipt_file_sha256
    assert result.remote_receipt_sha256 == receipt_sha256
    assert result.model == "qwen3.8-max"
    assert result.schema_version == 5
    assert result.cache_namespace == "feedback-evaluator-v11"
    assert result.parser_policy_version == "visual-feedback-free-text-trim-v3"
    assert (
        result.parser_policy_sha256
        == "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
    )
    assert result.prompt_policy_version == VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5
    assert result.prompt_policy_sha256 == VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5
    assert (
        result.transport_policy_version == VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V6
    )
    assert result.transport_policy_sha256 == VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V6
    assert result.requested_response_format == "json_schema"
    assert result.requested_json_schema_sha256 == VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1
    assert result.requested_thinking is True
    assert result.requested_thinking_budget == 2048
    assert result.requested_timeout_seconds == 600
    assert result.requested_temperature is None
    assert result.requested_top_p is None
    assert result.status == "parsed"
    assert result.parsed_feedback is not None
    assert result.parsed_feedback.summary == "leading and trailing"
    assert result.parsed_feedback.rule_violations[0].description == "finding text"
    assert result.parsed_feedback.rule_violations[0].grounded_in_image is True
    assert result.raw_response_text is not None
    assert result.raw_response_text.startswith("{")
    assert result.formal_eligible is False
    assert result.result_sha256
    assert preflight_calls == [(runtime, "dashscope-qwen38-feedback", packet.image)]
    provider, messages, kwargs = calls[0]
    assert provider == "qwen"
    assert result.wire_sha256 == sha256_bytes(
        canonical_json_bytes(
            {
                "messages": messages,
                "response_format": visual_feedback_response_format_v1().model_dump(
                    mode="json", by_alias=True
                ),
                "stream": False,
                "invocation_controls": {
                    "enable_thinking": True,
                    "thinking_budget": 2048,
                    "max_tokens": "omitted",
                    "max_completion_tokens": 4096,
                    "timeout_seconds": 600,
                    "temperature": None,
                    "top_p": None,
                },
            }
        )
    )
    assert "Inspect the provided image" in messages[0]["content"]
    assert "image-grounded rule violations" in messages[0]["content"]
    assert "do not copy or echo output_contract itself" in (
        messages[0]["content"].casefold()
    )
    assert "schema_version is mandatory" in messages[0]["content"]
    assert "one to four strings, never five" in messages[0]["content"]
    image_part, text_part = messages[1]["content"]
    assert image_part["type"] == "image_url"
    data_url = image_part["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == b"visual-feedback-image"
    assert "content_base64" not in text_part["text"]
    assert "output_contract" in text_part["text"]
    visible = json.loads(text_part["text"])
    assert visible["response_identity"]["mandatory_response_schema_version"] == {
        "json_type": "integer",
        "literal": 1,
    }
    assert visible["output_contract"]["top_level_fields_exactly_once"] == [
        "schema_version",
        "summary",
        "rule_violations",
        "ideal_response_gaps",
        "skill_suggestions",
    ]
    assert visible["output_contract"]["skill_suggestions_item_schema"] == {
        "nonblank_after_trim": True,
        "type": "string",
    }
    assert kwargs["thinking"] is True
    assert kwargs["thinking_budget"] == 2048
    assert kwargs["max_tokens"] is None
    assert kwargs["max_completion_tokens"] == 4096
    assert kwargs["timeout_seconds"] == 600
    assert kwargs["temperature"] is None
    assert kwargs["top_p"] is None
    assert kwargs["json_mode"] is False
    assert kwargs["response_format"] == visual_feedback_response_format_v1()
    assert kwargs["max_attempts"] == 1
    assert kwargs["record_usage"] is False
    assert result.reasoning_present is False
    assert result.reasoning_tokens is None
    assert result.reasoning_bytes == 0
    assert result.reasoning_sha256 is None
    receipt_path = write_feedback_evaluation_result(
        tmp_path / "feedback-result.json",
        result,
    )
    assert b"content_base64" not in receipt_path.read_bytes()
    assert b"data:image" not in receipt_path.read_bytes()
    assert (
        load_feedback_evaluation_result(
            receipt_path,
            expected_result_sha256=result.result_sha256,
        )
        == result
    )

    tampered = result.model_dump(mode="json")
    tampered["parsed_feedback"]["summary"] = "coordinated tamper"
    with pytest.raises(
        ValueError,
        match="differs from raw response",
    ):
        FeedbackEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )

    parser_drift = result.model_dump(mode="json")
    parser_drift["parser_policy_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="parser policy identity mismatch"):
        FeedbackEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(parser_drift)),
            strict=True,
        )

    prompt_drift = result.model_dump(mode="json")
    prompt_drift["prompt_policy_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="prompt policy identity mismatch"):
        FeedbackEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(prompt_drift)),
            strict=True,
        )

    transport_drift = result.model_dump(mode="json")
    transport_drift["transport_policy_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="transport policy identity mismatch"):
        FeedbackEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(transport_drift)),
            strict=True,
        )

    schema_drift = result.model_dump(mode="json")
    schema_drift["requested_json_schema_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="transport policy identity mismatch"):
        FeedbackEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(schema_drift)),
            strict=True,
        )


@pytest.mark.parametrize(
    ("suggestion", "expected_status"),
    [
        ("[policy_compatible] Preserve the exact fallback marker.", "parsed"),
        ("Preserve the exact fallback marker.", "parse_error"),
    ],
)
def test_round2_feedback_v11_binds_gcs_prompt_and_policy_labels(
    monkeypatch,
    suggestion: str,
    expected_status: str,
) -> None:
    packet = _packet_v3()
    runtime = SimpleNamespace(
        authorization=SimpleNamespace(authorization_id="feedback-round2-v1"),
        authorization_file_sha256="b" * 64,
        receipt_file_sha256="c" * 64,
        receipt=SimpleNamespace(receipt_sha256="d" * 64),
        catalog=SimpleNamespace(catalog_sha256="a" * 64),
    )
    observed_messages: list[list[dict]] = []

    def fake_chat(_provider, messages, **_kwargs):
        observed_messages.append(messages)
        payload = {
            "schema_version": 1,
            "summary": "The fallback contract must remain exact.",
            "rule_violations": [],
            "ideal_response_gaps": [],
            "skill_suggestions": [suggestion],
        }
        return LLMResponse(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            requested_model=config.FEEDBACK_JUDGE_MODEL,
            response_model=config.FEEDBACK_JUDGE_MODEL,
            request_id="req-feedback-round2",
            text=canonical_json_bytes(payload).decode().strip(),
            usage=LLMUsage(input_tokens=20, output_tokens=8),
            finish_reason="stop",
            latency_ms=3,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(
        "skillchain.evaluation.feedback_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, b"visual-feedback-image"),
    )

    result = run_visual_feedback(
        packet,
        _lock(),
        remote_runtime=runtime,
        max_completion_tokens=4096,
        record_usage=False,
    )

    assert result.schema_version == 5
    assert result.cache_namespace == "feedback-evaluator-v11"
    assert result.model == "qwen3.8-max"
    assert result.prompt_policy_version == VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6
    assert result.prompt_policy_sha256 == VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6
    assert result.status == expected_status
    visible = json.loads(observed_messages[0][1]["content"][1]["text"])
    assert visible["gcs_diagnostics"]["answer_mode"] == "fallback"
    assert visible["gcs_contract"]["preferred_fallback_marker"] == (
        "not enough evidence"
    )
    assert visible["output_contract"]["skill_suggestions_item_schema"][
        "required_prefix_exactly_one_of"
    ] == [
        "[policy_compatible] ",
        "[requires_new_evidence] ",
        "[rejected] ",
    ]


def test_fresh_v3_feedback_defaults_to_result6_cache_v12_and_wire6144(
    monkeypatch,
) -> None:
    packet = _packet_v3()
    runtime = SimpleNamespace(
        authorization=SimpleNamespace(authorization_id="feedback-fresh-v3"),
        authorization_file_sha256="b" * 64,
        receipt_file_sha256="c" * 64,
        receipt=SimpleNamespace(receipt_sha256="d" * 64),
        catalog=SimpleNamespace(catalog_sha256="a" * 64),
    )
    observed: list[dict] = []

    def fake_chat(_provider, _messages, **kwargs):
        observed.append(kwargs)
        payload = {
            "schema_version": 1,
            "summary": "Fresh-v3 structured Feedback.",
            "rule_violations": [],
            "ideal_response_gaps": [],
            "skill_suggestions": [
                "[policy_compatible] Preserve the exact fallback marker."
            ],
        }
        return LLMResponse(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            requested_model=config.FEEDBACK_JUDGE_MODEL,
            response_model=config.FEEDBACK_JUDGE_MODEL,
            request_id="req-feedback-fresh-v3",
            text=canonical_json_bytes(payload).decode().strip(),
            usage=LLMUsage(input_tokens=20, output_tokens=8),
            finish_reason="stop",
            latency_ms=3,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(
        "skillchain.evaluation.feedback_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, b"visual-feedback-image"),
    )
    result = run_visual_feedback(
        packet,
        _lock(),
        remote_runtime=runtime,
        record_usage=False,
    )
    assert result.schema_version == 6
    assert result.cache_namespace == "feedback-evaluator-v12"
    assert result.max_completion_tokens == 6144
    assert result.transport_policy_version == VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
    assert result.transport_policy_sha256 == VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    assert observed[0]["max_completion_tokens"] == 6144


def test_round2_creator_privacy_redaction_is_terminal_and_rebuildable(
    tmp_path,
) -> None:
    packet = _packet_v3()
    raw_response_text = (
        canonical_json_bytes(
            {
                "schema_version": 1,
                "summary": "A structurally valid provider response.",
                "rule_violations": [],
                "ideal_response_gaps": [],
                "skill_suggestions": [
                    "[policy_compatible] Preserve the exact fallback marker."
                ],
            }
        )
        .decode("utf-8")
        .strip()
    )
    parsed = parse_visual_feedback_output_v3(raw_response_text)
    result = _make_result(
        schema_version=4,
        cache_namespace="feedback-evaluator-v10",
        parser_policy_version="visual-feedback-free-text-trim-v3",
        parser_policy_sha256=(
            "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
        ),
        prompt_policy_version=VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V6,
        prompt_policy_sha256=VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V6,
        transport_policy_version=VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V5,
        transport_policy_sha256=VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V5,
        requested_response_format="json_schema",
        requested_json_schema_sha256=VISUAL_FEEDBACK_JSON_SCHEMA_SHA256_V1,
        requested_thinking=True,
        requested_thinking_budget=2048,
        requested_timeout_seconds=600,
        requested_temperature=None,
        requested_top_p=None,
        query_id=packet.query_id,
        packet_sha256=packet.packet_sha256,
        prompt_sha256="1" * 64,
        image_sha256=packet.image.sha256,
        wire_sha256="2" * 64,
        asset_catalog_sha256="3" * 64,
        remote_authorization_id="feedback-round2-v1",
        remote_authorization_file_sha256="4" * 64,
        remote_receipt_file_sha256="5" * 64,
        remote_receipt_sha256="6" * 64,
        provider="qwen",
        model="qwen3.7-plus-2026-05-26",
        endpoint=config.PROVIDER_ENDPOINTS["qwen"],
        max_tokens=None,
        max_completion_tokens=4096,
        status="parsed",
        request_id="req-feedback-round2-privacy",
        raw_response_text=raw_response_text,
        raw_response_sha256=sha256_bytes(raw_response_text.encode("utf-8")),
        raw_response_bytes=len(raw_response_text.encode("utf-8")),
        tool_calls=(),
        tool_call_count=0,
        parsed_feedback=parsed,
        usage=LLMUsage(input_tokens=20, output_tokens=8),
        finish_reason="stop",
        latency_ms=3,
        reasoning_present=False,
        reasoning_tokens=None,
        reasoning_bytes=0,
        reasoning_sha256=None,
    )

    redacted = redact_feedback_result_for_creator_privacy(result)

    assert redacted.status == "parse_error"
    assert redacted.error_code == "creator_projection_privacy"
    assert redacted.response_redaction_reason == "creator_projection_privacy"
    assert redacted.raw_response_text is None
    assert redacted.tool_calls is None
    assert redacted.parsed_feedback is None
    assert redacted.raw_response_sha256 == result.raw_response_sha256
    assert redacted.usage == result.usage
    path = write_feedback_evaluation_result(
        tmp_path / "privacy-terminal.json", redacted
    )
    assert load_feedback_evaluation_result(path) == redacted


def test_visual_feedback_runner_rejects_image_outside_authorized_catalog(
    monkeypatch,
) -> None:
    packet = _packet()
    runtime = SimpleNamespace()

    monkeypatch.setattr(
        "skillchain.evaluation.feedback_runtime.load_verified_evaluator_image",
        lambda *_, **__: (_ for _ in ()).throw(
            EvaluatorImageLoadError(
                "evaluator image is outside the verified remote-processing catalog"
            )
        ),
    )

    with pytest.raises(
        FeedbackRuntimeError,
        match="evaluator image is outside the verified remote-processing catalog",
    ):
        run_visual_feedback(
            packet,
            _lock(),
            remote_runtime=runtime,
            max_completion_tokens=4096,
            record_usage=False,
        )


def test_visual_feedback_parse_error_is_retained_without_retry(monkeypatch) -> None:
    packet = _packet()
    runtime = SimpleNamespace(
        authorization=SimpleNamespace(authorization_id="feedback-v2"),
        authorization_file_sha256="b" * 64,
        receipt_file_sha256="c" * 64,
        receipt=SimpleNamespace(receipt_sha256="d" * 64),
        catalog=SimpleNamespace(catalog_sha256="a" * 64),
    )
    calls = 0
    invalid_response = (
        canonical_json_bytes(
            {
                "schema_version": 1,
                "summary": "valid object before trailing garbage",
                "rule_violations": [],
                "ideal_response_gaps": [],
                "skill_suggestions": [],
            }
        )
        .decode("utf-8")
        .strip()
        + "\n["
    )

    def fake_chat(*args, **kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            requested_model=config.FEEDBACK_JUDGE_MODEL,
            response_model=config.FEEDBACK_JUDGE_MODEL,
            request_id="req-feedback-invalid",
            text=invalid_response,
            usage=LLMUsage(input_tokens=10, output_tokens=4),
            finish_reason="stop",
            latency_ms=2,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(
        "skillchain.evaluation.feedback_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, b"visual-feedback-image"),
    )
    result = run_visual_feedback(
        packet,
        _lock(),
        remote_runtime=runtime,
        max_completion_tokens=4096,
        record_usage=False,
    )

    assert calls == 1
    assert result.status == "parse_error"
    assert result.error_code == "invalid_feedback_json"
    assert result.parsed_feedback is None
    assert result.raw_response_text == invalid_response

    tampered = result.model_dump(mode="json")
    valid_response = (
        canonical_json_bytes(
            {
                "schema_version": 1,
                "summary": "valid feedback",
                "rule_violations": [],
                "ideal_response_gaps": [],
                "skill_suggestions": [],
            }
        )
        .decode("utf-8")
        .strip()
    )
    tampered["raw_response_text"] = valid_response
    tampered["raw_response_sha256"] = sha256_bytes(valid_response.encode("utf-8"))
    tampered["raw_response_bytes"] = len(valid_response.encode("utf-8"))
    with pytest.raises(
        ValueError,
        match="parse-error Feedback result contains a valid response",
    ):
        FeedbackEvaluationResult.model_validate_json(
            canonical_json_bytes(_resign_result(tampered)),
            strict=True,
        )


def test_legacy_fenced_parse_error_receipt_remains_terminal(tmp_path) -> None:
    """A frozen v1 parse error is not reinterpreted when receipts are loaded."""

    packet = _packet()
    payload = {
        "schema_version": 1,
        "summary": "Valid JSON was wrapped by the provider.",
        "rule_violations": [],
        "ideal_response_gaps": [],
        "skill_suggestions": [],
    }
    raw_response_text = (
        "```json\n" + canonical_json_bytes(payload).decode().strip() + "\n```"
    )
    unsigned = {
        "schema_version": 2,
        "result_kind": "visual-feedback",
        "cache_namespace": "feedback-evaluator-v2",
        "formal_eligible": False,
        "query_id": packet.query_id,
        "packet_sha256": packet.packet_sha256,
        "prompt_sha256": "1" * 64,
        "image_sha256": packet.image.sha256,
        "wire_sha256": "2" * 64,
        "asset_catalog_sha256": "3" * 64,
        "remote_authorization_id": "legacy-feedback-v1",
        "remote_authorization_file_sha256": "4" * 64,
        "remote_receipt_file_sha256": "5" * 64,
        "remote_receipt_sha256": "6" * 64,
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "endpoint": config.PROVIDER_ENDPOINTS["gemini"],
        "max_tokens": 2048,
        "attempts": 1,
        "max_attempts": 1,
        "status": "parse_error",
        "request_id": "req-legacy-fenced",
        "raw_response_text": raw_response_text,
        "raw_response_sha256": sha256_bytes(raw_response_text.encode()),
        "raw_response_bytes": len(raw_response_text.encode()),
        "tool_calls": [],
        "tool_call_count": 0,
        "response_redaction_reason": None,
        "parsed_feedback": None,
        "usage": {"input_tokens": 10, "output_tokens": 8},
        "finish_reason": "stop",
        "latency_ms": 2,
        "error_code": "invalid_feedback_json",
    }

    legacy_bytes = canonical_json_bytes(_resign_result(unsigned))
    result = FeedbackEvaluationResult.model_validate_json(legacy_bytes, strict=True)

    assert result.status == "parse_error"
    assert result.raw_response_text == raw_response_text
    assert result.parser_policy_version is None
    assert b"parser_policy" not in legacy_bytes
    legacy_path = tmp_path / "legacy-feedback-v2.json"
    legacy_path.write_bytes(legacy_bytes)
    assert load_feedback_evaluation_result(legacy_path) == result


def test_v2_policy_whitespace_parse_error_remains_terminal(tmp_path) -> None:
    """The new trim policy never reinterprets an existing v2-policy result."""

    packet = _packet()
    payload = {
        "schema_version": 1,
        "summary": " leading space rejected by the frozen v2 policy",
        "rule_violations": [],
        "ideal_response_gaps": [],
        "skill_suggestions": [],
    }
    raw_response_text = (
        "```json\n" + canonical_json_bytes(payload).decode().strip() + "\n```"
    )
    result = _make_result(
        cache_namespace="feedback-evaluator-v3",
        parser_policy_version="visual-feedback-complete-fence-wrapper-v2",
        parser_policy_sha256=(
            "09196a532b6369a24bebabfcc1916c6714a9066e127037c97ef9b022343fa9ee"
        ),
        query_id=packet.query_id,
        packet_sha256=packet.packet_sha256,
        prompt_sha256="1" * 64,
        image_sha256=packet.image.sha256,
        wire_sha256="2" * 64,
        asset_catalog_sha256="3" * 64,
        remote_authorization_id="legacy-feedback-v2-policy",
        remote_authorization_file_sha256="4" * 64,
        remote_receipt_file_sha256="5" * 64,
        remote_receipt_sha256="6" * 64,
        provider="gemini",
        model="gemini-3.6-flash",
        endpoint=config.PROVIDER_ENDPOINTS["gemini"],
        max_tokens=2048,
        status="parse_error",
        request_id="req-legacy-v2-whitespace",
        raw_response_text=raw_response_text,
        raw_response_sha256=sha256_bytes(raw_response_text.encode()),
        raw_response_bytes=len(raw_response_text.encode()),
        tool_calls=(),
        tool_call_count=0,
        usage=LLMUsage(input_tokens=10, output_tokens=8),
        finish_reason="stop",
        latency_ms=2,
        error_code="invalid_feedback_json",
    )
    legacy_payload = result.model_dump(mode="json")
    legacy_payload.pop("prompt_policy_version")
    legacy_payload.pop("prompt_policy_sha256")
    legacy_payload.pop("transport_policy_version")
    legacy_payload.pop("transport_policy_sha256")
    legacy_payload.pop("requested_response_format")
    legacy_payload.pop("requested_json_schema_sha256")
    legacy_payload.pop("requested_thinking")
    legacy_payload.pop("requested_thinking_budget")
    legacy_payload.pop("requested_timeout_seconds")
    legacy_payload.pop("requested_temperature")
    legacy_payload.pop("requested_top_p")
    legacy_payload.pop("reasoning_present")
    legacy_payload.pop("reasoning_tokens")
    legacy_payload.pop("reasoning_bytes")
    legacy_payload.pop("reasoning_sha256")
    legacy_payload.pop("max_completion_tokens")
    legacy_bytes = canonical_json_bytes(legacy_payload)
    legacy_path = tmp_path / "legacy-feedback-v3-result.json"
    legacy_path.write_bytes(legacy_bytes)

    loaded = load_feedback_evaluation_result(legacy_path)

    assert loaded == result
    assert loaded.status == "parse_error"
    assert loaded.cache_namespace == "feedback-evaluator-v3"
    assert loaded.raw_response_text == raw_response_text


def test_v3_prompt_result_remains_terminal_without_v4_prompt_identity(
    tmp_path,
) -> None:
    """A control-v3 result remains byte-compatible under the active v4 prompt."""

    packet = _packet()
    raw_response_text = (
        '{"schema_version":1,"summary":"schema drift",'
        '"rule_violations":[],"ideal_response_gaps":[],'
        '"skill_suggestions":[],"output_contract":{}}'
    )
    result = _make_result(
        cache_namespace="feedback-evaluator-v4",
        parser_policy_version="visual-feedback-free-text-trim-v3",
        parser_policy_sha256=(
            "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
        ),
        query_id=packet.query_id,
        packet_sha256=packet.packet_sha256,
        prompt_sha256="1" * 64,
        image_sha256=packet.image.sha256,
        wire_sha256="2" * 64,
        asset_catalog_sha256="3" * 64,
        remote_authorization_id="legacy-feedback-v3-prompt",
        remote_authorization_file_sha256="4" * 64,
        remote_receipt_file_sha256="5" * 64,
        remote_receipt_sha256="6" * 64,
        provider="gemini",
        model="gemini-3.6-flash",
        endpoint=config.PROVIDER_ENDPOINTS["gemini"],
        max_tokens=2048,
        status="parse_error",
        request_id="req-legacy-v3-prompt",
        raw_response_text=raw_response_text,
        raw_response_sha256=sha256_bytes(raw_response_text.encode()),
        raw_response_bytes=len(raw_response_text.encode()),
        tool_calls=(),
        tool_call_count=0,
        usage=LLMUsage(input_tokens=10, output_tokens=8),
        finish_reason="stop",
        latency_ms=2,
        error_code="invalid_feedback_json",
    )
    path = write_feedback_evaluation_result(tmp_path / "legacy-v4-result.json", result)

    assert b"prompt_policy" not in path.read_bytes()
    loaded = load_feedback_evaluation_result(path)
    assert loaded == result
    assert loaded.status == "parse_error"
    assert loaded.prompt_policy_version is None


def test_prompt_v4_result_remains_byte_compatible_without_transport_identity(
    tmp_path,
) -> None:
    """A frozen result-v5 keeps its exact pre-JSON-mode byte contract."""

    packet = _packet()
    raw_response_text = (
        '{"schema_version":1,"summary":"valid feedback",'
        '"rule_violations":[],"ideal_response_gaps":[],'
        '"skill_suggestions":[]}'
    )
    parsed = parse_visual_feedback_output_v3(raw_response_text)
    result = _make_result(
        cache_namespace="feedback-evaluator-v5",
        parser_policy_version="visual-feedback-free-text-trim-v3",
        parser_policy_sha256=(
            "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
        ),
        prompt_policy_version=VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
        prompt_policy_sha256=VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
        query_id=packet.query_id,
        packet_sha256=packet.packet_sha256,
        prompt_sha256="1" * 64,
        image_sha256=packet.image.sha256,
        wire_sha256="2" * 64,
        asset_catalog_sha256="3" * 64,
        remote_authorization_id="legacy-feedback-v4-prompt",
        remote_authorization_file_sha256="4" * 64,
        remote_receipt_file_sha256="5" * 64,
        remote_receipt_sha256="6" * 64,
        provider="gemini",
        model="gemini-3.6-flash",
        endpoint=config.PROVIDER_ENDPOINTS["gemini"],
        max_tokens=2048,
        status="parsed",
        request_id="req-legacy-v5-result",
        raw_response_text=raw_response_text,
        raw_response_sha256=sha256_bytes(raw_response_text.encode()),
        raw_response_bytes=len(raw_response_text.encode()),
        tool_calls=(),
        tool_call_count=0,
        parsed_feedback=parsed,
        usage=LLMUsage(input_tokens=10, output_tokens=8),
        finish_reason="stop",
        latency_ms=2,
    )
    path = write_feedback_evaluation_result(tmp_path / "legacy-v5-result.json", result)

    frozen_bytes = path.read_bytes()
    assert b"transport_policy" not in frozen_bytes
    assert b"requested_response_format" not in frozen_bytes
    assert b"prompt_policy" in frozen_bytes
    assert load_feedback_evaluation_result(path) == result


def test_gemini_json_transport_result_remains_byte_compatible_after_role_swap(
    tmp_path,
) -> None:
    """A terminal Gemini result-v6 never acquires Kimi sampling fields."""

    packet = _packet()
    raw_response_text = canonical_json_bytes(
        {
            "schema_version": 1,
            "summary": "historical Gemini JSON transport result",
            "rule_violations": [],
            "ideal_response_gaps": [],
            "skill_suggestions": [],
        }
    ).decode("utf-8")
    parsed = parse_visual_feedback_output_v3(raw_response_text)
    result = _make_result(
        cache_namespace="feedback-evaluator-v6",
        parser_policy_version="visual-feedback-free-text-trim-v3",
        parser_policy_sha256=(
            "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
        ),
        prompt_policy_version=VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
        prompt_policy_sha256=VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
        transport_policy_version=VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1,
        transport_policy_sha256=VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1,
        requested_response_format="json_object",
        query_id=packet.query_id,
        packet_sha256=packet.packet_sha256,
        prompt_sha256="1" * 64,
        image_sha256=packet.image.sha256,
        wire_sha256="2" * 64,
        asset_catalog_sha256="3" * 64,
        remote_authorization_id="historical-gemini-v5",
        remote_authorization_file_sha256="4" * 64,
        remote_receipt_file_sha256="5" * 64,
        remote_receipt_sha256="6" * 64,
        provider="gemini",
        model="gemini-3.6-flash",
        endpoint=config.PROVIDER_ENDPOINTS["gemini"],
        max_tokens=2048,
        status="parsed",
        request_id="req-historical-gemini-v6",
        raw_response_text=raw_response_text,
        raw_response_sha256=sha256_bytes(raw_response_text.encode("utf-8")),
        raw_response_bytes=len(raw_response_text.encode("utf-8")),
        tool_calls=(),
        tool_call_count=0,
        parsed_feedback=parsed,
        usage=LLMUsage(input_tokens=10, output_tokens=8),
        finish_reason="stop",
        latency_ms=2,
    )
    path = write_feedback_evaluation_result(
        tmp_path / "historical-gemini-result-v6.json", result
    )

    frozen_bytes = path.read_bytes()
    assert b'"cache_namespace":"feedback-evaluator-v6"' in frozen_bytes
    assert b'"requested_response_format":"json_object"' in frozen_bytes
    assert b"requested_thinking" not in frozen_bytes
    assert b"requested_temperature" not in frozen_bytes
    assert b"requested_top_p" not in frozen_bytes
    assert load_feedback_evaluation_result(path) == result


def test_kimi_prompt_v4_result_remains_byte_compatible_after_prompt_v5(
    tmp_path,
) -> None:
    """A terminal Kimi result-v7 keeps its frozen prompt/transport identity."""

    packet = _packet()
    raw_response_text = canonical_json_bytes(
        {
            "schema_version": 1,
            "summary": "historical Kimi prompt-v4 result",
            "rule_violations": [],
            "ideal_response_gaps": [],
            "skill_suggestions": [],
        }
    ).decode("utf-8")
    parsed = parse_visual_feedback_output_v3(raw_response_text)
    result = _make_result(
        cache_namespace="feedback-evaluator-v7",
        parser_policy_version="visual-feedback-free-text-trim-v3",
        parser_policy_sha256=(
            "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
        ),
        prompt_policy_version=VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
        prompt_policy_sha256=VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
        transport_policy_version=VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2,
        transport_policy_sha256=VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V2,
        requested_response_format=None,
        requested_thinking=False,
        requested_temperature=0.6,
        requested_top_p=0.95,
        query_id=packet.query_id,
        packet_sha256=packet.packet_sha256,
        prompt_sha256="1" * 64,
        image_sha256=packet.image.sha256,
        wire_sha256="2" * 64,
        asset_catalog_sha256="3" * 64,
        remote_authorization_id="historical-kimi-prompt-v4",
        remote_authorization_file_sha256="4" * 64,
        remote_receipt_file_sha256="5" * 64,
        remote_receipt_sha256="6" * 64,
        provider="kimi",
        model="kimi-k2.6",
        endpoint=config.PROVIDER_ENDPOINTS["kimi"],
        max_tokens=2048,
        status="parsed",
        request_id="req-historical-kimi-v7",
        raw_response_text=raw_response_text,
        raw_response_sha256=sha256_bytes(raw_response_text.encode("utf-8")),
        raw_response_bytes=len(raw_response_text.encode("utf-8")),
        tool_calls=(),
        tool_call_count=0,
        parsed_feedback=parsed,
        usage=LLMUsage(input_tokens=10, output_tokens=8),
        finish_reason="stop",
        latency_ms=2,
    )
    path = write_feedback_evaluation_result(
        tmp_path / "historical-kimi-result-v7.json", result
    )

    frozen_bytes = path.read_bytes()
    assert b'"cache_namespace":"feedback-evaluator-v7"' in frozen_bytes
    assert VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4.encode() in frozen_bytes
    assert VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2.encode() in frozen_bytes
    assert VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5.encode() not in frozen_bytes
    assert VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3.encode() not in frozen_bytes
    assert load_feedback_evaluation_result(path) == result


def test_visual_feedback_retains_wrapped_timeout(monkeypatch) -> None:
    packet = _packet()
    runtime = SimpleNamespace(
        authorization=SimpleNamespace(authorization_id="feedback-v2"),
        authorization_file_sha256="b" * 64,
        receipt_file_sha256="c" * 64,
        receipt=SimpleNamespace(receipt_sha256="d" * 64),
        catalog=SimpleNamespace(catalog_sha256="a" * 64),
    )
    monkeypatch.setattr(
        "skillchain.evaluation.feedback_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, b"visual-feedback-image"),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: (_ for _ in ()).throw(
            llm.LLMTimeoutError("provider request timed out")
        ),
    )

    result = run_visual_feedback(
        packet,
        _lock(),
        remote_runtime=runtime,
        max_completion_tokens=4096,
        record_usage=False,
    )

    assert result.status == "timeout"
    assert result.error_code == "timeout"
    assert result.request_id is None


def test_visual_feedback_redacts_input_image_echo(
    monkeypatch,
    tmp_path,
) -> None:
    packet = _packet()
    runtime = SimpleNamespace(
        authorization=SimpleNamespace(authorization_id="feedback-v2"),
        authorization_file_sha256="b" * 64,
        receipt_file_sha256="c" * 64,
        receipt=SimpleNamespace(receipt_sha256="d" * 64),
        catalog=SimpleNamespace(catalog_sha256="a" * 64),
    )
    image_bytes = b"visual-feedback-image"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    echoed_payload = (
        canonical_json_bytes(
            {
                "schema_version": 1,
                "summary": data_url,
                "rule_violations": [],
                "ideal_response_gaps": [],
                "skill_suggestions": [],
            }
        )
        .decode("utf-8")
        .strip()
    )

    monkeypatch.setattr(
        "skillchain.evaluation.feedback_runtime.load_verified_evaluator_image",
        lambda *_, **__: (runtime, image_bytes),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_, **__: LLMResponse(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            requested_model=config.FEEDBACK_JUDGE_MODEL,
            response_model=config.FEEDBACK_JUDGE_MODEL,
            request_id="req-feedback-image-echo",
            text=echoed_payload,
            usage=LLMUsage(input_tokens=10, output_tokens=10),
            finish_reason="stop",
            latency_ms=2,
        ),
    )
    result = run_visual_feedback(
        packet,
        _lock(),
        remote_runtime=runtime,
        max_completion_tokens=4096,
        record_usage=False,
    )

    assert result.status == "parse_error"
    assert result.error_code == "input_image_echo"
    assert result.response_redaction_reason == "input_image_echo"
    assert result.raw_response_text is None
    assert result.raw_response_sha256 == sha256_bytes(echoed_payload.encode("utf-8"))
    receipt = write_feedback_evaluation_result(
        tmp_path / "redacted-feedback.json",
        result,
    ).read_bytes()
    assert data_url.encode("utf-8") not in receipt
    assert base64.b64encode(image_bytes) not in receipt
