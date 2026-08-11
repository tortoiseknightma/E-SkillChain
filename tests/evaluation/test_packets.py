from __future__ import annotations

import base64
import hashlib
import json

import pytest
from PIL import Image
from pydantic import ValidationError

from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.evaluation.packets import (
    AssistantResult,
    AssistantToolTrace,
    FeedbackPacket,
    FinalEvaluationPacket,
    JudgeDimensionScore,
    JudgeScores,
    RubricSnapshot,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
    VISUAL_FEEDBACK_PROMPT_OUTPUT_IDENTITY_SHA256_V5,
    VisibleCard,
    VisibleCitation,
    VisibleToolEvidence,
    build_feedback_evaluator_prompt,
    build_feedback_evaluator_prompt_v3,
    build_feedback_evaluator_prompt_v4,
    build_feedback_packet,
    build_final_evaluation_packet,
    build_final_evaluator_prompt,
    canonical_feedback_packet_bytes,
    canonical_final_packet_bytes,
    conservative_assistant_error,
    conservative_judge_error,
    evaluator_wire_messages,
    validate_paired_result_rows,
    visual_feedback_prompt_policy_v4,
    visual_feedback_prompt_policy_v5,
    visual_feedback_prompt_output_identity_v5,
)
from skillchain.llm import LLMUsage
from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes
from skillchain.taxonomy import TASK_SPEC_VERSION, TAXONOMY_VERSION


BLINDING_KEY = b"fixed-test-only-blinding-key-32-bytes!!"


@pytest.fixture
def verified_query_catalog(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "input.jpg"
    Image.new("RGB", (32, 32), color=(12, 34, 56)).save(image_path, format="JPEG")
    draft = DatasetAssetDraft(
        source_dataset="fixture",
        source_revision="fixture-v1",
        source_record_id="fixture:input",
        transform_policy_version="identity-v1",
        local_path="images/input.jpg",
        license_id="CC0-1.0",
        source_url="https://example.org/source",
        attribution="fixture",
        cloud_upload_allowed=True,
        public_demo_allowed=False,
    )
    asset = inventory_dataset_asset(draft, tmp_path)
    publish_asset_catalog(
        [asset], tmp_path / "catalog", tmp_path, coverage_roots=["images"]
    )
    catalog = load_asset_catalog(tmp_path / "catalog", tmp_path, verify_files=True)
    query = _query().model_copy(
        update={
            "asset_id": asset.asset_id,
            "image_path": asset.local_path,
            "leakage_group_id": catalog.component_for_asset(asset.asset_id),
        }
    )
    return query, catalog


def _rubric() -> RubricSnapshot:
    content = "Frozen TCR/CCC/CQ/CA rubric v0"
    return RubricSnapshot(
        rubric_id="judge-rubric-v0",
        rubric_version="0",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def _query(*, query_id: str = "query-001", requires_card: bool = True) -> Query:
    capability = (
        "product.exact_match" if requires_card else "knowledge.visual_encyclopedia"
    )
    intent = "exact_match" if requires_card else "encyclopedia"
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=TASK_SPEC_VERSION,
        query_id=query_id,
        asset_id="asset.secret-source",
        image_path="secret/test_frozen/image.jpg",
        leakage_group_id="secret-leakage-group",
        boundary_group_id=None,
        template_family="secret-template-family",
        generator_batch_id="secret-generator",
        text="请识别图片里的商品",
        turns=[{"role": "user", "content": "请识别图片里的商品"}],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=False,
        boundary_strategy=None,
        requires_card=requires_card,
        split="test_frozen",
        label_status="arbitrated",
        label_provenance=[
            LabelDecision(
                decision_type="arbitration",
                annotator_kind="human",
                annotator_id="secret-annotator",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _result(
    *, config: str = "llm_static", query_id: str = "query-001"
) -> AssistantResult:
    trace = AssistantToolTrace(
        call_index=1,
        tool_name="image_product_search",
        status="success",
        arguments_sha256="1" * 64,
        result_sha256="2" * 64,
        runtime_binding_sha256="3" * 64,
        latency_ms=4,
    )
    visible_card = VisibleCard(
        title="公开商品标题",
        body="公开商品描述",
        fields=(("颜色", "蓝色"),),
    )
    visible_tool = VisibleToolEvidence(
        tool_name="image_product_search",
        status="success",
        visible_text="找到一个候选商品",
        cards=(visible_card,),
        citations=(
            VisibleCitation(
                title="公开商品页",
                uri="https://example.org/item",
                excerpt="与图片外观一致",
            ),
        ),
    )
    skilled = config != "noskill"
    return AssistantResult(
        run_id="run-secret-lineage",
        query_id=query_id,
        config=config,
        response_text="这是用户实际看到的回答",
        visible_cards=(visible_card,),
        visible_tool_evidence=(visible_tool,),
        tool_trace=(trace,),
        selected_capability="product.exact_match" if skilled else None,
        skill_slug="secret-skill-slug" if skilled else None,
        bank_sha256="4" * 64 if skilled else None,
        route_trace_sha256="5" * 64 if skilled else None,
        query_artifact_sha256="6" * 64,
        split_manifest_sha256="7" * 64,
        registry_sha256="8" * 64,
        registry_runtime_sha256="9" * 64,
        backbone_provider="qwen",
        backbone_model="qwen-secret-model",
        backbone_request_id="secret-request-id",
        usage=LLMUsage(input_tokens=10, output_tokens=20),
        latency_ms=30,
    )


def test_final_packet_allowlist_omits_treatment_ground_truth_and_paths(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    packet = build_final_evaluation_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )
    raw = canonical_final_packet_bytes(packet)
    assert packet.schema_version == 3
    for secret in (
        b"llm_static",
        b"secret-skill-slug",
        b"product.exact_match",
        b"test_frozen",
        b"secret/test_frozen/image.jpg",
        b"secret-leakage-group",
        b"secret-template-family",
        b"qwen-secret-model",
        b"secret-request-id",
        b"query-001",
    ):
        assert secret not in raw
    assert b"final-evaluator-v2" in raw
    assert b"feedback-evaluator-v2" not in raw
    assert FinalEvaluationPacket.model_validate_json(raw) == packet


def test_forbidden_card_requirement_is_preserved_without_hidden_labels_or_ccc(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    query = query.model_copy(
        update={
            "canonical_intent": "encyclopedia",
            "canonical_capability": "knowledge.visual_encyclopedia",
            "acceptable_capabilities": ["knowledge.visual_encyclopedia"],
            "requires_card": False,
        }
    )
    result = _result(config="noskill").model_copy(
        update={
            "response_text": "Public grounded answer.",
            "visible_cards": (),
            "visible_tool_evidence": (),
            "tool_trace": (),
        }
    )
    packet = build_final_evaluation_packet(
        query,
        result,
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )
    prompt = build_final_evaluator_prompt(packet)
    visible = json.loads(prompt.messages[1].content)

    assert packet.schema_version == 3
    assert packet.card_requirement == "forbidden"
    assert visible["card_requirement"] == "forbidden"
    assert visible["output_contract"]["requires_card"] is False
    assert visible["output_contract"]["dimension_order"] == ["CA", "CQ", "TCR"]
    assert "CCC" not in visible["output_contract"]["dimension_maxima"]
    serialized = prompt.messages[1].content
    for hidden in (
        query.query_id,
        query.canonical_intent,
        query.canonical_capability,
        query.asset_id,
        query.split,
    ):
        assert hidden not in serialized


def test_legacy_v2_not_applicable_final_packet_remains_loadable(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    query = query.model_copy(
        update={
            "canonical_intent": "encyclopedia",
            "canonical_capability": "knowledge.visual_encyclopedia",
            "acceptable_capabilities": ["knowledge.visual_encyclopedia"],
            "requires_card": False,
        }
    )
    result = _result(config="noskill").model_copy(
        update={
            "response_text": "Public grounded answer.",
            "visible_cards": (),
            "visible_tool_evidence": (),
            "tool_trace": (),
        }
    )
    packet = build_final_evaluation_packet(
        query,
        result,
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )
    legacy = packet.model_dump(mode="json")
    legacy["schema_version"] = 2
    legacy["card_requirement"] = "not_applicable"
    unsigned = {key: value for key, value in legacy.items() if key != "packet_sha256"}
    legacy["packet_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    content = canonical_json_bytes(legacy)

    reparsed = FinalEvaluationPacket.model_validate_json(content, strict=True)

    assert reparsed.schema_version == 2
    assert reparsed.card_requirement == "not_applicable"
    assert canonical_final_packet_bytes(reparsed) == content


def test_public_tool_name_does_not_false_positive_as_hidden_intent(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    encyclopedia_query = query.model_copy(
        update={
            "canonical_intent": "encyclopedia",
            "canonical_capability": "knowledge.visual_encyclopedia",
            "acceptable_capabilities": ["knowledge.visual_encyclopedia"],
            "requires_card": False,
        }
    )
    evidence = VisibleToolEvidence(
        tool_name="encyclopedia_lookup",
        status="success",
        visible_text="Public grounded description.",
    )
    result = _result(config="noskill").model_copy(
        update={
            "response_text": "Public answer.",
            "visible_cards": (),
            "visible_tool_evidence": (evidence,),
        }
    )

    packet = build_final_evaluation_packet(
        encyclopedia_query,
        result,
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )

    assert packet.tool_evidence[0].tool_name == "encyclopedia_lookup"


@pytest.mark.parametrize("location", ["response_text", "visible_evidence"])
def test_public_intent_word_does_not_false_positive_as_hidden_identity(
    verified_query_catalog,
    location: str,
) -> None:
    query, catalog = verified_query_catalog
    encyclopedia_query = query.model_copy(
        update={
            "canonical_intent": "encyclopedia",
            "canonical_capability": "knowledge.visual_encyclopedia",
            "acceptable_capabilities": ["knowledge.visual_encyclopedia"],
            "requires_card": False,
        }
    )
    result_updates: dict[str, object] = {
        "response_text": "Public grounded answer.",
        "visible_cards": (),
        "visible_tool_evidence": (),
        "selected_capability": "knowledge.visual_encyclopedia",
    }
    if location == "response_text":
        result_updates["response_text"] = (
            "The identification is supported by encyclopedia evidence."
        )
    else:
        result_updates["visible_tool_evidence"] = (
            VisibleToolEvidence(
                tool_name="encyclopedia_lookup",
                status="success",
                visible_text="Public encyclopedia evidence.",
            ),
        )
    result = _result().model_copy(update=result_updates)

    packet = build_final_evaluation_packet(
        encyclopedia_query,
        result,
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )

    assert packet.response_text == result.response_text
    assert packet.tool_evidence == result.visible_tool_evidence


@pytest.mark.parametrize(
    ("location", "leaked_value"),
    [
        ("response_text", "secret-skill-slug"),
        ("visible_evidence", "knowledge.visual_encyclopedia"),
    ],
)
def test_public_output_still_rejects_true_hidden_identity_literals(
    verified_query_catalog,
    location: str,
    leaked_value: str,
) -> None:
    query, catalog = verified_query_catalog
    encyclopedia_query = query.model_copy(
        update={
            "canonical_intent": "encyclopedia",
            "canonical_capability": "knowledge.visual_encyclopedia",
            "acceptable_capabilities": ["knowledge.visual_encyclopedia"],
            "requires_card": False,
        }
    )
    result_updates: dict[str, object] = {
        "response_text": "Public grounded answer.",
        "visible_cards": (),
        "visible_tool_evidence": (),
        "selected_capability": "knowledge.visual_encyclopedia",
    }
    if location == "response_text":
        result_updates["response_text"] = f"Debug identity: {leaked_value}"
    else:
        result_updates["visible_tool_evidence"] = (
            VisibleToolEvidence(
                tool_name="encyclopedia_lookup",
                status="success",
                visible_text=f"Debug identity: {leaked_value}",
            ),
        )
    result = _result().model_copy(update=result_updates)

    with pytest.raises(
        ValueError,
        match="Assistant public output contains hidden evaluation identity",
    ):
        build_final_evaluation_packet(
            encyclopedia_query,
            result,
            asset_catalog=catalog,
            rubric=_rubric(),
            blinding_key=BLINDING_KEY,
        )


def test_feedback_is_separate_and_cannot_validate_as_final(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    feedback = build_feedback_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
    )
    raw = canonical_feedback_packet_bytes(feedback)
    assert b"feedback-evaluator-v2" in raw
    assert b"product.exact_match" in raw
    assert feedback.turns == tuple(query.turns)
    assert feedback.image.sha256
    assert b"content_base64" not in raw
    assert FeedbackPacket.model_validate_json(raw) == feedback
    with pytest.raises(ValidationError):
        FinalEvaluationPacket.model_validate_json(raw)


def test_failed_assistant_bypasses_judge_packet_and_scores_zero(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    failed = _result().model_copy(update={"response_text": "", "error_code": "timeout"})

    with pytest.raises(ValueError, match="bypass Judge packets"):
        build_final_evaluation_packet(
            query,
            failed,
            asset_catalog=catalog,
            rubric=_rubric(),
            blinding_key=BLINDING_KEY,
        )

    outcome = conservative_assistant_error(
        query=query,
        result=failed,
        blinding_key=BLINDING_KEY,
    )
    assert outcome.status == "assistant_error"
    assert outcome.judge_provider == "not-called"
    assert outcome.scores.j_project == 0.0
    assert all(item.score == 0 for item in outcome.scores.dimensions)


def test_prompt_snapshots_keep_final_and_feedback_namespaces_separate(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    final = build_final_evaluation_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )
    feedback = build_feedback_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
    )
    final_prompt = build_final_evaluator_prompt(final)
    feedback_prompt = build_feedback_evaluator_prompt(feedback)
    final_bytes = canonical_json_bytes(final_prompt)
    judge_visible_bytes = final_prompt.messages[1].content.encode()
    assert final_prompt.packet_kind == "final"
    assert feedback_prompt.packet_kind == "feedback"
    assert final_prompt.prompt_sha256 != feedback_prompt.prompt_sha256
    assert b"feedback-evaluator-v2" not in final_bytes
    assert b"llm_static" not in final_bytes
    for hidden in (
        final.evaluation_id.encode(),
        final.packet_sha256.encode(),
        final.image.sha256.encode(),
        final.rubric.content_sha256.encode(),
        b"final-evaluator-v2",
        b"query-001",
    ):
        assert hidden not in judge_visible_bytes
    assert b"content_base64" not in final_bytes
    assert b"content_base64" not in canonical_json_bytes(feedback_prompt)
    image_bytes = (catalog.asset_root / query.image_path).read_bytes()
    wire = evaluator_wire_messages(feedback_prompt, image_bytes=image_bytes)
    image_url = wire[1]["content"][0]["image_url"]["url"]
    encoded = image_url.split(",", 1)[1]
    assert hashlib.sha256(base64.b64decode(encoded)).hexdigest() == (
        feedback.image.sha256
    )
    with pytest.raises(TypeError, match="FinalEvaluationPacket"):
        build_final_evaluator_prompt(feedback)  # type: ignore[arg-type]


def test_v5_feedback_prompt_locks_response_v1_and_preserves_v3_v4_builders(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    packet = build_feedback_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
    )

    historical = build_feedback_evaluator_prompt_v3(packet)
    historical_v4 = build_feedback_evaluator_prompt_v4(packet)
    active = build_feedback_evaluator_prompt(packet)
    visible = json.loads(active.messages[1].content)
    historical_visible = json.loads(historical.messages[1].content)
    historical_v4_visible = json.loads(historical_v4.messages[1].content)

    assert active.prompt_sha256 != historical.prompt_sha256
    assert active.prompt_sha256 != historical_v4.prompt_sha256
    assert "response_identity" not in historical_v4_visible
    assert historical_v4_visible["output_contract"] == visible["output_contract"]
    assert "do not copy or echo output_contract itself" in (
        historical_v4.messages[0].content
    )
    assert "do not copy or echo output_contract itself" in (
        active.messages[0].content.casefold()
    )
    assert "schema_version is mandatory" in active.messages[0].content
    assert "JSON integer 1" in active.messages[0].content
    assert "schema_version 2 or 3" in active.messages[0].content
    assert "one to four strings, never five" in active.messages[0].content
    assert "top_level_fields" in historical_visible["output_contract"]
    assert (
        "top_level_fields_exactly_once" not in (historical_visible["output_contract"])
    )
    contract = visible["output_contract"]
    assert contract["top_level_fields_exactly_once"] == [
        "schema_version",
        "summary",
        "rule_violations",
        "ideal_response_gaps",
        "skill_suggestions",
    ]
    assert contract["skill_suggestions_item_schema"]["type"] == "string"
    response_identity = visible["response_identity"]
    assert response_identity == visual_feedback_prompt_output_identity_v5()
    assert response_identity["mandatory_response_schema_version"] == {
        "json_type": "integer",
        "literal": 1,
    }
    assert response_identity["ignore_input_schema_versions"] == [2, 3]
    skeleton = response_identity["exact_five_key_skeleton"]
    assert set(skeleton) == {
        "schema_version",
        "summary",
        "rule_violations",
        "ideal_response_gaps",
        "skill_suggestions",
    }
    assert skeleton["schema_version"] == 1
    assert len(skeleton["rule_violations"][0]["evidence"]) == 1
    assert isinstance(skeleton["skill_suggestions"][0], str)
    assert (
        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4
        == "visual-feedback-exact-shape-prompt-v4"
    )
    assert (
        VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4
        == "954cd483e15d042d39f164b840d483281c159c1540f50dc8ec29888f8af181cd"
    )
    assert VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4 == sha256_bytes(
        canonical_json_bytes(visual_feedback_prompt_policy_v4())
    )
    assert (
        visual_feedback_prompt_policy_v4()["system_prompt_sha256"]
        == "824cc41f99f545d7d77096caf9040107cef86880ed4af8cf92027c7f3592c591"
    )
    assert (
        VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5
        == "visual-feedback-response-schema-v1-prompt-v5"
    )
    assert VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5 == sha256_bytes(
        canonical_json_bytes(visual_feedback_prompt_policy_v5())
    )
    assert VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5 == (
        "024c1a8831b3497a904272cd9b9c2352fbbb97b633755388536da8475cbdfffb"
    )
    assert VISUAL_FEEDBACK_PROMPT_OUTPUT_IDENTITY_SHA256_V5 == sha256_bytes(
        canonical_json_bytes(visual_feedback_prompt_output_identity_v5())
    )
    assert VISUAL_FEEDBACK_PROMPT_OUTPUT_IDENTITY_SHA256_V5 == (
        "b54a8f6f24b8a1caa69673818513423314070a6906122101a828acce7f5273c3"
    )


def test_rubric_rejects_hidden_treatment_fields() -> None:
    content = "Score the answer and inspect bank_sha256."
    with pytest.raises(ValidationError, match="forbidden"):
        RubricSnapshot(
            rubric_id="bad",
            rubric_version="0",
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        )


def test_final_packet_rejects_self_hash_tamper(verified_query_catalog) -> None:
    query, catalog = verified_query_catalog
    packet = build_final_evaluation_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )
    raw = packet.model_dump(mode="json")
    raw["response_text"] = "tampered"
    with pytest.raises(ValidationError, match="self hash"):
        FinalEvaluationPacket.model_validate_json(canonical_json_bytes(raw))


def test_final_packet_revalidates_catalog_bytes_and_blinds_query_id(
    verified_query_catalog,
) -> None:
    query, catalog = verified_query_catalog
    first = build_final_evaluation_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )
    second = build_final_evaluation_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=b"another-fixed-blinding-key-over-32-bytes",
    )
    assert first.evaluation_id != second.evaluation_id
    assert query.query_id.encode() not in canonical_final_packet_bytes(first)

    (catalog.asset_root / query.image_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset|SHA|fingerprint|catalog"):
        build_final_evaluation_packet(
            query,
            _result(),
            asset_catalog=catalog,
            rubric=_rubric(),
            blinding_key=BLINDING_KEY,
        )


def test_final_packet_rejects_short_blinding_key(verified_query_catalog) -> None:
    query, catalog = verified_query_catalog
    with pytest.raises(ValueError, match="blinding key"):
        build_final_evaluation_packet(
            query,
            _result(),
            asset_catalog=catalog,
            rubric=_rubric(),
            blinding_key=b"short",
        )


@pytest.mark.parametrize(
    ("dimension", "score", "tier"),
    [("TCR", 8, "Good"), ("CQ", 8, "Average"), ("CA", 3, "Poor")],
)
def test_judge_dimension_uses_frozen_tier_policy(dimension, score, tier) -> None:
    assert JudgeDimensionScore(dimension=dimension, score=score, tier=tier).tier == tier
    wrong_tier = "Good" if tier != "Good" else "Poor"
    with pytest.raises(ValidationError, match="tier"):
        JudgeDimensionScore(dimension=dimension, score=score, tier=wrong_tier)


def test_judge_scores_validate_card_denominators_and_dimension_sets() -> None:
    dimensions = (
        JudgeDimensionScore(dimension="CA", score=8, tier="Good"),
        JudgeDimensionScore(dimension="CCC", score=4, tier="Average"),
        JudgeDimensionScore(dimension="CQ", score=16, tier="Good"),
        JudgeDimensionScore(dimension="TCR", score=8, tier="Good"),
    )
    scores = JudgeScores(
        evaluation_id="e" * 64,
        requires_card=True,
        dimensions=dimensions,
        j_project=72.0,
    )
    assert scores.j_project == 72.0
    with pytest.raises(ValidationError, match="J_project"):
        JudgeScores(
            evaluation_id="e" * 64,
            requires_card=True,
            dimensions=dimensions,
            j_project=73.0,
        )


@pytest.mark.parametrize("requires_card", [False, True])
def test_failed_judge_rows_remain_with_conservative_zero_scores(requires_card) -> None:
    outcome = conservative_judge_error(
        evaluation_id="e" * 64,
        requires_card=requires_card,
        status="parse_error",
        attempts=2,
        max_attempts=2,
        judge_provider="provider",
        judge_model="model",
        prompt_sha256="a" * 64,
        error_code="invalid-json",
        raw_response_sha256="b" * 64,
    )
    assert outcome.status == "parse_error"
    assert outcome.scores.j_project == 0.0
    assert {item.dimension for item in outcome.scores.dimensions} == (
        {"TCR", "CCC", "CQ", "CA"} if requires_card else {"TCR", "CQ", "CA"}
    )


def test_paired_results_require_identical_order_and_keep_error_rows() -> None:
    configs = ("noskill", "llm_static", "s1", "s1s2", "full")
    rows = {
        config: [
            _result(config=config, query_id="q1"),
            _result(config=config, query_id="q2").model_copy(
                update={"response_text": "", "error_code": "timeout"}
            ),
        ]
        for config in configs
    }
    assert validate_paired_result_rows(rows) == ("q1", "q2")
    rows["full"] = list(reversed(rows["full"]))
    with pytest.raises(ValueError, match="identical query order"):
        validate_paired_result_rows(rows)


@pytest.mark.parametrize(
    "error_code",
    ("route_contract_error", "route_length"),
)
def test_route_failures_remain_in_the_frozen_zero_score_vocabulary(
    error_code: str,
) -> None:
    failed = _result().model_copy(
        update={"response_text": "", "error_code": error_code}
    )

    assert (
        AssistantResult.model_validate(
            failed.model_dump(mode="python"), strict=True
        ).error_code
        == error_code
    )


def test_models_forbid_hidden_or_unregistered_fields(verified_query_catalog) -> None:
    query, catalog = verified_query_catalog
    packet = build_final_evaluation_packet(
        query,
        _result(),
        asset_catalog=catalog,
        rubric=_rubric(),
        blinding_key=BLINDING_KEY,
    )
    raw = packet.model_dump(mode="json")
    raw["config"] = "full"
    raw["packet_sha256"] = hashlib.sha256(
        canonical_json_bytes({k: v for k, v in raw.items() if k != "packet_sha256"})
    ).hexdigest()
    with pytest.raises(ValidationError, match="Extra inputs"):
        FinalEvaluationPacket.model_validate(raw)

    with pytest.raises(ValidationError, match="unknown canonical tool"):
        VisibleToolEvidence(
            tool_name="hidden_debug_tool",
            status="success",
            visible_text="x",
        )


def test_context_violation_is_a_known_tool_error_code() -> None:
    trace = AssistantToolTrace(
        call_index=1,
        tool_name="image_product_search",
        status="error",
        arguments_sha256="1" * 64,
        runtime_binding_sha256="2" * 64,
        latency_ms=1,
        error_code="context_violation",
    )
    visible = VisibleToolEvidence(
        tool_name="image_product_search",
        status="error",
        error_code="context_violation",
    )

    assert trace.error_code == "context_violation"
    assert visible.error_code == "context_violation"
