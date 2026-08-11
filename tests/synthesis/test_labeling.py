import hashlib
import json
from pathlib import Path

import pytest

from skillchain import config
from skillchain.schemas import Query
from skillchain.synthesis.labeling import (
    BlindReviewInput,
    apply_arbitration,
    build_review_request,
    compare_reviews,
    pending_arbitrations,
    run_cross_review,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes
from conftest import make_query_v2


def make_query(index: int, *, intent: str = "exact_match") -> Query:
    return make_query_v2(
        query_id=f"dm-{index:03d}",
        image_path=f"query_images/mechanical-{index:03d}.jpg",
        text=f"机械占位-{index:03d}",
        intent=intent,
        split="dev_mini",
        generator_batch_id=f"dev-mini-{((index - 1) // 25) + 1:03d}",
    )


QUERY = make_query(1)
BLIND_INPUT = BlindReviewInput(query_id=QUERY.query_id, text=QUERY.text)


def write_queries(path: Path, queries: list[Query]) -> None:
    path.write_text(
        "".join(query.model_dump_json() + "\n" for query in queries),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def agreeing_chat(*args, **kwargs) -> str:
    return '{"intent":"exact_match","reason":"机械理由"}'


def test_blind_prompts_do_not_leak_constructed_label_or_peer_answer():
    assert set(BlindReviewInput.model_fields) == {"query_id", "text"}

    qwen = build_review_request(
        BLIND_INPUT,
        reviewer="qwen",
        qwen_image_path=QUERY.image_path,
    )
    deepseek = build_review_request(BLIND_INPUT, reviewer="deepseek")

    for request in (qwen, deepseek):
        serialized = str(request.messages)
        assert "构造标签" not in serialized
        assert "另一位审核者" not in serialized
        assert QUERY.image_path not in serialized
        for definition in ("找同款", "多商品", "发散推荐", "视觉百科", "工具型"):
            assert definition in serialized
    assert qwen.images == [QUERY.image_path]
    assert deepseek.images == []


def test_all_three_agree_marks_cross_agreed():
    result = compare_reviews(
        QUERY,
        qwen_intent="exact_match",
        deepseek_intent="exact_match",
        qwen_raw='{"intent":"exact_match","reason":"机械理由"}',
        deepseek_raw='{"intent":"exact_match","reason":"机械理由"}',
    )

    assert result.label_status == "cross_agreed"
    assert result.needs_arbitration is False


def test_review_calls_are_separate_and_use_independent_model_roles(tmp_path):
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [QUERY])
    calls: list[tuple[str, list[dict], dict]] = []
    image_root = tmp_path / "clean"
    expected_image = str((image_root / QUERY.image_path).resolve())

    def recording_chat(provider, messages, **kwargs):
        calls.append((provider, messages, kwargs))
        return '{"intent":"exact_match","reason":"机械理由"}'

    run_cross_review(
        accepted,
        output_dir=tmp_path / "labels",
        chat_fn=recording_chat,
        image_root=image_root,
        allow_provisional_asset_groups=True,
    )

    assert len(calls) == 2
    assert calls[0][0] == config.LABEL_VISION_SYNTH_PROVIDER
    assert calls[0][2] == {
        "model": config.LABEL_VISION_SYNTH_MODEL,
        "images": [expected_image],
        "temperature": 0.0,
        "json_mode": True,
    }
    assert calls[1][0] == config.LABEL_TEXT_REVIEW_PROVIDER
    assert calls[1][2] == {
        "model": config.LABEL_TEXT_REVIEW_MODEL,
        "temperature": 0.0,
        "json_mode": True,
        "thinking": False,
    }
    assert calls[0][1] is not calls[1][1]
    assert QUERY.image_path not in str(calls[0][1])
    assert QUERY.image_path not in str(calls[1][1])
    assert expected_image not in str(calls[0][1])
    assert expected_image not in str(calls[1][1])


def test_assistant_model_change_does_not_change_label_synthesis(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "ASSISTANT_MODEL", "assistant-test-sentinel")
    monkeypatch.setattr(config, "BACKBONE_MODEL", "assistant-test-sentinel")
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [QUERY])
    calls: list[tuple[str, dict]] = []

    def recording_chat(provider, _messages, **kwargs):
        calls.append((provider, kwargs))
        return '{"intent":"exact_match","reason":"机械理由"}'

    run_cross_review(
        accepted,
        output_dir=tmp_path / "labels",
        chat_fn=recording_chat,
        image_root=tmp_path / "clean",
        allow_provisional_asset_groups=True,
    )

    assert calls[0][0] == config.LABEL_VISION_SYNTH_PROVIDER
    assert calls[0][1]["model"] == config.LABEL_VISION_SYNTH_MODEL
    assert calls[0][1]["model"] != "assistant-test-sentinel"


def test_review_rejects_image_transport_path_outside_clean_root(tmp_path):
    accepted = tmp_path / "accepted-results.jsonl"
    escaped = make_query(1).model_copy(update={"image_path": "../secret.jpg"})
    write_queries(accepted, [escaped])

    with pytest.raises(ValueError, match="image_root|越界"):
        run_cross_review(
            accepted,
            output_dir=tmp_path / "labels",
            chat_fn=agreeing_chat,
            image_root=tmp_path / "clean",
            allow_provisional_asset_groups=True,
        )


def test_disagreement_and_exact_five_percent_spot_check_enter_queue(tmp_path):
    queries = [make_query(index) for index in range(1, 21)]
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, queries)

    def disagree_once(provider, messages, **kwargs):
        if (
            provider == config.LABEL_VISION_SYNTH_PROVIDER
            and "机械占位-001" in str(messages)
        ):
            return '{"intent":"multi_product","reason":"机械分歧"}'
        return '{"intent":"exact_match","reason":"机械一致"}'

    run_cross_review(
        accepted,
        output_dir=tmp_path / "labels",
        chat_fn=disagree_once,
        allow_provisional_asset_groups=True,
    )

    queue = read_jsonl(tmp_path / "labels/arbitration_queue.jsonl")
    agreed = queries[1:]
    expected_spot = min(
        agreed,
        key=lambda query: hashlib.sha256(query.query_id.encode("utf-8")).hexdigest(),
    )
    assert {item["query_id"] for item in queue} == {
        QUERY.query_id,
        expected_spot.query_id,
    }
    assert (
        next(item for item in queue if item["query_id"] == QUERY.query_id)["spot_check"]
        is False
    )
    assert (
        next(item for item in queue if item["query_id"] == expected_spot.query_id)[
            "spot_check"
        ]
        is True
    )


def test_pending_rebuilds_queue_and_rejects_removed_spot_check(tmp_path):
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [make_query(index) for index in range(1, 21)])
    labels = tmp_path / "labels"
    run_cross_review(
        accepted,
        output_dir=labels,
        chat_fn=agreeing_chat,
        allow_provisional_asset_groups=True,
    )

    queue_path = labels / "arbitration_queue.jsonl"
    manifest_path = labels / "manifest.json"
    assert len(queue_path.read_text("utf-8").splitlines()) == 1
    queue_path.write_bytes(b"")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["arbitration_queue_sha256"] = sha256_bytes(b"")
    manifest["spot_check_count"] = 0
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="确定性重建|计数"):
        pending_arbitrations(labels)


def test_parse_failure_retries_twice_and_preserves_every_raw_response(tmp_path):
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [QUERY])
    qwen_responses = iter(["not-json", "{}", '{"intent":"unknown","reason":"机械"}'])
    qwen_calls = 0

    def failing_qwen(provider, messages, **kwargs):
        nonlocal qwen_calls
        if provider == config.LABEL_VISION_SYNTH_PROVIDER:
            qwen_calls += 1
            return next(qwen_responses)
        return '{"intent":"exact_match","reason":"机械理由"}'

    run_cross_review(
        accepted,
        output_dir=tmp_path / "labels",
        chat_fn=failing_qwen,
        allow_provisional_asset_groups=True,
    )

    review = read_jsonl(tmp_path / "labels/reviews.jsonl")[0]
    queue = read_jsonl(tmp_path / "labels/arbitration_queue.jsonl")
    assert qwen_calls == 3
    assert review["qwen_raw"] == [
        "not-json",
        "{}",
        '{"intent":"unknown","reason":"机械"}',
    ]
    assert review["qwen_review_error"]
    assert queue[0]["query_id"] == QUERY.query_id


def test_review_outputs_are_derived_and_leave_accepted_bytes_unchanged(tmp_path):
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [QUERY])
    ledger = tmp_path / "accepted-ledger.jsonl"
    ledger.write_text('{"batch_id":"dev-mini-001-r1"}\n', encoding="utf-8")
    before = (accepted.read_bytes(), ledger.read_bytes())

    run_cross_review(
        accepted,
        output_dir=tmp_path / "labels",
        chat_fn=agreeing_chat,
        allow_provisional_asset_groups=True,
    )

    assert (accepted.read_bytes(), ledger.read_bytes()) == before
    manifest = json.loads((tmp_path / "labels/manifest.json").read_text("utf-8"))
    assert manifest["accepted_source_sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert (tmp_path / "labels/reviews.jsonl").is_file()
    assert (tmp_path / "labels/arbitration_queue.jsonl").is_file()
    assert (tmp_path / "labels/arbitrations.jsonl").is_file()
    assert (tmp_path / "labels/labeled_queries.jsonl").is_file()


def test_arbitration_updates_only_derived_label_and_clears_pending_item(tmp_path):
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [QUERY])
    labels = tmp_path / "labels"
    run_cross_review(
        accepted,
        output_dir=labels,
        chat_fn=agreeing_chat,
        allow_provisional_asset_groups=True,
    )
    immutable_before = {
        path.name: path.read_bytes()
        for path in (
            accepted,
            labels / "reviews.jsonl",
            labels / "arbitration_queue.jsonl",
        )
    }

    apply_arbitration(
        labels,
        query_id=QUERY.query_id,
        intent="multi_product",
        reason="机械人工裁决",
        allow_provisional_asset_groups=True,
    )

    labeled = Query.model_validate(read_jsonl(labels / "labeled_queries.jsonl")[0])
    assert labeled.gt_intent == "multi_product"
    assert labeled.label_status == "arbitrated"
    assert pending_arbitrations(labels) == []
    assert read_jsonl(labels / "arbitrations.jsonl")[0]["reason"] == "机械人工裁决"
    assert {
        path.name: path.read_bytes()
        for path in (
            accepted,
            labels / "reviews.jsonl",
            labels / "arbitration_queue.jsonl",
        )
    } == immutable_before


def test_arbitration_recomputes_card_eligibility_from_canonical_intent(tmp_path):
    query = make_query(1, intent="encyclopedia")
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [query])
    labels = tmp_path / "labels"
    run_cross_review(
        accepted,
        output_dir=labels,
        chat_fn=agreeing_chat,
        allow_provisional_asset_groups=True,
    )

    apply_arbitration(
        labels,
        query_id=query.query_id,
        intent="exact_match",
        reason="机械跨卡片资格裁决",
        allow_provisional_asset_groups=True,
    )

    labeled = Query.model_validate(read_jsonl(labels / "labeled_queries.jsonl")[0])
    assert labeled.canonical_capability == "product.exact_match"
    assert labeled.requires_card is True


def test_utility_arbitration_requires_explicit_capability_and_preserves_set(
    tmp_path,
):
    query = make_query(1, intent="utility")
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [query])
    labels = tmp_path / "labels"
    run_cross_review(
        accepted,
        output_dir=labels,
        chat_fn=agreeing_chat,
        allow_provisional_asset_groups=True,
    )

    with pytest.raises(ValueError, match="multiple capabilities|canonical_capability"):
        apply_arbitration(
            labels,
            query_id=query.query_id,
            intent="utility",
            reason="explicit subtype is required",
            allow_provisional_asset_groups=True,
        )
    assert (labels / "arbitrations.jsonl").read_bytes() == b""

    acceptable = ["utility.document_reading", "utility.recipe_guidance"]
    apply_arbitration(
        labels,
        query_id=query.query_id,
        intent="utility",
        canonical_capability="utility.recipe_guidance",
        acceptable_capabilities=acceptable,
        reason="image and request are recipe guidance",
        allow_provisional_asset_groups=True,
    )

    labeled = Query.model_validate(read_jsonl(labels / "labeled_queries.jsonl")[0])
    arbitration = read_jsonl(labels / "arbitrations.jsonl")[0]
    assert labeled.canonical_capability == "utility.recipe_guidance"
    assert labeled.acceptable_capabilities == acceptable
    assert labeled.requires_card is False
    assert labeled.label_provenance[-1].canonical_capability == (
        "utility.recipe_guidance"
    )
    assert arbitration["canonical_capability"] == "utility.recipe_guidance"
    assert arbitration["acceptable_capabilities"] == acceptable
    assert arbitration["requires_card"] is False


@pytest.mark.parametrize(
    ("canonical_capability", "acceptable_capabilities"),
    [
        ("product.exact_match", ["product.exact_match"]),
        (
            "utility.document_reading",
            ["product.exact_match", "utility.document_reading"],
        ),
    ],
)
def test_arbitration_rejects_capability_outside_target_intent(
    tmp_path,
    canonical_capability,
    acceptable_capabilities,
):
    query = make_query(1, intent="utility")
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, [query])
    labels = tmp_path / "labels"
    run_cross_review(
        accepted,
        output_dir=labels,
        chat_fn=agreeing_chat,
        allow_provisional_asset_groups=True,
    )

    with pytest.raises(ValueError, match="does not belong|capability"):
        apply_arbitration(
            labels,
            query_id=query.query_id,
            intent="utility",
            canonical_capability=canonical_capability,
            acceptable_capabilities=acceptable_capabilities,
            reason="mechanical invalid mapping",
            allow_provisional_asset_groups=True,
        )
    assert (labels / "arbitrations.jsonl").read_bytes() == b""


def test_second_arbitration_preserves_existing_record_bytes_as_prefix(tmp_path):
    queries = [make_query(1), make_query(2)]
    accepted = tmp_path / "accepted-results.jsonl"
    write_queries(accepted, queries)

    def disagreeing_chat(provider, messages, **kwargs):
        intent = (
            (
                "multi_product"
                if provider == config.LABEL_VISION_SYNTH_PROVIDER
                else "exact_match"
            )
        )
        return json.dumps({"intent": intent, "reason": "机械分歧"}, ensure_ascii=False)

    labels = tmp_path / "labels"
    run_cross_review(
        accepted,
        output_dir=labels,
        chat_fn=disagreeing_chat,
        allow_provisional_asset_groups=True,
    )
    apply_arbitration(
        labels,
        query_id=queries[0].query_id,
        intent="exact_match",
        reason="机械裁决一",
        allow_provisional_asset_groups=True,
    )
    arbitration_path = labels / "arbitrations.jsonl"
    first = json.loads(arbitration_path.read_text("utf-8"))
    first["decided_at"] = first["decided_at"].replace("+00:00", "Z")
    arbitration_path.write_text(
        json.dumps(first, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prefix = arbitration_path.read_bytes()

    apply_arbitration(
        labels,
        query_id=queries[1].query_id,
        intent="multi_product",
        reason="机械裁决二",
        allow_provisional_asset_groups=True,
    )

    after = arbitration_path.read_bytes()
    assert after.startswith(prefix)
    assert len(after.splitlines()) == 2
