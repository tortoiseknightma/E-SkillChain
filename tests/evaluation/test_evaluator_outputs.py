from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillchain.evaluation.evaluator_outputs import (
    EvaluatorOutputParseError,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V2,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
    FINAL_JUDGE_PARSER_POLICY_SHA256_V4,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V2,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V1,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V1,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
    VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_SHA256_V4,
    VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_VERSION_V4,
    feedback_output_contract_v4,
    final_judge_parser_policy_v2,
    final_judge_parser_policy_v3,
    final_judge_parser_policy_v4,
    parse_final_judge_output,
    parse_final_judge_output_v2,
    parse_final_judge_output_v3,
    parse_final_judge_output_v4,
    parse_visual_feedback_output,
    parse_visual_feedback_output_v2,
    parse_visual_feedback_output_v3,
    visual_feedback_parser_policy_v1,
    visual_feedback_parser_policy_v2,
    visual_feedback_parser_policy_v3,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


_V3_RAW_FAILURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "evaluation"
    / "portfolio_s1_feedback_v3_raw_failures.json"
)


def _final_payload(*, requires_card: bool = False) -> dict:
    dimensions = [
        {"dimension": "CA", "score": 8},
        {"dimension": "CQ", "score": 14},
        {"dimension": "TCR", "score": 9},
    ]
    if requires_card:
        dimensions.insert(1, {"dimension": "CCC", "score": 7})
    return {
        "schema_version": 1,
        "requires_card": requires_card,
        "dimensions": dimensions,
    }


def _feedback_payload() -> dict:
    return {
        "schema_version": 1,
        "summary": "The answer missed visible product evidence.",
        "rule_violations": [],
        "ideal_response_gaps": [],
        "skill_suggestions": [],
    }


def _feedback_finding() -> dict:
    return {
        "dimension": "TCR",
        "severity": "high",
        "grounded_in_image": False,
        "description": "Internal tool line identifiers leaked.",
        "evidence": ["tool-call-1-line-1"],
    }


def _v2_final_payload(
    shape: str,
    *,
    requires_card: bool = False,
) -> dict:
    canonical = _final_payload(requires_card=requires_card)
    assessments = canonical["dimensions"]
    if shape == "assessment_array":
        dimensions = assessments
    elif shape == "score_mapping":
        dimensions = {item["dimension"]: item["score"] for item in assessments}
    elif shape == "dimension_score_pairs":
        dimensions = [[item["dimension"], item["score"]] for item in assessments]
    else:
        raise AssertionError(f"unsupported test shape: {shape}")
    return {
        "schema_version": 1,
        "requires_card": requires_card,
        "dimensions": dimensions,
    }


def _v3_bare_payload(
    shape: str,
    *,
    requires_card: bool = False,
) -> object:
    canonical = _final_payload(requires_card=requires_card)
    assessments = canonical["dimensions"]
    if shape == "bare_assessment_array":
        return assessments
    if shape == "bare_score_mapping":
        return {item["dimension"]: item["score"] for item in assessments}
    if shape == "bare_dimension_score_pairs":
        return [[item["dimension"], item["score"]] for item in assessments]
    raise AssertionError(f"unsupported test shape: {shape}")


def _set_v3_bare_score(
    payload: object,
    shape: str,
    *,
    dimension: str,
    score: object,
) -> None:
    if shape == "bare_assessment_array":
        assert isinstance(payload, list)
        item = next(item for item in payload if item["dimension"] == dimension)
        item["score"] = score
        return
    if shape == "bare_score_mapping":
        assert isinstance(payload, dict)
        payload[dimension] = score
        return
    if shape == "bare_dimension_score_pairs":
        assert isinstance(payload, list)
        item = next(item for item in payload if item[0] == dimension)
        item[1] = score
        return
    raise AssertionError(f"unsupported test shape: {shape}")


def test_final_judge_output_accepts_exact_card_and_no_card_contracts() -> None:
    no_card = parse_final_judge_output(
        json.dumps(_final_payload(), separators=(",", ":")),
        expected_requires_card=False,
    )
    card = parse_final_judge_output(
        json.dumps(_final_payload(requires_card=True), separators=(",", ":")),
        expected_requires_card=True,
    )

    assert [item.dimension for item in no_card.dimensions] == ["CA", "CQ", "TCR"]
    assert [item.dimension for item in card.dimensions] == [
        "CA",
        "CCC",
        "CQ",
        "TCR",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "```json\n{}\n```",
        '{"schema_version":1,"requires_card":false,"dimensions":[]}\ntrailing',
        '{"schema_version":1,"schema_version":1,"requires_card":false,"dimensions":[]}',
        "[]",
        '{"requires_card":false,"dimensions":[]}',
        '{"schema_version":1,"requires_card":false,"dimensions":[],"extra":true}',
        '{"schema_version":1,"requires_card":false,"dimensions":['
        '{"dimension":"CA","score":"8"},{"dimension":"CQ","score":14},'
        '{"dimension":"TCR","score":9}]}',
        '{"schema_version":1,"requires_card":false,"dimensions":['
        '{"dimension":"CA","score":8.0},{"dimension":"CQ","score":14},'
        '{"dimension":"TCR","score":9}]}',
        '{"schema_version":1,"requires_card":false,"dimensions":['
        '{"dimension":"CQ","score":14},{"dimension":"CA","score":8},'
        '{"dimension":"TCR","score":9}]}',
        '{"schema_version":1,"requires_card":false,"dimensions":['
        '{"dimension":"CA","score":11},{"dimension":"CQ","score":14},'
        '{"dimension":"TCR","score":9}]}',
    ],
)
def test_final_judge_output_rejects_nonexact_responses(text: str) -> None:
    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output(text, expected_requires_card=False)


def test_final_judge_output_rejects_card_applicability_mismatch() -> None:
    text = json.dumps(_final_payload(), separators=(",", ":"))
    with pytest.raises(EvaluatorOutputParseError, match="requires_card"):
        parse_final_judge_output(text, expected_requires_card=True)


@pytest.mark.parametrize("shape", ["score_mapping", "dimension_score_pairs"])
def test_legacy_final_judge_parser_rejects_forward_shapes(shape: str) -> None:
    text = json.dumps(_v2_final_payload(shape), separators=(",", ":"))

    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output(text, expected_requires_card=False)


@pytest.mark.parametrize("requires_card", [False, True])
@pytest.mark.parametrize(
    ("shape", "expected_shape"),
    [
        ("assessment_array", "assessment_array"),
        ("score_mapping", "score_mapping"),
        ("dimension_score_pairs", "dimension_score_pairs"),
    ],
)
def test_v2_final_judge_parser_normalizes_equivalent_shapes(
    shape: str,
    expected_shape: str,
    requires_card: bool,
) -> None:
    payload = _v2_final_payload(shape, requires_card=requires_card)
    parsed = parse_final_judge_output_v2(
        json.dumps(payload, separators=(",", ":")),
        expected_requires_card=requires_card,
    )

    assert parsed.submission.model_dump(mode="json") == _final_payload(
        requires_card=requires_card
    )
    assert parsed.raw_dimensions_shape == expected_shape
    assert parsed.policy_version == FINAL_JUDGE_PARSER_POLICY_VERSION_V2
    assert parsed.policy_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V2


@pytest.mark.parametrize(
    ("shape", "illegal_score"),
    [
        ("assessment_array", True),
        ("assessment_array", 8.0),
        ("assessment_array", "8"),
        ("score_mapping", True),
        ("score_mapping", 8.0),
        ("score_mapping", "8"),
        ("dimension_score_pairs", True),
        ("dimension_score_pairs", 8.0),
        ("dimension_score_pairs", "8"),
    ],
)
def test_v2_final_judge_parser_rejects_noninteger_scores(
    shape: str,
    illegal_score: object,
) -> None:
    payload = _v2_final_payload(shape)
    if shape == "assessment_array":
        payload["dimensions"][0]["score"] = illegal_score
    elif shape == "score_mapping":
        payload["dimensions"]["CA"] = illegal_score
    else:
        payload["dimensions"][0][1] = illegal_score

    with pytest.raises(EvaluatorOutputParseError, match="strict JSON integer"):
        parse_final_judge_output_v2(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=False,
        )


@pytest.mark.parametrize(
    "shape", ["assessment_array", "score_mapping", "dimension_score_pairs"]
)
@pytest.mark.parametrize("illegal_score", [-1, 11])
def test_v2_final_judge_parser_rejects_scores_outside_inclusive_bounds(
    shape: str,
    illegal_score: int,
) -> None:
    payload = _v2_final_payload(shape)
    if shape == "assessment_array":
        payload["dimensions"][0]["score"] = illegal_score
    elif shape == "score_mapping":
        payload["dimensions"]["CA"] = illegal_score
    else:
        payload["dimensions"][0][1] = illegal_score

    with pytest.raises(EvaluatorOutputParseError, match="inclusive maximum"):
        parse_final_judge_output_v2(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=False,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": {"CQ": 14, "CA": 8, "TCR": 9},
        },
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": [["CQ", 14], ["CA", 8], ["TCR", 9]],
        },
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": [
                {"dimension": "CA", "score": 8, "extra": 0},
                {"dimension": "CQ", "score": 14},
                {"dimension": "TCR", "score": 9},
            ],
        },
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": [["CA", 8, "extra"], ["CQ", 14], ["TCR", 9]],
        },
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": {"CA": 8, "CQ": 14},
        },
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": {"CA": 8, "CCC": 7, "CQ": 14, "TCR": 9},
        },
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": {"CA": 8, "CQ": 14, "TCR": 9},
            "extra": True,
        },
        {
            "schema_version": True,
            "requires_card": False,
            "dimensions": {"CA": 8, "CQ": 14, "TCR": 9},
        },
    ],
)
def test_v2_final_judge_parser_rejects_illegal_shape_boundaries(
    payload: dict,
) -> None:
    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output_v2(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=False,
        )


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"schema_version":1}\n```',
        '{"schema_version":1,"requires_card":false,"dimensions":{'
        '"CA":8,"CQ":14,"TCR":9}} trailing',
        '{"schema_version":1,"schema_version":1,"requires_card":false,'
        '"dimensions":{"CA":8,"CQ":14,"TCR":9}}',
        '{"schema_version":1,"requires_card":false,"dimensions":{'
        '"CA":8,"CA":9,"CQ":14,"TCR":9}}',
    ],
)
def test_v2_final_judge_parser_does_not_repair_nonexact_json(text: str) -> None:
    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output_v2(text, expected_requires_card=False)


def test_v2_final_judge_parser_policy_hash_is_stable() -> None:
    assert (
        FINAL_JUDGE_PARSER_POLICY_SHA256_V2
        == "d151456396bd1bef560de116dc3eb83e94cfcb3fe686ac3ec79fb429e89d2509"
    )
    assert FINAL_JUDGE_PARSER_POLICY_SHA256_V2 == sha256_bytes(
        canonical_json_bytes(final_judge_parser_policy_v2())
    )


@pytest.mark.parametrize("requires_card", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        "assessment_array",
        "score_mapping",
        "dimension_score_pairs",
        "bare_assessment_array",
        "bare_score_mapping",
        "bare_dimension_score_pairs",
    ],
)
def test_v3_final_judge_parser_normalizes_wrapped_and_bare_shapes(
    shape: str,
    requires_card: bool,
) -> None:
    if shape.startswith("bare_"):
        payload = _v3_bare_payload(shape, requires_card=requires_card)
    else:
        payload = _v2_final_payload(shape, requires_card=requires_card)

    parsed = parse_final_judge_output_v3(
        json.dumps(payload, separators=(",", ":")),
        expected_requires_card=requires_card,
    )

    assert parsed.submission.model_dump(mode="json") == _final_payload(
        requires_card=requires_card
    )
    assert parsed.raw_dimensions_shape == shape
    assert parsed.policy_version == FINAL_JUDGE_PARSER_POLICY_VERSION_V3
    assert parsed.policy_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V3


@pytest.mark.parametrize(
    "shape",
    [
        "bare_assessment_array",
        "bare_score_mapping",
        "bare_dimension_score_pairs",
    ],
)
def test_v2_final_judge_parser_still_rejects_v3_bare_shapes(
    shape: str,
) -> None:
    payload = _v3_bare_payload(shape)

    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output_v2(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=False,
        )


@pytest.mark.parametrize(
    "shape",
    [
        "bare_assessment_array",
        "bare_score_mapping",
        "bare_dimension_score_pairs",
    ],
)
@pytest.mark.parametrize("illegal_score", [True, 8.0, "8"])
def test_v3_bare_shapes_reject_noninteger_scores(
    shape: str,
    illegal_score: object,
) -> None:
    payload = _v3_bare_payload(shape)
    _set_v3_bare_score(
        payload,
        shape,
        dimension="CA",
        score=illegal_score,
    )

    with pytest.raises(EvaluatorOutputParseError, match="strict JSON integer"):
        parse_final_judge_output_v3(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=False,
        )


@pytest.mark.parametrize(
    "shape",
    [
        "bare_assessment_array",
        "bare_score_mapping",
        "bare_dimension_score_pairs",
    ],
)
@pytest.mark.parametrize(
    ("dimension", "illegal_score"),
    [
        ("CA", -1),
        ("CA", 11),
        ("CCC", 11),
        ("CQ", 21),
        ("TCR", 11),
    ],
)
def test_v3_bare_shapes_enforce_each_dimension_maximum(
    shape: str,
    dimension: str,
    illegal_score: int,
) -> None:
    payload = _v3_bare_payload(shape, requires_card=True)
    _set_v3_bare_score(
        payload,
        shape,
        dimension=dimension,
        score=illegal_score,
    )

    with pytest.raises(EvaluatorOutputParseError, match="inclusive maximum"):
        parse_final_judge_output_v3(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=True,
        )


@pytest.mark.parametrize(
    "shape",
    [
        "bare_assessment_array",
        "bare_score_mapping",
        "bare_dimension_score_pairs",
    ],
)
def test_v3_bare_shapes_accept_inclusive_dimension_maxima(shape: str) -> None:
    payload = _v3_bare_payload(shape, requires_card=True)
    for dimension, score in {"CA": 10, "CCC": 10, "CQ": 20, "TCR": 10}.items():
        _set_v3_bare_score(
            payload,
            shape,
            dimension=dimension,
            score=score,
        )

    parsed = parse_final_judge_output_v3(
        json.dumps(payload, separators=(",", ":")),
        expected_requires_card=True,
    )

    assert [item.score for item in parsed.submission.dimensions] == [
        10,
        10,
        20,
        10,
    ]


@pytest.mark.parametrize(
    ("payload", "expected_requires_card"),
    [
        ({"CQ": 14, "CA": 8, "TCR": 9}, False),
        ({"CA": 8, "CQ": 14}, False),
        ({"CA": 8, "CCC": 7, "CQ": 14, "TCR": 9}, False),
        (
            [
                {"dimension": "CQ", "score": 14},
                {"dimension": "CA", "score": 8},
                {"dimension": "TCR", "score": 9},
            ],
            False,
        ),
        ([["CQ", 14], ["CA", 8], ["TCR", 9]], False),
        (
            [
                {"dimension": "CA", "score": 8},
                {"dimension": "CQ", "score": 14},
                {"dimension": "TCR", "score": 9},
            ],
            True,
        ),
        (
            [
                {"dimension": "CA", "score": 8},
                {"dimension": "CCC", "score": 7},
                {"dimension": "CQ", "score": 14},
                {"dimension": "TCR", "score": 9},
            ],
            False,
        ),
    ],
)
def test_v3_bare_shapes_require_exact_applicable_set_and_order(
    payload: object,
    expected_requires_card: bool,
) -> None:
    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output_v3(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=expected_requires_card,
        )


@pytest.mark.parametrize(
    "payload",
    [
        [
            {"dimension": "CA", "score": 8, "extra": 0},
            {"dimension": "CQ", "score": 14},
            {"dimension": "TCR", "score": 9},
        ],
        [["CA", 8, "extra"], ["CQ", 14], ["TCR", 9]],
        {"CA": 8, "CQ": 14, "TCR": 9, "extra": 0},
        [],
        [{"dimension": "CA", "score": 8}, ["CQ", 14], ["TCR", 9]],
        8,
        "scores",
        None,
    ],
)
def test_v3_bare_shapes_reject_extra_or_nonexact_structure(
    payload: object,
) -> None:
    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output_v3(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=False,
        )


def test_v3_wrapped_shape_rejects_requires_card_mismatch() -> None:
    payload = _v2_final_payload("score_mapping", requires_card=False)

    with pytest.raises(EvaluatorOutputParseError, match="requires_card"):
        parse_final_judge_output_v3(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=True,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": {"CA": 8, "CQ": 14, "TCR": 9},
            "extra": 0,
        },
        {
            "schema_version": True,
            "requires_card": False,
            "dimensions": {"CA": 8, "CQ": 14, "TCR": 9},
        },
        {
            "schema_version": 1,
            "requires_card": 0,
            "dimensions": {"CA": 8, "CQ": 14, "TCR": 9},
        },
    ],
)
def test_v3_wrapped_shapes_reject_extra_or_nonstrict_envelope_fields(
    payload: dict,
) -> None:
    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output_v3(
            json.dumps(payload, separators=(",", ":")),
            expected_requires_card=False,
        )


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"CA":8,"CQ":14,"TCR":9}\n```',
        '{"CA":8,"CQ":14,"TCR":9} trailing',
        '{"CA":8,"CQ":14,"TCR":9}{"CA":8,"CQ":14,"TCR":9}',
        '{"CA":8,"CA":9,"CQ":14,"TCR":9}',
        '[{"dimension":"CA","score":8,"score":9},'
        '{"dimension":"CQ","score":14},{"dimension":"TCR","score":9}]',
        '{"CA":8,"CQ":NaN,"TCR":9}',
    ],
)
def test_v3_final_judge_parser_does_not_repair_nonexact_json(text: str) -> None:
    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output_v3(
            text,
            expected_requires_card=False,
        )


def test_v3_rejects_nonboolean_packet_applicability() -> None:
    text = '{"CA":8,"CQ":14,"TCR":9}'

    with pytest.raises(TypeError, match="strict boolean"):
        parse_final_judge_output_v3(
            text,
            expected_requires_card=0,  # type: ignore[arg-type]
        )


def test_v4_accepts_dm_109_complete_json_fence() -> None:
    text = '```json\n{"CA":8,"CQ":14,"TCR":9}\n```'

    parsed = parse_final_judge_output_v4(
        text,
        expected_requires_card=False,
    )

    assert parsed.submission.model_dump(mode="json") == _final_payload()
    assert parsed.raw_dimensions_shape == "bare_score_mapping"
    assert parsed.input_wrapper == "markdown_json_fence"
    assert parsed.policy_version == FINAL_JUDGE_PARSER_POLICY_VERSION_V4
    assert parsed.policy_sha256 == FINAL_JUDGE_PARSER_POLICY_SHA256_V4


@pytest.mark.parametrize(
    ("text", "expected_wrapper"),
    [
        ('{"CA":8,"CQ":14,"TCR":9}', "none"),
        ('```\n{"CA":8,"CQ":14,"TCR":9}\n```', "markdown_fence"),
        (
            '```json\r\n{"CA":8,"CQ":14,"TCR":9}\r\n```',
            "markdown_json_fence",
        ),
    ],
)
def test_v4_accepts_only_supported_complete_outer_wrappers(
    text: str,
    expected_wrapper: str,
) -> None:
    parsed = parse_final_judge_output_v4(
        text,
        expected_requires_card=False,
    )

    assert parsed.input_wrapper == expected_wrapper
    assert parsed.raw_dimensions_shape == "bare_score_mapping"


@pytest.mark.parametrize(
    "text",
    [
        'prose\n```json\n{"CA":8,"CQ":14,"TCR":9}\n```',
        '```json\n{"CA":8,"CQ":14,"TCR":9}\n```\ntrailing',
        '```json\n{"CA":8,"CQ":14,"TCR":9}\n',
        (
            '```json\n{"CA":8,"CQ":14,"TCR":9}\n```\n'
            '```json\n{"CA":8,"CQ":14,"TCR":9}\n```'
        ),
        '```json\n```\n{"CA":8,"CQ":14,"TCR":9}\n```\n```',
        '```JSON\n{"CA":8,"CQ":14,"TCR":9}\n```',
        '```json\n{"CA":8,"CQ":14,"TCR":9} trailing\n```',
        '```json\n{"CA":8,"CA":9,"CQ":14,"TCR":9}\n```',
        '```json\n{"CA":8,"CQ":NaN,"TCR":9}\n```',
        '```json\n{"CA":8,"CQ":14,"TCR":9,"extra":0}\n```',
    ],
)
def test_v4_rejects_nonexclusive_fences_and_nonexact_v3_payloads(
    text: str,
) -> None:
    with pytest.raises(EvaluatorOutputParseError):
        parse_final_judge_output_v4(
            text,
            expected_requires_card=False,
        )


def test_v3_final_judge_parser_policy_hash_is_stable() -> None:
    assert (
        FINAL_JUDGE_PARSER_POLICY_SHA256_V3
        == "cb71705fe5a1ce97c8e4e409df91b780efc436a70194f998578becaa0ec3dda0"
    )
    assert FINAL_JUDGE_PARSER_POLICY_SHA256_V3 == sha256_bytes(
        canonical_json_bytes(final_judge_parser_policy_v3())
    )
    assert (
        FINAL_JUDGE_PARSER_POLICY_SHA256_V2
        == "d151456396bd1bef560de116dc3eb83e94cfcb3fe686ac3ec79fb429e89d2509"
    )
    assert (
        FINAL_JUDGE_PARSER_POLICY_SHA256_V4
        == "b3f62f44131fe1b87904a37cb23cddd20234a95e380349638c6bf692aeca181c"
    )
    assert FINAL_JUDGE_PARSER_POLICY_SHA256_V4 == sha256_bytes(
        canonical_json_bytes(final_judge_parser_policy_v4())
    )


@pytest.mark.parametrize(
    ("opening", "newline"),
    [
        ("```", "\n"),
        ("```json", "\n"),
        ("```json", "\r\n"),
    ],
)
def test_visual_feedback_v2_accepts_one_complete_outer_fence(
    opening: str,
    newline: str,
) -> None:
    body = json.dumps(_feedback_payload(), separators=(",", ":"))
    text = f"{opening}{newline}{body}{newline}```"

    parsed = parse_visual_feedback_output_v2(text)

    assert parsed.summary == _feedback_payload()["summary"]
    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output(text)


@pytest.mark.parametrize(
    "text_factory",
    [
        lambda body: f"prose\n```json\n{body}\n```",
        lambda body: f"```json\n{body}\n```\n",
        lambda body: f"```json\n{body}\n```\ntrailing",
        lambda body: f"```json\n{body}\n",
        lambda body: f"```JSON\n{body}\n```",
        lambda body: f"``` json\n{body}\n```",
        lambda body: f"```json\n```\n{body}\n```\n```",
        lambda body: f"```json\n{body}\n```\n```json\n{body}\n```",
        lambda body: f"```json\n{body} ```\n```",
    ],
)
def test_visual_feedback_v2_rejects_nonexact_fence_boundaries(
    text_factory,
) -> None:
    body = json.dumps(_feedback_payload(), separators=(",", ":"))

    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v2(text_factory(body))


@pytest.mark.parametrize("mutation", ["duplicate_key", "nonfinite"])
def test_visual_feedback_v2_preserves_strict_json_rules(mutation: str) -> None:
    body = json.dumps(_feedback_payload(), separators=(",", ":"))
    if mutation == "duplicate_key":
        body = body.replace(
            '"schema_version":1',
            '"schema_version":1,"schema_version":1',
            1,
        )
    else:
        body = body.replace(
            '"skill_suggestions":[]',
            '"skill_suggestions":[NaN]',
            1,
        )

    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v2(f"```json\n{body}\n```")


def test_visual_feedback_v3_normalizes_exact_v2_leading_space_failure() -> None:
    payload = _feedback_payload()
    payload["summary"] = " sensitivity/formatting error"
    payload["rule_violations"] = [
        {
            "dimension": "TCR",
            "severity": "high",
            "grounded_in_image": False,
            "description": " Internal tool line identifiers leaked.",
            "evidence": ["tool-call-1-line-1"],
        },
        {
            "dimension": "CQ",
            "severity": "medium",
            "grounded_in_image": False,
            "description": " The answer did not match the user's language.",
            "evidence": ["answer in English"],
        },
    ]
    payload["ideal_response_gaps"] = [
        {
            "dimension": "CA",
            "severity": "medium",
            "grounded_in_image": True,
            "description": " The answer should summarize the receipt clearly.",
            "evidence": ["24.000", "26.000"],
        }
    ]
    body = json.dumps(payload, separators=(",", ":"))
    text = f"```json\n{body}\n```"

    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v2(text)
    parsed = parse_visual_feedback_output_v3(text)

    assert parsed.summary == "sensitivity/formatting error"
    assert [item.description for item in parsed.rule_violations] == [
        "Internal tool line identifiers leaked.",
        "The answer did not match the user's language.",
    ]
    assert parsed.ideal_response_gaps[0].description == (
        "The answer should summarize the receipt clearly."
    )


def test_visual_feedback_v3_trims_only_allowlisted_free_text_leaves() -> None:
    payload = _feedback_payload()
    payload["summary"] = "\t summary \n"
    payload["rule_violations"] = [_feedback_finding()]
    finding = payload["rule_violations"][0]
    finding["description"] = " description "
    finding["evidence"] = [" evidence ", "\tsecond\n"]
    payload["skill_suggestions"] = [" suggestion "]
    text = json.dumps(payload, separators=(",", ":"))

    parsed = parse_visual_feedback_output_v3(text)

    assert parsed.summary == "summary"
    assert parsed.rule_violations[0].description == "description"
    assert parsed.rule_violations[0].evidence == ("evidence", "second")
    assert parsed.skill_suggestions == ("suggestion",)

    payload["rule_violations"][0]["dimension"] = " TCR"
    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v3(json.dumps(payload, separators=(",", ":")))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", " \t\n"),
        ("description", " \t\n"),
        ("evidence", " \t\n"),
        ("skill_suggestions", " \t\n"),
    ],
)
def test_visual_feedback_v3_rejects_empty_after_trim(
    field: str,
    value: str,
) -> None:
    payload = _feedback_payload()
    payload["rule_violations"] = [_feedback_finding()]
    if field == "summary":
        payload[field] = value
    elif field == "description":
        payload["rule_violations"][0][field] = value
    elif field == "evidence":
        payload["rule_violations"][0][field] = [value]
    else:
        payload[field] = [value]

    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v3(json.dumps(payload, separators=(",", ":")))


def test_visual_feedback_v3_rejects_duplicates_and_schema_drift() -> None:
    payload = _feedback_payload()
    payload["skill_suggestions"] = ["same", " same "]
    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v3(json.dumps(payload, separators=(",", ":")))

    body = json.dumps(_feedback_payload(), separators=(",", ":"))
    duplicate = body.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v3(duplicate)

    drifted = _feedback_payload()
    drifted["unexpected"] = " value "
    with pytest.raises(EvaluatorOutputParseError):
        parse_visual_feedback_output_v3(json.dumps(drifted, separators=(",", ":")))


def test_visual_feedback_v3_rejects_both_exact_terminal_v3_raw_responses() -> None:
    rows = json.loads(_V3_RAW_FAILURES.read_text(encoding="utf-8"))

    assert [row["query_id"] for row in rows] == ["r2-core-0202", "r2-core-0945"]
    for row in rows:
        raw = row["raw_response_text"]
        assert len(raw.encode("utf-8")) == row["raw_response_bytes"]
        assert sha256_bytes(raw.encode("utf-8")) == row["raw_response_sha256"]
        with pytest.raises(EvaluatorOutputParseError):
            parse_visual_feedback_output_v3(raw)

    object_suggestions = json.loads(rows[0]["raw_response_text"])
    echoed_contract = json.loads(rows[1]["raw_response_text"])
    assert isinstance(object_suggestions["skill_suggestions"][0], dict)
    assert "output_contract" in echoed_contract


def test_visual_feedback_v4_output_contract_is_exact_shape_only() -> None:
    contract = feedback_output_contract_v4()

    assert (
        VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_VERSION_V4
        == "visual-feedback-exact-shape-output-contract-v4"
    )
    assert (
        VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_SHA256_V4
        == "da9c639f0e7e179642ec32471c29bc38238d8e6d150c92e5a67f9ef38cd9d5c6"
    )
    assert VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_SHA256_V4 == sha256_bytes(
        canonical_json_bytes(contract)
    )
    assert contract["top_level_fields_exactly_once"] == [
        "schema_version",
        "summary",
        "rule_violations",
        "ideal_response_gaps",
        "skill_suggestions",
    ]
    assert contract["finding_fields_exactly_once"] == [
        "dimension",
        "severity",
        "grounded_in_image",
        "description",
        "evidence",
    ]
    assert contract["skill_suggestions_item_schema"] == {
        "type": "string",
        "nonblank_after_trim": True,
    }
    rules = " ".join(contract["rules"])
    assert "Do not copy or echo the input field named output_contract" in rules
    assert "never finding objects" in rules


def test_visual_feedback_parser_policy_hashes_are_stable() -> None:
    assert VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V1 == "visual-feedback-exact-json-v1"
    assert (
        VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2
        == "visual-feedback-complete-fence-wrapper-v2"
    )
    assert (
        VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3 == "visual-feedback-free-text-trim-v3"
    )
    assert (
        VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V1
        == "5425a19a69e345ebe3b33864f774b32b3d2a5aa6f88520be25f7ae925d194ec2"
    )
    assert (
        VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2
        == "09196a532b6369a24bebabfcc1916c6714a9066e127037c97ef9b022343fa9ee"
    )
    assert (
        VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
        == "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
    )
    assert VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V1 == sha256_bytes(
        canonical_json_bytes(visual_feedback_parser_policy_v1())
    )
    assert VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2 == sha256_bytes(
        canonical_json_bytes(visual_feedback_parser_policy_v2())
    )
    assert VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3 == sha256_bytes(
        canonical_json_bytes(visual_feedback_parser_policy_v3())
    )


def test_visual_feedback_output_requires_grounded_structured_evidence() -> None:
    payload = {
        "schema_version": 1,
        "summary": "回答遗漏了图中可见的颜色差异。",
        "rule_violations": [
            {
                "dimension": "CA",
                "severity": "high",
                "grounded_in_image": True,
                "description": "颜色描述与图片不符。",
                "evidence": ["图片主体为红色，回答称为蓝色。"],
            }
        ],
        "ideal_response_gaps": [],
        "skill_suggestions": ["回答前先核对可见属性。"],
    }
    parsed = parse_visual_feedback_output(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )

    assert parsed.rule_violations[0].grounded_in_image is True
    assert parsed.skill_suggestions == ("回答前先核对可见属性。",)
