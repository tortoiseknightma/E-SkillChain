"""Strict model-visible output contracts for visual evaluators.

Every provider response remains subject to the local frozen parser even when a
gateway offers JSON-object framing.  The final-Judge v4 parser also tolerates
one complete outer Markdown code fence while preserving the historical v1-v3
semantics and immutable Kimi-era receipts.
"""

from __future__ import annotations

import json
import re
from typing import Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


JudgeDimension = Literal["TCR", "CCC", "CQ", "CA"]
FeedbackDimension = Literal["TCR", "CCC", "CQ", "CA", "routing", "tool_use"]
FeedbackSeverity = Literal["low", "medium", "high"]
FinalJudgeDimensionsShape = Literal[
    "assessment_array",
    "score_mapping",
    "dimension_score_pairs",
]
FinalJudgeDimensionsShapeV3 = Literal[
    "assessment_array",
    "score_mapping",
    "dimension_score_pairs",
    "bare_assessment_array",
    "bare_score_mapping",
    "bare_dimension_score_pairs",
]

_DIMENSION_MAX = {"TCR": 10, "CCC": 10, "CQ": 20, "CA": 10}
_MAX_RESPONSE_BYTES = 256 * 1024
FINAL_JUDGE_PARSER_POLICY_VERSION_V2 = "final-judge-equivalent-shapes-v2"
FINAL_JUDGE_PARSER_POLICY_VERSION_V3 = "final-judge-semantic-score-shapes-v3"
FINAL_JUDGE_PARSER_POLICY_VERSION_V4 = "final-judge-complete-fence-wrapper-v4"
VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V1 = "visual-feedback-exact-json-v1"
VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2 = "visual-feedback-complete-fence-wrapper-v2"
VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3 = "visual-feedback-free-text-trim-v3"


def visual_feedback_parser_policy_v1() -> dict[str, object]:
    """Return the historical strict Feedback JSON parser policy."""

    return {
        "policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V1,
        "input_format": "exactly_one_json_object_no_markdown_or_repair",
        "normalization_target": "VisualFeedbackOutput",
        "rules": [
            "Reject duplicate keys and non-finite JSON numbers.",
            "Reject trailing content, coercion, repair, and schema drift.",
            "Apply the strict frozen VisualFeedbackOutput schema.",
        ],
    }


VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V1 = sha256_bytes(
    canonical_json_bytes(visual_feedback_parser_policy_v1())
)


def visual_feedback_parser_policy_v2() -> dict[str, object]:
    """Return the frozen Feedback policy that adds one complete outer fence."""

    return {
        "policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2,
        "base_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V1,
        "base_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V1,
        "input_format": "one_strict_json_object_with_optional_complete_outer_fence",
        "accepted_outer_wrappers": [
            "none",
            "single_unlabelled_triple_backtick_fence",
            "single_lowercase_json_triple_backtick_fence",
        ],
        "fence_rules": [
            "The opening and closing fence delimiters must each occupy their own line.",
            "The opening delimiter must be exactly ``` or ```json.",
            "No whitespace, prose, or other content may occur outside the fence.",
            "Reject incomplete, nested, or multiple fences.",
            "After removing the one outer fence, apply the complete v1 policy unchanged.",
        ],
    }


VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2 = sha256_bytes(
    canonical_json_bytes(visual_feedback_parser_policy_v2())
)


def visual_feedback_parser_policy_v3() -> dict[str, object]:
    """Return the frozen policy for narrowly normalized Feedback text leaves."""

    return {
        "policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        "base_policy_version": VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2,
        "base_policy_sha256": VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2,
        "normalization_target": "VisualFeedbackOutput",
        "normalization_operation": "python_str_strip",
        "normalized_free_text_paths": [
            "summary",
            "rule_violations[*].description",
            "rule_violations[*].evidence[*]",
            "ideal_response_gaps[*].description",
            "ideal_response_gaps[*].evidence[*]",
            "skill_suggestions[*]",
        ],
        "rules": [
            "Decode with the complete v2 wrapper and strict JSON rules.",
            "Apply str.strip() only to string values at the listed free-text paths.",
            "Do not normalize enum, boolean, numeric, container, unknown, or other values.",
            "Reject any normalized free-text value that is empty.",
            "Reject duplicate normalized evidence or skill-suggestion strings.",
            "Apply the strict frozen VisualFeedbackOutput schema after normalization.",
            "Reject all other schema drift, coercion, repair, and trailing content.",
        ],
    }


VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3 = sha256_bytes(
    canonical_json_bytes(visual_feedback_parser_policy_v3())
)


def final_judge_parser_policy_v2() -> dict[str, object]:
    """Return the frozen forward parser policy committed by its SHA-256."""

    return {
        "policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V2,
        "input_format": "exactly_one_json_object_no_markdown_or_repair",
        "top_level_fields": ["schema_version", "requires_card", "dimensions"],
        "applicable_dimensions": {
            "requires_card_false": ["CA", "CQ", "TCR"],
            "requires_card_true": ["CA", "CCC", "CQ", "TCR"],
        },
        "dimension_maxima": {"CA": 10, "CCC": 10, "CQ": 20, "TCR": 10},
        "accepted_dimensions_shapes": [
            {
                "shape": "assessment_array",
                "item_fields": ["dimension", "score"],
            },
            {
                "shape": "score_mapping",
                "keys": "exact_applicable_dimensions_in_alphabetical_order",
            },
            {
                "shape": "dimension_score_pairs",
                "item_form": ["dimension", "score"],
            },
        ],
        "score_type": "json_integer_excluding_boolean",
        "normalization_target": "FinalJudgeOutput",
        "rules": [
            "Reject missing, duplicate, or extra keys.",
            "Reject missing, duplicate, extra, or out-of-order dimensions.",
            "Reject scores outside each dimension's inclusive maximum.",
            "Reject fences, trailing content, coercion, and result repair.",
        ],
    }


FINAL_JUDGE_PARSER_POLICY_SHA256_V2 = sha256_bytes(
    canonical_json_bytes(final_judge_parser_policy_v2())
)


def final_judge_parser_policy_v3() -> dict[str, object]:
    """Return the frozen parser policy for wrapped and bare score shapes."""

    return {
        "policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
        "input_format": "exactly_one_json_value_no_markdown_or_repair",
        "applicable_dimensions": {
            "requires_card_false": ["CA", "CQ", "TCR"],
            "requires_card_true": ["CA", "CCC", "CQ", "TCR"],
        },
        "dimension_maxima": {"CA": 10, "CCC": 10, "CQ": 20, "TCR": 10},
        "accepted_shapes": [
            {
                "shape": "assessment_array",
                "envelope_fields": [
                    "schema_version",
                    "requires_card",
                    "dimensions",
                ],
                "dimension_item_fields": ["dimension", "score"],
            },
            {
                "shape": "score_mapping",
                "envelope_fields": [
                    "schema_version",
                    "requires_card",
                    "dimensions",
                ],
                "dimension_keys": ("exact_applicable_dimensions_in_alphabetical_order"),
            },
            {
                "shape": "dimension_score_pairs",
                "envelope_fields": [
                    "schema_version",
                    "requires_card",
                    "dimensions",
                ],
                "dimension_item_form": ["dimension", "score"],
            },
            {
                "shape": "bare_assessment_array",
                "item_fields": ["dimension", "score"],
                "requires_card_source": "bound_evaluation_packet",
            },
            {
                "shape": "bare_score_mapping",
                "keys": "exact_applicable_dimensions_in_alphabetical_order",
                "requires_card_source": "bound_evaluation_packet",
            },
            {
                "shape": "bare_dimension_score_pairs",
                "item_form": ["dimension", "score"],
                "requires_card_source": "bound_evaluation_packet",
            },
        ],
        "score_type": "json_integer_excluding_boolean",
        "normalization_target": "FinalJudgeOutput",
        "rules": [
            "Wrapped objects require exactly schema_version, requires_card, and "
            "dimensions.",
            "Bare payloads inherit requires_card from the bound evaluation packet.",
            "Reject missing, duplicate, or extra keys.",
            "Reject missing, duplicate, extra, or out-of-order dimensions.",
            "Reject scores outside each dimension's inclusive maximum.",
            "Reject fences, trailing content, coercion, and result repair.",
        ],
    }


FINAL_JUDGE_PARSER_POLICY_SHA256_V3 = sha256_bytes(
    canonical_json_bytes(final_judge_parser_policy_v3())
)


def final_judge_parser_policy_v4() -> dict[str, object]:
    """Return the frozen v4 policy that adds one complete outer fence."""

    return {
        "policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V4,
        "base_policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
        "base_policy_sha256": FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
        "input_format": "one_strict_json_value_with_optional_complete_outer_fence",
        "accepted_outer_wrappers": [
            "none",
            "single_unlabelled_triple_backtick_fence",
            "single_lowercase_json_triple_backtick_fence",
        ],
        "fence_rules": [
            "The opening and closing fence delimiters must each occupy their own line.",
            "The opening delimiter must be exactly ``` or ```json.",
            "No whitespace, prose, or other content may occur outside the fence.",
            "Reject incomplete, nested, or multiple fences.",
            "After removing the one outer fence, apply the complete v3 policy unchanged.",
        ],
        "strict_json_rules": [
            "Reject duplicate keys and non-finite JSON numbers.",
            "Reject trailing non-whitespace content, coercion, repair, and shape drift.",
        ],
    }


FINAL_JUDGE_PARSER_POLICY_SHA256_V4 = sha256_bytes(
    canonical_json_bytes(final_judge_parser_policy_v4())
)


class EvaluatorOutputParseError(ValueError):
    """The evaluator response is not one exact contract-compliant JSON object."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank and trimmed")
    return value


class GroundedFeedbackFinding(_StrictFrozenModel):
    dimension: FeedbackDimension
    severity: FeedbackSeverity
    grounded_in_image: bool
    description: str
    evidence: tuple[str, ...] = Field(min_length=1, max_length=4)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _nonblank(value, "description")

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        checked = tuple(_nonblank(item, "evidence") for item in value)
        if len(set(checked)) != len(checked):
            raise ValueError("feedback evidence must be unique")
        return checked


class VisualFeedbackOutput(_StrictFrozenModel):
    """Structured development feedback consumed by later evolution stages."""

    schema_version: Literal[1]
    summary: str
    rule_violations: tuple[GroundedFeedbackFinding, ...] = Field(max_length=20)
    ideal_response_gaps: tuple[GroundedFeedbackFinding, ...] = Field(max_length=20)
    skill_suggestions: tuple[str, ...] = Field(max_length=8)
    evidence_quality_notes: str | None = None

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _nonblank(value, "summary")

    @field_validator("evidence_quality_notes")
    @classmethod
    def validate_evidence_quality_notes(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "evidence_quality_notes")

    @field_validator("skill_suggestions")
    @classmethod
    def validate_suggestions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        checked = tuple(_nonblank(item, "skill_suggestions") for item in value)
        if len(set(checked)) != len(checked):
            raise ValueError("skill suggestions must be unique")
        return checked


class FinalJudgeDimensionAssessment(_StrictFrozenModel):
    """One raw model-selected integer score; all derived fields stay local."""

    dimension: JudgeDimension
    score: int = Field(ge=0, le=20)

    @model_validator(mode="after")
    def validate_maximum(self) -> Self:
        if self.score > _DIMENSION_MAX[self.dimension]:
            raise ValueError(
                f"{self.dimension} score exceeds {_DIMENSION_MAX[self.dimension]}"
            )
        return self


class FinalJudgeOutput(_StrictFrozenModel):
    """Raw strict JSON contract returned by the visual final Judge."""

    schema_version: Literal[1]
    requires_card: bool
    dimensions: tuple[FinalJudgeDimensionAssessment, ...]

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        expected = {"CA", "CQ", "TCR"}
        if self.requires_card:
            expected.add("CCC")
        actual = tuple(item.dimension for item in self.dimensions)
        if (
            set(actual) != expected
            or actual != tuple(sorted(actual))
            or len(actual) != len(set(actual))
        ):
            raise ValueError(
                "Judge dimensions must be the exact applicable set in "
                "alphabetical order"
            )
        return self


class ParsedFinalJudgeOutputV2(_StrictFrozenModel):
    """One normalized Judge submission plus its raw-shape policy identity."""

    submission: FinalJudgeOutput
    raw_dimensions_shape: FinalJudgeDimensionsShape
    policy_version: Literal["final-judge-equivalent-shapes-v2"] = (
        FINAL_JUDGE_PARSER_POLICY_VERSION_V2
    )
    policy_sha256: str = FINAL_JUDGE_PARSER_POLICY_SHA256_V2

    @model_validator(mode="after")
    def validate_policy_identity(self) -> Self:
        if self.policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V2:
            raise ValueError("final Judge parser policy hash mismatch")
        return self


class ParsedFinalJudgeOutputV3(_StrictFrozenModel):
    """One normalized Judge submission plus its v3 raw-shape identity."""

    submission: FinalJudgeOutput
    raw_dimensions_shape: FinalJudgeDimensionsShapeV3
    policy_version: Literal["final-judge-semantic-score-shapes-v3"] = (
        FINAL_JUDGE_PARSER_POLICY_VERSION_V3
    )
    policy_sha256: str = FINAL_JUDGE_PARSER_POLICY_SHA256_V3

    @model_validator(mode="after")
    def validate_policy_identity(self) -> Self:
        if self.policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V3:
            raise ValueError("final Judge parser policy hash mismatch")
        return self


FinalJudgeInputWrapperV4 = Literal[
    "none",
    "markdown_fence",
    "markdown_json_fence",
]


class ParsedFinalJudgeOutputV4(_StrictFrozenModel):
    """One v3-compatible submission plus its optional outer-wrapper identity."""

    submission: FinalJudgeOutput
    raw_dimensions_shape: FinalJudgeDimensionsShapeV3
    input_wrapper: FinalJudgeInputWrapperV4
    policy_version: Literal["final-judge-complete-fence-wrapper-v4"] = (
        FINAL_JUDGE_PARSER_POLICY_VERSION_V4
    )
    policy_sha256: str = FINAL_JUDGE_PARSER_POLICY_SHA256_V4

    @model_validator(mode="after")
    def validate_policy_identity(self) -> Self:
        if self.policy_sha256 != FINAL_JUDGE_PARSER_POLICY_SHA256_V4:
            raise ValueError("final Judge parser policy hash mismatch")
        return self


OutputModel = TypeVar("OutputModel", bound=BaseModel)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _parse_exact_json(text: str, model_type: type[OutputModel]) -> OutputModel:
    if type(text) is not str:
        raise TypeError("evaluator output must be text")
    encoded = text.encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise EvaluatorOutputParseError("evaluator output size is invalid")
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EvaluatorOutputParseError(
            "evaluator output is not one exact JSON value"
        ) from error
    if not isinstance(decoded, dict):
        raise EvaluatorOutputParseError("evaluator output must be one JSON object")
    try:
        return model_type.model_validate_json(text, strict=True)
    except ValidationError as error:
        raise EvaluatorOutputParseError(
            "evaluator output violates the frozen JSON contract"
        ) from error


def parse_visual_feedback_output(text: str) -> VisualFeedbackOutput:
    """Parse the historical exact-JSON Feedback contract."""

    return _parse_exact_json(text, VisualFeedbackOutput)


_COMPLETE_VISUAL_FEEDBACK_FENCE_V2 = re.compile(
    r"\A```(?P<label>json)?\r?\n(?P<body>.*?)\r?\n```\Z",
    re.DOTALL,
)


def _unwrap_visual_feedback_output_v2(text: str) -> str:
    if type(text) is not str:
        raise TypeError("evaluator output must be text")
    encoded = text.encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise EvaluatorOutputParseError("evaluator output size is invalid")
    if not text.startswith("```"):
        return text

    match = _COMPLETE_VISUAL_FEEDBACK_FENCE_V2.fullmatch(text)
    if match is None:
        raise EvaluatorOutputParseError(
            "evaluator output is not one complete supported outer fence"
        )
    body = match.group("body")
    if not body or "```" in body:
        raise EvaluatorOutputParseError(
            "evaluator output contains an empty, nested, or multiple fence"
        )
    return body


def parse_visual_feedback_output_v2(text: str) -> VisualFeedbackOutput:
    """Parse strict Feedback JSON with at most one complete outer fence."""

    return parse_visual_feedback_output(_unwrap_visual_feedback_output_v2(text))


def parse_final_judge_output(
    text: str,
    *,
    expected_requires_card: bool,
) -> FinalJudgeOutput:
    parsed = _parse_exact_json(text, FinalJudgeOutput)
    if parsed.requires_card is not expected_requires_card:
        raise EvaluatorOutputParseError(
            "Judge requires_card does not match the evaluated packet"
        )
    return parsed


def _decode_exact_json_object_v2(text: str) -> dict[str, object]:
    if type(text) is not str:
        raise TypeError("evaluator output must be text")
    encoded = text.encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise EvaluatorOutputParseError("evaluator output size is invalid")
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EvaluatorOutputParseError(
            "evaluator output is not one exact JSON value"
        ) from error
    if type(decoded) is not dict:
        raise EvaluatorOutputParseError("evaluator output must be one JSON object")
    return decoded


def _trim_visual_feedback_free_text_v3(
    decoded: dict[str, object],
) -> dict[str, object]:
    """Copy and trim only the policy-v3 allowlisted free-text leaves."""

    normalized = dict(decoded)
    summary = decoded.get("summary")
    if type(summary) is str:
        normalized["summary"] = summary.strip()

    for collection_name in ("rule_violations", "ideal_response_gaps"):
        collection = decoded.get(collection_name)
        if type(collection) is not list:
            continue
        normalized_collection: list[object] = []
        for finding in collection:
            if type(finding) is not dict:
                normalized_collection.append(finding)
                continue
            normalized_finding = dict(finding)
            description = finding.get("description")
            if type(description) is str:
                normalized_finding["description"] = description.strip()
            evidence = finding.get("evidence")
            if type(evidence) is list:
                normalized_finding["evidence"] = [
                    item.strip() if type(item) is str else item for item in evidence
                ]
            normalized_collection.append(normalized_finding)
        normalized[collection_name] = normalized_collection

    suggestions = decoded.get("skill_suggestions")
    if type(suggestions) is list:
        normalized["skill_suggestions"] = [
            item.strip() if type(item) is str else item for item in suggestions
        ]
    return normalized


def parse_visual_feedback_output_v3(text: str) -> VisualFeedbackOutput:
    """Parse v2-wrapped Feedback while trimming only approved text leaves."""

    body = _unwrap_visual_feedback_output_v2(text)
    decoded = _decode_exact_json_object_v2(body)
    normalized = _trim_visual_feedback_free_text_v3(decoded)
    try:
        return VisualFeedbackOutput.model_validate_json(
            canonical_json_bytes(normalized),
            strict=True,
        )
    except ValidationError as error:
        raise EvaluatorOutputParseError(
            "evaluator output violates the frozen normalized JSON contract"
        ) from error


def parse_visual_feedback_output_v4(text: str) -> VisualFeedbackOutput:
    """Accept one observed empty provider annotation, then apply v3 strictly.

    Qwen3.8 occasionally appends ``summary_note`` despite the strict response
    schema.  The field carries no evidence and is safe to discard only when it
    is null or blank.  Non-empty annotations and every other schema drift stay
    rejected.
    """

    body = _unwrap_visual_feedback_output_v2(text)
    decoded = _decode_exact_json_object_v2(body)
    if "summary_note" in decoded:
        note = decoded.pop("summary_note")
        if note is not None and (type(note) is not str or note.strip()):
            raise EvaluatorOutputParseError(
                "Feedback summary_note must be absent, null, or blank"
            )
    # Qwen3.8 has also emitted one null/blank annotation adjacent to a valid
    # strict-schema field.  Preserve the same narrow rule for any top-level
    # ``*_note`` key: it carries no evaluation evidence, while a nonempty or
    # non-note extra remains a hard schema failure.
    for key in tuple(decoded):
        if key.endswith("_note"):
            note = decoded.pop(key)
            if note is not None and (type(note) is not str or note.strip()):
                raise EvaluatorOutputParseError(
                    "Feedback note annotations must be absent, null, or blank"
                )
    # Strict-schema Qwen3.8 has returned JSON string spellings for this one
    # boolean leaf. The mapping is lossless and closed; no free text, labels,
    # evidence, severity, or suggestion content is changed.
    for collection_name in ("rule_violations", "ideal_response_gaps"):
        collection = decoded.get(collection_name)
        if type(collection) is not list:
            continue
        for finding in collection:
            if type(finding) is not dict:
                continue
            grounded = finding.get("grounded_in_image")
            if grounded == "true":
                finding["grounded_in_image"] = True
            elif grounded == "false":
                finding["grounded_in_image"] = False
    normalized = _trim_visual_feedback_free_text_v3(decoded)
    try:
        return VisualFeedbackOutput.model_validate_json(
            canonical_json_bytes(normalized),
            strict=True,
        )
    except ValidationError as error:
        raise EvaluatorOutputParseError(
            "evaluator output violates the frozen normalized JSON contract"
        ) from error


def _applicable_judge_dimensions(
    requires_card: bool,
) -> tuple[JudgeDimension, ...]:
    if requires_card:
        return ("CA", "CCC", "CQ", "TCR")
    return ("CA", "CQ", "TCR")


def _validate_v2_score(value: object, dimension: JudgeDimension) -> int:
    if type(value) is not int:
        raise EvaluatorOutputParseError(
            f"{dimension} score must be a strict JSON integer"
        )
    if value < 0 or value > _DIMENSION_MAX[dimension]:
        raise EvaluatorOutputParseError(
            f"{dimension} score must be within its inclusive maximum"
        )
    return value


def _parse_v2_assessment_array(
    raw_dimensions: list[object],
    expected_dimensions: tuple[JudgeDimension, ...],
) -> tuple[tuple[int, ...], FinalJudgeDimensionsShape]:
    if len(raw_dimensions) != len(expected_dimensions):
        raise EvaluatorOutputParseError(
            "Judge dimensions must be the exact applicable set"
        )
    observed_dimensions: list[object] = []
    scores: list[int] = []
    for item, expected_dimension in zip(
        raw_dimensions, expected_dimensions, strict=True
    ):
        if type(item) is not dict or set(item) != {"dimension", "score"}:
            raise EvaluatorOutputParseError(
                "assessment-array items require exactly dimension and score"
            )
        observed_dimensions.append(item["dimension"])
        scores.append(_validate_v2_score(item["score"], expected_dimension))
    if tuple(observed_dimensions) != expected_dimensions:
        raise EvaluatorOutputParseError(
            "Judge dimensions must be in exact alphabetical applicable order"
        )
    return tuple(scores), "assessment_array"


def _parse_v2_score_mapping(
    raw_dimensions: dict[str, object],
    expected_dimensions: tuple[JudgeDimension, ...],
) -> tuple[tuple[int, ...], FinalJudgeDimensionsShape]:
    if tuple(raw_dimensions) != expected_dimensions:
        raise EvaluatorOutputParseError(
            "score-mapping keys must be exact alphabetical applicable dimensions"
        )
    scores = tuple(
        _validate_v2_score(raw_dimensions[dimension], dimension)
        for dimension in expected_dimensions
    )
    return scores, "score_mapping"


def _parse_v2_dimension_score_pairs(
    raw_dimensions: list[object],
    expected_dimensions: tuple[JudgeDimension, ...],
) -> tuple[tuple[int, ...], FinalJudgeDimensionsShape]:
    if len(raw_dimensions) != len(expected_dimensions):
        raise EvaluatorOutputParseError(
            "Judge dimensions must be the exact applicable set"
        )
    observed_dimensions: list[object] = []
    scores: list[int] = []
    for item, expected_dimension in zip(
        raw_dimensions, expected_dimensions, strict=True
    ):
        if type(item) is not list or len(item) != 2:
            raise EvaluatorOutputParseError(
                "dimension-score-pair items must contain exactly two values"
            )
        observed_dimensions.append(item[0])
        scores.append(_validate_v2_score(item[1], expected_dimension))
    if tuple(observed_dimensions) != expected_dimensions:
        raise EvaluatorOutputParseError(
            "Judge dimensions must be in exact alphabetical applicable order"
        )
    return tuple(scores), "dimension_score_pairs"


def parse_final_judge_output_v2(
    text: str,
    *,
    expected_requires_card: bool,
) -> ParsedFinalJudgeOutputV2:
    """Parse one of three exact equivalent shapes without changing legacy v1."""

    decoded = _decode_exact_json_object_v2(text)
    if set(decoded) != {"schema_version", "requires_card", "dimensions"}:
        raise EvaluatorOutputParseError(
            "Judge output requires the exact frozen top-level keys"
        )
    if type(decoded["schema_version"]) is not int or decoded["schema_version"] != 1:
        raise EvaluatorOutputParseError("Judge schema_version must be integer 1")
    requires_card = decoded["requires_card"]
    if type(requires_card) is not bool:
        raise EvaluatorOutputParseError(
            "Judge requires_card must be a strict JSON boolean"
        )
    if requires_card is not expected_requires_card:
        raise EvaluatorOutputParseError(
            "Judge requires_card does not match the evaluated packet"
        )

    expected_dimensions = _applicable_judge_dimensions(requires_card)
    raw_dimensions = decoded["dimensions"]
    if type(raw_dimensions) is dict:
        scores, shape = _parse_v2_score_mapping(raw_dimensions, expected_dimensions)
    elif (
        type(raw_dimensions) is list
        and raw_dimensions
        and all(type(item) is dict for item in raw_dimensions)
    ):
        scores, shape = _parse_v2_assessment_array(raw_dimensions, expected_dimensions)
    elif (
        type(raw_dimensions) is list
        and raw_dimensions
        and all(type(item) is list for item in raw_dimensions)
    ):
        scores, shape = _parse_v2_dimension_score_pairs(
            raw_dimensions, expected_dimensions
        )
    else:
        raise EvaluatorOutputParseError(
            "Judge dimensions do not match an accepted exact shape"
        )

    submission = FinalJudgeOutput(
        schema_version=1,
        requires_card=requires_card,
        dimensions=tuple(
            FinalJudgeDimensionAssessment(dimension=dimension, score=score)
            for dimension, score in zip(expected_dimensions, scores, strict=True)
        ),
    )
    return ParsedFinalJudgeOutputV2(
        submission=submission,
        raw_dimensions_shape=shape,
    )


def _decode_exact_json_value_v3(text: str) -> object:
    if type(text) is not str:
        raise TypeError("evaluator output must be text")
    encoded = text.encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise EvaluatorOutputParseError("evaluator output size is invalid")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EvaluatorOutputParseError(
            "evaluator output is not one exact JSON value"
        ) from error


def _parse_v3_dimensions(
    raw_dimensions: object,
    expected_dimensions: tuple[JudgeDimension, ...],
) -> tuple[tuple[int, ...], FinalJudgeDimensionsShape]:
    if type(raw_dimensions) is dict:
        return _parse_v2_score_mapping(raw_dimensions, expected_dimensions)
    if (
        type(raw_dimensions) is list
        and raw_dimensions
        and all(type(item) is dict for item in raw_dimensions)
    ):
        return _parse_v2_assessment_array(raw_dimensions, expected_dimensions)
    if (
        type(raw_dimensions) is list
        and raw_dimensions
        and all(type(item) is list for item in raw_dimensions)
    ):
        return _parse_v2_dimension_score_pairs(raw_dimensions, expected_dimensions)
    raise EvaluatorOutputParseError(
        "Judge dimensions do not match an accepted exact shape"
    )


def parse_final_judge_output_v3(
    text: str,
    *,
    expected_requires_card: bool,
) -> ParsedFinalJudgeOutputV3:
    """Parse exact wrapped or bare score shapes without repair or coercion."""

    if type(expected_requires_card) is not bool:
        raise TypeError("expected_requires_card must be a strict boolean")

    decoded = _decode_exact_json_value_v3(text)
    expected_dimensions = _applicable_judge_dimensions(expected_requires_card)
    wrapped_keys = {"schema_version", "requires_card", "dimensions"}

    if type(decoded) is dict and set(decoded) == wrapped_keys:
        if type(decoded["schema_version"]) is not int or decoded["schema_version"] != 1:
            raise EvaluatorOutputParseError("Judge schema_version must be integer 1")
        requires_card = decoded["requires_card"]
        if type(requires_card) is not bool:
            raise EvaluatorOutputParseError(
                "Judge requires_card must be a strict JSON boolean"
            )
        if requires_card is not expected_requires_card:
            raise EvaluatorOutputParseError(
                "Judge requires_card does not match the evaluated packet"
            )
        scores, raw_shape = _parse_v3_dimensions(
            decoded["dimensions"], expected_dimensions
        )
        shape: FinalJudgeDimensionsShapeV3 = raw_shape
    elif type(decoded) is dict:
        scores, _ = _parse_v2_score_mapping(decoded, expected_dimensions)
        shape = "bare_score_mapping"
    elif (
        type(decoded) is list
        and decoded
        and all(type(item) is dict for item in decoded)
    ):
        scores, _ = _parse_v2_assessment_array(decoded, expected_dimensions)
        shape = "bare_assessment_array"
    elif (
        type(decoded) is list
        and decoded
        and all(type(item) is list for item in decoded)
    ):
        scores, _ = _parse_v2_dimension_score_pairs(decoded, expected_dimensions)
        shape = "bare_dimension_score_pairs"
    else:
        raise EvaluatorOutputParseError(
            "Judge output does not match an accepted exact wrapped or bare shape"
        )

    submission = FinalJudgeOutput(
        schema_version=1,
        requires_card=expected_requires_card,
        dimensions=tuple(
            FinalJudgeDimensionAssessment(dimension=dimension, score=score)
            for dimension, score in zip(expected_dimensions, scores, strict=True)
        ),
    )
    return ParsedFinalJudgeOutputV3(
        submission=submission,
        raw_dimensions_shape=shape,
    )


_COMPLETE_FINAL_JUDGE_FENCE_V4 = re.compile(
    r"\A```(?P<label>json)?\r?\n(?P<body>.*?)\r?\n```\Z",
    re.DOTALL,
)


def _unwrap_final_judge_output_v4(
    text: str,
) -> tuple[str, FinalJudgeInputWrapperV4]:
    if type(text) is not str:
        raise TypeError("evaluator output must be text")
    encoded = text.encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise EvaluatorOutputParseError("evaluator output size is invalid")
    if not text.startswith("```"):
        return text, "none"

    match = _COMPLETE_FINAL_JUDGE_FENCE_V4.fullmatch(text)
    if match is None:
        raise EvaluatorOutputParseError(
            "evaluator output is not one complete supported outer fence"
        )
    body = match.group("body")
    if not body or "```" in body:
        raise EvaluatorOutputParseError(
            "evaluator output contains an empty, nested, or multiple fence"
        )
    wrapper: FinalJudgeInputWrapperV4
    if match.group("label") == "json":
        wrapper = "markdown_json_fence"
    else:
        wrapper = "markdown_fence"
    return body, wrapper


def parse_final_judge_output_v4(
    text: str,
    *,
    expected_requires_card: bool,
) -> ParsedFinalJudgeOutputV4:
    """Parse v3 shapes with at most one complete outer Markdown fence."""

    unwrapped, wrapper = _unwrap_final_judge_output_v4(text)
    parsed = parse_final_judge_output_v3(
        unwrapped,
        expected_requires_card=expected_requires_card,
    )
    return ParsedFinalJudgeOutputV4(
        submission=parsed.submission,
        raw_dimensions_shape=parsed.raw_dimensions_shape,
        input_wrapper=wrapper,
    )


def feedback_output_contract() -> dict[str, object]:
    """Return the historical response instructions embedded in v1-v3 prompts."""

    return {
        "format": "exactly_one_json_object_no_markdown",
        "schema_version": 1,
        "top_level_fields": [
            "schema_version",
            "summary",
            "rule_violations",
            "ideal_response_gaps",
            "skill_suggestions",
        ],
        "finding_fields": [
            "dimension",
            "severity",
            "grounded_in_image",
            "description",
            "evidence",
        ],
        "finding_dimensions": ["TCR", "CCC", "CQ", "CA", "routing", "tool_use"],
        "severity_values": ["low", "medium", "high"],
        "rules": [
            "Use empty arrays when there are no findings or suggestions.",
            "Every finding needs one to four concrete evidence strings.",
            "Set grounded_in_image true only when the image itself supports it.",
            "Return no prose, markdown fence, or key outside this contract.",
        ],
    }


VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_VERSION_V4 = (
    "visual-feedback-exact-shape-output-contract-v4"
)


def feedback_output_contract_v4() -> dict[str, object]:
    """Return the clarified Feedback shape without changing parser semantics."""

    return {
        "format": "exactly_one_json_object_no_markdown",
        "schema_version": 1,
        "top_level_fields_exactly_once": [
            "schema_version",
            "summary",
            "rule_violations",
            "ideal_response_gaps",
            "skill_suggestions",
        ],
        "finding_array_fields": [
            "rule_violations",
            "ideal_response_gaps",
        ],
        "finding_fields_exactly_once": [
            "dimension",
            "severity",
            "grounded_in_image",
            "description",
            "evidence",
        ],
        "finding_dimensions": ["TCR", "CCC", "CQ", "CA", "routing", "tool_use"],
        "severity_values": ["low", "medium", "high"],
        "skill_suggestions_item_schema": {
            "type": "string",
            "nonblank_after_trim": True,
        },
        "rules": [
            "Return exactly the five top-level fields listed above, each once.",
            "Do not copy or echo the input field named output_contract.",
            "Use empty arrays when there are no findings or suggestions.",
            "Every rule_violations or ideal_response_gaps item must be an object "
            "with exactly the five finding fields listed above.",
            "Every finding needs one to four nonblank evidence strings.",
            "skill_suggestions must be an array of nonblank strings, never finding "
            "objects or any other object.",
            "Set grounded_in_image true only when the image itself supports it.",
            "Return no prose, markdown fence, duplicate key, or unlisted key.",
        ],
    }


VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_SHA256_V4 = sha256_bytes(
    canonical_json_bytes(feedback_output_contract_v4())
)


def final_output_contract(*, requires_card: bool) -> dict[str, object]:
    """Return the applicable final-Judge response contract."""

    dimensions = ["CA", "CQ", "TCR"]
    if requires_card:
        dimensions.insert(1, "CCC")
    maxima = {name: _DIMENSION_MAX[name] for name in dimensions}
    return {
        "format": "exactly_one_json_object_no_markdown",
        "schema_version": 1,
        "requires_card": requires_card,
        "top_level_fields": ["schema_version", "requires_card", "dimensions"],
        "dimension_order": dimensions,
        "dimension_maxima": maxima,
        "dimension_fields": ["dimension", "score"],
        "rules": [
            "Return every applicable dimension exactly once in the stated order.",
            "Scores must be integers within the stated inclusive maximum.",
            "Use only visible packet and image evidence.",
            "Do not return tier, total score, rationale, or derived fields.",
            "Return no prose, markdown fence, or key outside this contract.",
        ],
    }


__all__ = [
    "EvaluatorOutputParseError",
    "FINAL_JUDGE_PARSER_POLICY_SHA256_V2",
    "FINAL_JUDGE_PARSER_POLICY_SHA256_V3",
    "FINAL_JUDGE_PARSER_POLICY_SHA256_V4",
    "FINAL_JUDGE_PARSER_POLICY_VERSION_V2",
    "FINAL_JUDGE_PARSER_POLICY_VERSION_V3",
    "FINAL_JUDGE_PARSER_POLICY_VERSION_V4",
    "VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V1",
    "VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2",
    "VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3",
    "VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V1",
    "VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2",
    "VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3",
    "VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_SHA256_V4",
    "VISUAL_FEEDBACK_OUTPUT_CONTRACT_POLICY_VERSION_V4",
    "FinalJudgeDimensionAssessment",
    "FinalJudgeDimensionsShape",
    "FinalJudgeDimensionsShapeV3",
    "FinalJudgeInputWrapperV4",
    "FinalJudgeOutput",
    "GroundedFeedbackFinding",
    "ParsedFinalJudgeOutputV2",
    "ParsedFinalJudgeOutputV3",
    "ParsedFinalJudgeOutputV4",
    "VisualFeedbackOutput",
    "feedback_output_contract",
    "feedback_output_contract_v4",
    "final_judge_parser_policy_v2",
    "final_judge_parser_policy_v3",
    "final_judge_parser_policy_v4",
    "final_output_contract",
    "parse_final_judge_output",
    "parse_final_judge_output_v2",
    "parse_final_judge_output_v3",
    "parse_final_judge_output_v4",
    "parse_visual_feedback_output",
    "parse_visual_feedback_output_v2",
    "parse_visual_feedback_output_v3",
    "parse_visual_feedback_output_v4",
    "visual_feedback_parser_policy_v1",
    "visual_feedback_parser_policy_v2",
    "visual_feedback_parser_policy_v3",
]
