from __future__ import annotations

from skillchain.runners.assistant_response_contract import (
    AMBIGUITY_SIGNAL_POLICY,
    AssistantResponseToolObservation,
    fixed_response_repair_prompt,
    repair_preserves_material_atoms,
    validate_assistant_response_contract,
)


_ENCYCLOPEDIA_CONTRACT = {
    "required_sections": ["answer", "evidence", "uncertainty"],
}


def _empty_lookup() -> tuple[AssistantResponseToolObservation, ...]:
    return (
        AssistantResponseToolObservation(
            tool_name="encyclopedia_lookup",
            status="success",
            public_output={"result_kind": "knowledge_sources", "sources": []},
        ),
    )


def test_encyclopedia_preflight_requires_exact_sections_and_fallback_marker() -> None:
    invalid = validate_assistant_response_contract(
        "Answer: ambiguous\n<|im_end|>",
        observations=_empty_lookup(),
        selected_capability="knowledge.visual_encyclopedia",
        contract=_ENCYCLOPEDIA_CONTRACT,
    )

    assert invalid.valid is False
    assert invalid.reason_codes == (
        "response_control_token",
        "response_fallback_marker_missing",
        "response_section_invalid",
    )
    assert invalid.fallback_marker_required is True
    assert invalid.ambiguity_signal_policy == AMBIGUITY_SIGNAL_POLICY

    valid = validate_assistant_response_contract(
        "answer:\nnot enough evidence\n"
        "evidence:\nnot enough evidence\n"
        "uncertainty:\nidentity remains unresolved",
        observations=_empty_lookup(),
        selected_capability="knowledge.visual_encyclopedia",
        contract=_ENCYCLOPEDIA_CONTRACT,
    )
    assert valid.valid is True

    wrong_case = validate_assistant_response_contract(
        "answer:\nNOT ENOUGH EVIDENCE\n"
        "evidence:\nNOT ENOUGH EVIDENCE\n"
        "uncertainty:\nidentity remains unresolved",
        observations=_empty_lookup(),
        selected_capability="knowledge.visual_encyclopedia",
        contract=_ENCYCLOPEDIA_CONTRACT,
    )
    assert "response_fallback_marker_missing" in wrong_case.reason_codes

    extra_section = validate_assistant_response_contract(
        "answer:\nnot enough evidence\n"
        "notes:\nnot enough evidence\n"
        "evidence:\nnot enough evidence\n"
        "uncertainty:\nidentity remains unresolved",
        observations=_empty_lookup(),
        selected_capability="knowledge.visual_encyclopedia",
        contract=_ENCYCLOPEDIA_CONTRACT,
    )
    assert "response_section_invalid" in extra_section.reason_codes


def test_detector_only_final_is_rejected_without_guessing_ambiguity() -> None:
    validation = validate_assistant_response_contract(
        "answer:\nnot enough evidence\n"
        "evidence:\nnot enough evidence\n"
        "uncertainty:\nidentity remains unresolved",
        observations=(
            AssistantResponseToolObservation(
                tool_name="object_detect",
                status="success",
                public_output={"detections": [{"label": "plant", "confidence": 0.2}]},
            ),
        ),
        selected_capability="knowledge.visual_encyclopedia",
        contract=_ENCYCLOPEDIA_CONTRACT,
    )

    assert validation.valid is False
    assert "response_detector_only_final" in validation.reason_codes
    assert "response_tool_sequence_invalid" in validation.reason_codes
    # There is no frozen confidence threshold or public ambiguity field.  The
    # guard therefore records, rather than invents, that limitation.
    assert validation.ambiguity_signal_policy == (
        "empty_sources_only_no_reliable_public_ambiguity_field"
    )


def test_non_encyclopedia_contract_rejects_an_extra_ascii_section() -> None:
    validation = validate_assistant_response_contract(
        "answer:\nmatching products\n"
        "notes:\nuncontracted material\n"
        "product_cards:\nproduct one\n"
        "uncertainty:\nnone",
        selected_capability="product.exact_match",
        contract={
            "required_sections": ["answer", "product_cards", "uncertainty"]
        },
    )

    assert validation.valid is False
    assert validation.reason_codes == ("response_section_invalid",)


def test_encyclopedia_without_successful_lookup_requires_fallback() -> None:
    validation = validate_assistant_response_contract(
        "answer:\nidentity claimed\n"
        "evidence:\nnone retrieved\n"
        "uncertainty:\nidentity unresolved",
        observations=(),
        selected_capability="knowledge.visual_encyclopedia",
        contract=_ENCYCLOPEDIA_CONTRACT,
    )

    assert validation.fallback_marker_required is True
    assert "response_fallback_marker_missing" in validation.reason_codes
    assert "response_tool_sequence_invalid" in validation.reason_codes


def test_format_repair_cannot_introduce_an_ordinary_fact_or_entity() -> None:
    original = "identity remains unresolved <|im_end|>"
    safe = (
        "answer:\nnot enough evidence\n"
        "evidence:\nnot enough evidence\n"
        "uncertainty:\nidentity remains unresolved"
    )
    invented = safe.replace("identity", "Amanita identity")

    assert repair_preserves_material_atoms(original, safe) is True
    assert repair_preserves_material_atoms(original, invented) is False


def test_fixed_repair_prompt_exposes_no_tool_and_no_new_fact_policy() -> None:
    validation = validate_assistant_response_contract(
        "identity remains unresolved",
        observations=_empty_lookup(),
        selected_capability="knowledge.visual_encyclopedia",
        contract=_ENCYCLOPEDIA_CONTRACT,
    )
    prompt = fixed_response_repair_prompt(validation=validation)

    assert "Do not call a tool" in prompt
    assert "add a fact" in prompt
    assert "`not enough evidence`" in prompt
    assert prompt.count("One-time") == 1
