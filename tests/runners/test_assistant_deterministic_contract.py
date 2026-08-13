from __future__ import annotations

from skillchain.runners.assistant_deterministic_contract import (
    DETERMINISTIC_ASSISTANT_CONTRACT_SHA256,
    DeterministicToolObservation,
    DeterministicSemanticPolicy,
    compile_deterministic_response,
    deterministic_contract_payload,
    deterministic_tool_names,
    next_deterministic_tool,
    parse_deterministic_semantic_policy,
    render_deterministic_semantic_policy,
)


def _success(tool_name: str, output: dict) -> DeterministicToolObservation:
    return DeterministicToolObservation(
        tool_name=tool_name,
        status="success",
        public_output=output,
    )


def test_contract_identity_is_versioned_and_canonical() -> None:
    payload = deterministic_contract_payload()
    assert payload["policy_version"] == "core-fast-deterministic-action-response-v5"
    assert len(DETERMINISTIC_ASSISTANT_CONTRACT_SHA256) == 64
    assert payload["tool_owner"] == "runner"
    assert payload["response_owner"] == "deterministic_public_dto_compiler"
    assert deterministic_tool_names("utility.recipe_guidance") == (
        "object_detect",
        "recipe_lookup",
    )


def test_tool_first_sequence_is_runner_owned() -> None:
    first = next_deterministic_tool("utility.recipe_guidance", ())
    assert first is not None
    assert (first.tool_name, first.arguments) == (
        "object_detect",
        {"asset_id": "query_asset"},
    )
    detected = _success(
        "object_detect",
        {"result_kind": "detections", "detections": [{"label": "ramen"}]},
    )
    second = next_deterministic_tool("utility.recipe_guidance", (detected,))
    assert second is not None
    assert (second.tool_name, second.arguments) == (
        "recipe_lookup",
        {"dish": "ramen"},
    )


def test_multi_compiler_copies_every_item_and_deduplicates_cards() -> None:
    candidate = {
        "evidence_reference": "tool-call-1-evidence-1",
        "product_id": "tool-call-1-product-1",
        "title": "Blue shoe",
    }
    observation = _success(
        "multi_product_search",
        {
            "items": [
                {
                    "item_ref": "item-001",
                    "label": "shoe",
                    "status": "matched",
                    "candidate_ordinal": 1,
                    "candidate": candidate,
                },
                {
                    "item_ref": "item-002",
                    "label": "shoe",
                    "status": "matched",
                    "candidate_ordinal": 1,
                    "candidate": candidate,
                },
                {
                    "item_ref": "item-003",
                    "label": "bag",
                    "status": "unresolved",
                    "candidate_ordinal": None,
                    "candidate": None,
                },
            ]
        },
    )
    response = compile_deterministic_response("product.multi_search", (observation,))
    assert response is not None
    assert response.count("item-001") == 1
    assert response.count("item-002") == 1
    assert response.count("item-003") == 1
    assert response.count("tool-call-1-product-1") == 1
    assert "item-003 | bag | unresolved" in response


def test_exact_semantic_selector_changes_candidates_but_preserves_card_closure() -> (
    None
):
    observations = (
        _success(
            "image_product_search",
            {
                "candidates": [
                    {
                        "evidence_reference": "tool-call-1-evidence-1",
                        "product_id": "tool-call-1-product-1",
                        "title": "Red shoe",
                    },
                    {
                        "evidence_reference": "tool-call-1-evidence-2",
                        "product_id": "tool-call-1-product-2",
                        "title": "Blue bag",
                    },
                ]
            },
        ),
    )
    parent = compile_deterministic_response("product.exact_match", observations)
    candidate = compile_deterministic_response(
        "product.exact_match",
        observations,
        semantic_policy=DeterministicSemanticPolicy(
            capability_id="product.exact_match",
            evidence_terms=("shoe",),
        ),
    )
    assert parent is not None and candidate is not None
    assert "Red shoe" in candidate
    assert "Blue bag" in parent and "Blue bag" not in candidate
    assert candidate.count("tool-call-1-evidence-1") == 1


def test_multi_semantic_selector_preserves_every_item_and_referential_integrity() -> (
    None
):
    observation = _success(
        "multi_product_search",
        {
            "items": [
                {
                    "item_ref": "item-001",
                    "label": "shoe",
                    "status": "matched",
                    "candidate_ordinal": 1,
                    "candidate": {
                        "evidence_reference": "tool-call-1-evidence-1",
                        "product_id": "tool-call-1-product-1",
                        "title": "Blue shoe",
                    },
                },
                {
                    "item_ref": "item-002",
                    "label": "bag",
                    "status": "matched",
                    "candidate_ordinal": 2,
                    "candidate": {
                        "evidence_reference": "tool-call-1-evidence-2",
                        "product_id": "tool-call-1-product-2",
                        "title": "Black bag",
                    },
                },
            ]
        },
    )
    response = compile_deterministic_response(
        "product.multi_search",
        (observation,),
        semantic_policy=DeterministicSemanticPolicy(
            capability_id="product.multi_search",
            evidence_terms=("shoe",),
        ),
    )
    assert response is not None
    assert response.count("item-001") == 1
    assert response.count("item-002") == 1
    assert "item-001 | shoe | matched | candidate-1" in response
    assert "item-002 | bag | unresolved" in response
    assert "tool-call-1-product-1" in response
    assert "tool-call-1-product-2" not in response


def test_supported_and_fallback_branches_are_mutually_exclusive() -> None:
    supported = compile_deterministic_response(
        "product.exact_match",
        (
            _success(
                "image_product_search",
                {
                    "candidates": [
                        {
                            "evidence_reference": "tool-call-1-evidence-1",
                            "product_id": "tool-call-1-product-1",
                            "title": "Supported product",
                        }
                    ]
                },
            ),
        ),
    )
    fallback = compile_deterministic_response(
        "product.exact_match",
        (_success("image_product_search", {"candidates": []}),),
    )
    assert supported is not None and "no supported match" not in supported
    assert fallback is not None and "no supported match" in fallback
    assert "tool-call-" not in fallback
    assert "product_cards:\nnone" in fallback


def test_style_closure_copies_evidence_and_cards_exactly() -> None:
    response = compile_deterministic_response(
        "product.style_recommendation",
        (
            _success(
                "style_similar_search",
                {
                    "support_status": "supported",
                    "candidates": [
                        {
                            "evidence_reference": "tool-call-1-evidence-1",
                            "product_id": "tool-call-1-product-1",
                            "title": "Red dress",
                            "style_evidence": [
                                {
                                    "evidence_reference": (
                                        "tool-call-1-style-evidence-1-1"
                                    ),
                                    "facet": "color",
                                    "value": "red",
                                }
                            ],
                        }
                    ],
                },
            ),
        ),
    )
    assert response is not None
    assert "tool-call-1-style-evidence-1-1 | color: red" in response
    assert "tool-call-1-evidence-1 | tool-call-1-product-1 | Red dress" in response


def test_detection_without_entity_compiles_safe_lookup_fallback() -> None:
    observation = _success(
        "object_detect",
        {"result_kind": "detections", "detections": []},
    )
    assert (
        next_deterministic_tool("knowledge.visual_encyclopedia", (observation,)) is None
    )
    response = compile_deterministic_response(
        "knowledge.visual_encyclopedia", (observation,)
    )
    assert response is not None
    assert response.count("not enough evidence") == 2
    assert "tool-call-" not in response


def test_ocr_compiler_uses_literal_line_substrings_and_handles() -> None:
    response = compile_deterministic_response(
        "utility.document_reading",
        (
            _success(
                "document_ocr",
                {
                    "lines": [
                        {
                            "line_reference": "tool-call-1-line-1",
                            "text": "Total 19.99",
                            "fields": [["total", "19.99"]],
                        }
                    ]
                },
            ),
        ),
    )
    assert response is not None
    assert response.count("total: 19.99 tool-call-1-line-1") == 2
    assert response.endswith("uncertainty:\nuntrusted document text")


def test_typed_ocr_plan_changes_only_literal_material_spans() -> None:
    observations = (
        _success(
            "document_ocr",
            {
                "lines": [
                    {
                        "line_reference": "tool-call-1-line-1",
                        "text": "ACME LTD.",
                        "fields": [],
                    },
                    {
                        "line_reference": "tool-call-1-line-2",
                        "text": "��",
                        "fields": [],
                    },
                ]
            },
        ),
    )
    parent = compile_deterministic_response("utility.document_reading", observations)
    candidate = compile_deterministic_response(
        "utility.document_reading",
        observations,
        semantic_policy=DeterministicSemanticPolicy(
            capability_id="utility.document_reading",
            ocr_extraction_plan="literal-material-spans",
        ),
    )
    assert parent is not None and candidate is not None
    assert "text: ACME LTD. tool-call-1-line-1" in parent
    assert "text: �� tool-call-1-line-2" in parent
    assert "text: ACME LTD tool-call-1-line-1" in candidate
    assert "tool-call-1-line-2" not in candidate


def test_only_typed_semantic_policy_changes_deterministic_response() -> None:
    observations = (
        _success(
            "object_detect",
            {"result_kind": "detections", "detections": [{"label": "soup"}]},
        ),
        _success(
            "recipe_lookup",
            {
                "sources": [
                    {
                        "evidence_reference": "tool-call-2-source-1",
                        "title": "Ingredient note",
                        "text": "Use tofu and broth.",
                    },
                    {
                        "evidence_reference": "tool-call-2-source-2",
                        "title": "Serving note",
                        "text": "Serve in a deep bowl.",
                    },
                ]
            },
        ),
    )
    parent_body = "# Recipe\n\nRuntime-owned prose A.\n"
    prose_only_body = "# Recipe\n\nRuntime-owned prose B.\n"
    parent = compile_deterministic_response(
        "utility.recipe_guidance",
        observations,
        semantic_policy=parse_deterministic_semantic_policy(
            parent_body, capability_id="utility.recipe_guidance"
        ),
    )
    prose_only = compile_deterministic_response(
        "utility.recipe_guidance",
        observations,
        semantic_policy=parse_deterministic_semantic_policy(
            prose_only_body, capability_id="utility.recipe_guidance"
        ),
    )
    policy = DeterministicSemanticPolicy(
        capability_id="utility.recipe_guidance",
        evidence_terms=("tofu",),
    )
    candidate_body = parent_body + render_deterministic_semantic_policy(policy)
    candidate = compile_deterministic_response(
        "utility.recipe_guidance",
        observations,
        semantic_policy=parse_deterministic_semantic_policy(
            candidate_body, capability_id="utility.recipe_guidance"
        ),
    )

    assert parent == prose_only
    assert candidate != parent
    assert candidate is not None and "tofu" in candidate
    assert "deep bowl" not in candidate


def test_source_compiler_repeats_handle_for_every_material_statement() -> None:
    response = compile_deterministic_response(
        "utility.recipe_guidance",
        (
            _success(
                "object_detect",
                {"result_kind": "detections", "detections": [{"label": "soup"}]},
            ),
            _success(
                "recipe_lookup",
                {
                    "sources": [
                        {
                            "evidence_reference": "tool-call-2-source-1",
                            "title": "Soup",
                            "text": "Add tofu. Simmer for 10 minutes. Serve hot.",
                        }
                    ]
                },
            ),
        ),
    )

    assert response is not None
    answer, evidence = response.split("evidence:\n", maxsplit=1)
    assert answer.count("tool-call-2-source-1") == 3
    assert evidence.count("tool-call-2-source-1") == 3
    assert "tool-call-2-source-1 | Simmer for 10 minutes." in response
