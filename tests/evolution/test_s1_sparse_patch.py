from __future__ import annotations

from pathlib import Path

import pytest

from skillchain.evaluation.portfolio_treatments import (
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
)
from skillchain.evolution.s1_sparse_patch import (
    S1SparsePatchError,
    SparseSkillContentPatchV1,
    bind_sparse_patch_draft,
    compile_sparse_s1_candidate,
    compile_counterfactual_policy_branch,
    compile_counterfactual_typed_policy_branch,
    compose_screened_sparse_bank,
    compile_policy_surface_branch,
    compose_policy_surface_branches,
    decode_sparse_parent_content,
    load_sparse_compilation_receipt,
    load_sparse_patch_draft,
    parse_dual_policy_patch,
    parse_counterfactual_policy_patch,
    parse_counterfactual_typed_policy_patch,
    counterfactual_typed_policy_patch_output_json_schema,
    sparse_patch_output_json_schema,
    _body_sections,
    _decode_parent_authoring_content,
)
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
CODEX_INPUT = ROOT / "specs/authoring/authoring-packet-codex-high-v5.json"
SEMANTIC_INPUT = ROOT / "specs/authoring/authoring-packet-primary-v5-candidate.json"
CODEX_DRAFT = (
    ROOT
    / "runs/formal-authoring/llm-static-codex-primary-20260724-high-v5"
    / "pre-review-draft.json"
)
RUNTIME_SHA = "a" * 64
FEEDBACK_SHA = "f" * 64


def _file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def test_dual_policy_surfaces_compile_and_compose_without_cross_capability_edits(
    parent_materials,
) -> None:
    parent, _authoring_input = parent_materials
    capability = "utility.recipe_guidance"
    parent_skill = next(
        item for item in parent.skills if item.capability_id == capability
    )
    proposal = parse_dual_policy_patch(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "capability_id": capability,
                "parent_skill_sha256": parent_skill.skill_sha256,
                "action_policy": {
                    "action": "patch",
                    "policy_text": "Call the lookup before answering and stop after one successful source call.",
                },
                "response_policy": {
                    "action": "patch",
                    "policy_text": "State only source-supported steps and abstain when public sources are empty.",
                },
            }
        ),
        capability_id=capability,
        parent_skill_sha256=parent_skill.skill_sha256,
    )
    action = compile_policy_surface_branch(
        parent_bank=parent, proposal=proposal, surface="action-policy"
    )
    response = compile_policy_surface_branch(
        parent_bank=parent, proposal=proposal, surface="response-policy"
    )
    assert action is not None and response is not None
    assert "S1 action policy overlay" in next(
        item.body for item in action.bank.skills if item.capability_id == capability
    )
    assert "S1 response policy overlay" not in next(
        item.body for item in action.bank.skills if item.capability_id == capability
    )
    combined = compose_policy_surface_branches(
        parent_bank=parent,
        capability_id=capability,
        branches=(action, response),
    )
    combined_skill = next(
        item for item in combined.bank.skills if item.capability_id == capability
    )
    assert "S1 action policy overlay" in combined_skill.body
    assert "S1 response policy overlay" in combined_skill.body
    assert all(
        left == right
        for left, right in zip(parent.skills, combined.bank.skills, strict=True)
        if left.capability_id != capability
    )
    assert combined.receipt.surfaces == ("action-policy", "response-policy")


def test_counterfactual_creator_compiles_exactly_one_conditional_surface(
    parent_materials,
) -> None:
    parent, _authoring_input = parent_materials
    capability = "utility.recipe_guidance"
    parent_skill = next(
        item for item in parent.skills if item.capability_id == capability
    )
    success_ids = ("parent-success-1", "parent-success-2", "parent-success-3")
    proposal = parse_counterfactual_policy_patch(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "capability_id": capability,
                "parent_skill_sha256": parent_skill.skill_sha256,
                "target_surface": "action-policy",
                "non_target_surface_action": "inherit",
                "when": "a visible detection supplies one tentative food class",
                "then": "call the recipe lookup once with that visible class and stop after its terminal tool output",
                "must_preserve": [
                    {
                        "query_id": query_id,
                        "provider_visible_state": f"the successful parent state {index} remains unchanged",
                    }
                    for index, query_id in enumerate(success_ids, start=1)
                ],
            }
        ),
        capability_id=capability,
        parent_skill_sha256=parent_skill.skill_sha256,
        target_surface="action-policy",
        parent_success_query_ids=success_ids,
    )
    compiled = compile_counterfactual_policy_branch(
        parent_bank=parent, proposal=proposal
    )
    expected = (
        "If and only if a visible detection supplies one tentative food class, "
        "call the recipe lookup once with that visible class and stop after its "
        "terminal tool output. Otherwise preserve the parent behavior, including "
        "the successful parent state 1 remains unchanged, the successful parent "
        "state 2 remains unchanged, the successful parent state 3 remains unchanged."
    )
    assert compiled.policy_text == expected
    candidate_skill = next(
        item for item in compiled.bank.skills if item.capability_id == capability
    )
    assert expected in candidate_skill.body
    assert "S1 response policy overlay" not in candidate_skill.body
    assert all(
        left == right
        for left, right in zip(parent.skills, compiled.bank.skills, strict=True)
        if left.capability_id != capability
    )


def test_counterfactual_creator_rejects_cross_surface_or_unprovided_success() -> None:
    payload = {
        "schema_version": 1,
        "capability_id": "product.multi_search",
        "parent_skill_sha256": "a" * 64,
        "target_surface": "action-policy",
        "non_target_surface_action": "inherit",
        "when": "the public mapping contains one matched item",
        "then": "copy its public association into the answer",
        "must_preserve": [
            {
                "query_id": query_id,
                "provider_visible_state": "the parent response remains unchanged",
            }
            for query_id in ("p1", "p2", "unknown")
        ],
    }
    with pytest.raises(S1SparsePatchError, match="binding drifted"):
        parse_counterfactual_policy_patch(
            canonical_json_bytes(payload),
            capability_id="product.multi_search",
            parent_skill_sha256="a" * 64,
            target_surface="response-policy",
            parent_success_query_ids=("p1", "p2", "p3"),
        )


def test_typed_action_counterfactual_has_no_response_text_channel(
    parent_materials,
) -> None:
    parent, _authoring_input = parent_materials
    capability = "utility.recipe_guidance"
    parent_skill = next(
        item for item in parent.skills if item.capability_id == capability
    )
    success_ids = ("parent-success-1", "parent-success-2", "parent-success-3")
    schema = counterfactual_typed_policy_patch_output_json_schema(
        capability_id=capability,
        parent_skill_sha256=parent_skill.skill_sha256,
        target_surface="action-policy",
        parent_success_query_ids=success_ids,
    )
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "when" not in properties and "then" not in properties
    proposal = parse_counterfactual_typed_policy_patch(
        canonical_json_bytes(
            {
                "schema_version": 2,
                "capability_id": capability,
                "parent_skill_sha256": parent_skill.skill_sha256,
                "target_surface": "action-policy",
                "non_target_surface_action": "inherit",
                "action_when": {
                    "phase": "after-tool",
                    "prior_tool_name": "object_detect",
                    "prior_tool_status": "success",
                    "public_evidence": "nonempty",
                },
                "action_then": {
                    "operation": "invoke-tool-once",
                    "tool_name": "recipe_lookup",
                    "arguments_from": "last-visible-tool-output",
                },
                "must_preserve": [
                    {
                        "query_id": query_id,
                        "provider_visible_state": f"parent state {index} remains unchanged",
                    }
                    for index, query_id in enumerate(success_ids, start=1)
                ],
            }
        ),
        capability_id=capability,
        parent_skill_sha256=parent_skill.skill_sha256,
        target_surface="action-policy",
        parent_success_query_ids=success_ids,
    )
    compiled = compile_counterfactual_typed_policy_branch(
        parent_bank=parent, proposal=proposal
    )
    assert "invoke recipe_lookup exactly once" in compiled.policy_text
    assert "parent state 1" not in compiled.policy_text
    assert compiled.policy_text.endswith(
        "Otherwise preserve the parent behavior in every other provider-visible state."
    )
    assert "answer" not in compiled.policy_text.casefold()
    assert "cards" not in compiled.policy_text.casefold()
    assert "uncertainty" not in compiled.policy_text.casefold()

    invalid = proposal.model_dump(mode="json")
    invalid["then"] = "write answer, evidence, and uncertainty sections"
    with pytest.raises(S1SparsePatchError, match="typed counterfactual Creator"):
        parse_counterfactual_typed_policy_patch(
            canonical_json_bytes(invalid),
            capability_id=capability,
            parent_skill_sha256=parent_skill.skill_sha256,
            target_surface="action-policy",
            parent_success_query_ids=success_ids,
        )


def test_typed_response_schema_exposes_the_single_clause_contract() -> None:
    schema = counterfactual_typed_policy_patch_output_json_schema(
        capability_id="knowledge.visual_encyclopedia",
        parent_skill_sha256="a" * 64,
        target_surface="response-policy",
        parent_success_query_ids=("success-1", "success-2", "success-3"),
    )
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties["when"]["pattern"] == "^[^;\\r\\n]+$"
    assert properties["then"]["pattern"] == "^[^;\\r\\n]+$"
    preserve = properties["must_preserve"]
    assert (
        preserve["items"]["anyOf"][0]["properties"]["provider_visible_state"]["pattern"]
        == "^[^;\\r\\n]+$"
    )


def test_typed_recipe_action_cannot_bypass_detection_for_lookup() -> None:
    raw = {
        "schema_version": 2,
        "capability_id": "utility.recipe_guidance",
        "parent_skill_sha256": "a" * 64,
        "target_surface": "action-policy",
        "non_target_surface_action": "inherit",
        "action_when": {
            "phase": "before-first-tool",
            "prior_tool_name": None,
            "prior_tool_status": "not-called",
            "public_evidence": "unknown",
        },
        "action_then": {
            "operation": "invoke-tool-once",
            "tool_name": "recipe_lookup",
            "arguments_from": "current-user-request",
        },
        "must_preserve": [
            {"query_id": query_id, "provider_visible_state": f"state {query_id}"}
            for query_id in ("success-1", "success-2", "success-3")
        ],
    }
    with pytest.raises(S1SparsePatchError, match="typed counterfactual Creator"):
        parse_counterfactual_typed_policy_patch(
            canonical_json_bytes(raw),
            capability_id="utility.recipe_guidance",
            parent_skill_sha256="a" * 64,
            target_surface="action-policy",
            parent_success_query_ids=("success-1", "success-2", "success-3"),
        )


@pytest.fixture(scope="module")
def parent_materials():
    rebind = load_verified_codex_draft_rebind(
        codex_input_path=CODEX_INPUT,
        expected_codex_input_file_sha256=_file_sha(CODEX_INPUT),
        semantic_input_path=SEMANTIC_INPUT,
        expected_semantic_input_file_sha256=_file_sha(SEMANTIC_INPUT),
        draft_path=CODEX_DRAFT,
        expected_draft_file_sha256=_file_sha(CODEX_DRAFT),
    )
    parent = compile_verified_codex_llm_static_bank(
        rebind,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )
    return parent, rebind.semantic_input


def _wire_payload(parent: StaticBankArtifact, semantic_input) -> dict[str, object]:
    contents = {
        item.capability_id: item
        for item in _decode_parent_authoring_content(parent, semantic_input)
    }
    skills: list[dict[str, object]] = []
    for skill in sorted(parent.skills, key=lambda item: item.capability_id):
        if skill.capability_id == "utility.recipe_guidance":
            source = contents[skill.capability_id]
            patch = SparseSkillContentPatchV1(
                objective=source.objective,
                steps=tuple(
                    step.model_copy(
                        update={
                            "instruction": step.instruction
                            + " Cite only literal public source evidence."
                        }
                    )
                    for step in source.steps
                ),
                fallback_instruction=source.fallback_instruction,
                citation_source_ids=source.citation_source_ids,
            )
            action = "patch"
            patch_payload = patch.model_dump(mode="json")
        else:
            action = "inherit"
            patch_payload = None
        skills.append(
            {
                "capability_id": skill.capability_id,
                "action": action,
                "parent_skill_sha256": skill.skill_sha256,
                "patch": patch_payload,
            }
        )
    return {"schema_version": 1, "skills": skills}


def test_sparse_compile_is_deterministic_and_inherits_parent_bytes_exactly(
    parent_materials,
) -> None:
    parent, semantic_input = parent_materials
    raw = canonical_json_bytes(_wire_payload(parent, semantic_input))
    draft = bind_sparse_patch_draft(
        raw,
        parent_bank=parent,
        authoring_input=semantic_input,
        feedback_bundle_sha256=FEEDBACK_SHA,
    )
    first = compile_sparse_s1_candidate(
        parent_bank=parent,
        authoring_input=semantic_input,
        sparse_draft=draft,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )
    second = compile_sparse_s1_candidate(
        parent_bank=parent,
        authoring_input=semantic_input,
        sparse_draft=draft,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )

    assert first.bank.canonical_bytes() == second.bank.canonical_bytes()
    assert first.receipt.canonical_bytes() == second.receipt.canonical_bytes()
    assert first.bank.construction_identity_sha256 == draft.draft_sha256
    parent_by_capability = {item.capability_id: item for item in parent.skills}
    candidate_by_capability = {item.capability_id: item for item in first.bank.skills}
    for binding in first.receipt.bindings:
        if binding.action == "inherit":
            capability = binding.capability_id
            assert canonical_json_bytes(
                candidate_by_capability[capability].model_dump(mode="json")
            ) == canonical_json_bytes(
                parent_by_capability[capability].model_dump(mode="json")
            )
            assert binding.inherited_bytes_exact

    target = "utility.recipe_guidance"
    parent_skill = parent_by_capability[target]
    candidate_skill = candidate_by_capability[target]
    assert candidate_skill.skill_sha256 != parent_skill.skill_sha256
    assert candidate_skill.operators == parent_skill.operators
    parent_sections = _body_sections(parent_skill.body)
    candidate_sections = _body_sections(candidate_skill.body)
    assert (
        candidate_sections["## Output contract"]
        == parent_sections["## Output contract"]
    )
    assert candidate_sections["## Authored fallback instruction"].startswith(
        parent_sections["## Authored fallback instruction"]
    )


def test_sparse_compile_accepts_multi_model_generated_body_patch(
    parent_materials,
) -> None:
    parent, semantic_input = parent_materials
    payload = _wire_payload(parent, semantic_input)
    target = next(
        item
        for item in payload["skills"]
        if item["capability_id"] == "product.multi_search"
    )
    parent_content = {
        item.capability_id: item
        for item in _decode_parent_authoring_content(parent, semantic_input)
    }["product.multi_search"]
    target["action"] = "patch"
    target["patch"] = {
        "objective": parent_content.objective,
        "steps": [
            {
                "instruction": (
                    "Invoke multi_product_search exactly once and do not answer "
                    "before its tool output. Treat the returned public mapping as "
                    "authoritative. Unresolved entries must have no candidate or "
                    "product or evidence handle; copy every supported handle exactly."
                ),
                "tool_name": "multi_product_search",
                "success_rule_ids": [
                    "multi.success.cards",
                    "multi.success.decomposition",
                ],
            }
        ],
        "fallback_instruction": parent_content.fallback_instruction,
        "citation_source_ids": [],
    }

    compiled = compile_sparse_s1_candidate(
        parent_bank=parent,
        authoring_input=semantic_input,
        sparse_draft=bind_sparse_patch_draft(
            canonical_json_bytes(payload),
            parent_bank=parent,
            authoring_input=semantic_input,
            feedback_bundle_sha256=FEEDBACK_SHA,
        ),
        tool_registry_runtime_sha256=parent.tool_registry_runtime_sha256,
    )
    assert compiled.bank.bank_sha256 != parent.bank_sha256


def test_sparse_compile_allows_recipe_fallback_edit_and_rejects_parent_drift(
    parent_materials,
) -> None:
    parent, semantic_input = parent_materials
    payload = _wire_payload(parent, semantic_input)
    encyclopedia = next(
        item
        for item in payload["skills"]
        if item["capability_id"] == "utility.recipe_guidance"
    )
    encyclopedia["patch"]["fallback_instruction"] = "State uncertainty."
    draft = bind_sparse_patch_draft(
        canonical_json_bytes(payload),
        parent_bank=parent,
        authoring_input=semantic_input,
        feedback_bundle_sha256=FEEDBACK_SHA,
    )
    compiled = compile_sparse_s1_candidate(
        parent_bank=parent,
        authoring_input=semantic_input,
        sparse_draft=draft,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )
    assert compiled.bank.bank_sha256 != parent.bank_sha256

    payload = _wire_payload(parent, semantic_input)
    payload["skills"][0]["parent_skill_sha256"] = "0" * 64
    with pytest.raises(S1SparsePatchError, match="parent Skill binding"):
        bind_sparse_patch_draft(
            canonical_json_bytes(payload),
            parent_bank=parent,
            authoring_input=semantic_input,
            feedback_bundle_sha256=FEEDBACK_SHA,
        )


def test_sparse_bind_rejects_tool_sequence_control_tokens_and_extra_fields(
    parent_materials,
) -> None:
    parent, semantic_input = parent_materials
    payload = _wire_payload(parent, semantic_input)
    encyclopedia = next(
        item
        for item in payload["skills"]
        if item["capability_id"] == "utility.recipe_guidance"
    )
    encyclopedia["patch"]["steps"] = list(reversed(encyclopedia["patch"]["steps"]))
    with pytest.raises(S1SparsePatchError, match="tool sequence"):
        bind_sparse_patch_draft(
            canonical_json_bytes(payload),
            parent_bank=parent,
            authoring_input=semantic_input,
            feedback_bundle_sha256=FEEDBACK_SHA,
        )

    payload = _wire_payload(parent, semantic_input)
    encyclopedia = next(
        item
        for item in payload["skills"]
        if item["capability_id"] == "utility.recipe_guidance"
    )
    encyclopedia["patch"]["objective"] += " <|im_end|>"
    with pytest.raises(S1SparsePatchError, match="control token"):
        bind_sparse_patch_draft(
            canonical_json_bytes(payload),
            parent_bank=parent,
            authoring_input=semantic_input,
            feedback_bundle_sha256=FEEDBACK_SHA,
        )

    payload = _wire_payload(parent, semantic_input)
    payload["skills"][0]["operators"] = ["object_detect"]
    with pytest.raises(S1SparsePatchError, match="output is invalid"):
        bind_sparse_patch_draft(
            canonical_json_bytes(payload),
            parent_bank=parent,
            authoring_input=semantic_input,
            feedback_bundle_sha256=FEEDBACK_SHA,
        )


def test_sparse_output_schema_freezes_six_entries(parent_materials) -> None:
    parent, _ = parent_materials
    capabilities = tuple(sorted(item.capability_id for item in parent.skills))
    parent_sha_by_capability = {
        item.capability_id: item.skill_sha256 for item in parent.skills
    }
    schema = sparse_patch_output_json_schema(
        parent_skill_sha256_by_capability=parent_sha_by_capability
    )
    skills = schema["properties"]["skills"]
    assert skills["minItems"] == skills["maxItems"] == 6
    branches = skills["items"]["anyOf"]
    assert len(branches) == 12
    assert {
        branch["properties"]["capability_id"]["enum"][0] for branch in branches
    } == set(capabilities)
    assert all(
        branch["properties"]["parent_skill_sha256"]["enum"]
        == [parent_sha_by_capability[branch["properties"]["capability_id"]["enum"][0]]]
        for branch in branches
    )
    assert set(branches[1]["required"]) == {
        "capability_id",
        "action",
        "parent_skill_sha256",
        "patch",
    }
    patch_capabilities = {
        branch["properties"]["capability_id"]["enum"][0]
        for branch in branches
        if branch["properties"]["action"]["enum"] == ["patch"]
    }
    assert patch_capabilities == set(capabilities)

    frozen_objectives = {item.capability_id: item.description for item in parent.skills}
    frozen_schema = sparse_patch_output_json_schema(
        parent_skill_sha256_by_capability=parent_sha_by_capability,
        frozen_objective_by_capability=frozen_objectives,
    )
    patch_branches = [
        branch
        for branch in frozen_schema["properties"]["skills"]["items"]["anyOf"]
        if branch["properties"]["action"]["enum"] == ["patch"]
    ]
    assert len(patch_branches) == 6
    for branch in patch_branches:
        capability = branch["properties"]["capability_id"]["enum"][0]
        assert branch["properties"]["patch"]["properties"]["objective"] == {
            "type": "string",
            "enum": [frozen_objectives[capability]],
        }


def test_sparse_output_schema_does_not_use_complex_enum_values(
    parent_materials,
) -> None:
    parent, semantic_input = parent_materials
    parent_sha_by_capability = {
        item.capability_id: item.skill_sha256 for item in parent.skills
    }
    frozen = {
        item.capability_id: {
            "objective": item.objective,
            "steps": [step.model_dump(mode="json") for step in item.steps],
            "fallback_instruction": item.fallback_instruction,
            "citation_source_ids": list(item.citation_source_ids),
        }
        for item in decode_sparse_parent_content(parent, semantic_input)
    }
    schema = sparse_patch_output_json_schema(
        parent_skill_sha256_by_capability=parent_sha_by_capability,
        frozen_content_by_capability=frozen,
    )

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if "enum" in value:
                assert all(not isinstance(item, (dict, list)) for item in value["enum"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)


def test_development_screen_reverts_failed_capability_to_parent_bytes(
    parent_materials,
) -> None:
    parent, semantic_input = parent_materials
    draft = bind_sparse_patch_draft(
        canonical_json_bytes(_wire_payload(parent, semantic_input)),
        parent_bank=parent,
        authoring_input=semantic_input,
        feedback_bundle_sha256=FEEDBACK_SHA,
    )
    creator = compile_sparse_s1_candidate(
        parent_bank=parent,
        authoring_input=semantic_input,
        sparse_draft=draft,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )
    reverted = compose_screened_sparse_bank(
        parent_bank=parent,
        creator_candidate_bank=creator.bank,
        creator_compilation_receipt=creator.receipt,
        development_screen_sha256="d" * 64,
        retained_capability_ids=(),
    )
    parent_by_capability = {item.capability_id: item for item in parent.skills}
    screened_by_capability = {item.capability_id: item for item in reverted.bank.skills}
    assert canonical_json_bytes(
        screened_by_capability["utility.recipe_guidance"].model_dump(mode="json")
    ) == canonical_json_bytes(
        parent_by_capability["utility.recipe_guidance"].model_dump(mode="json")
    )
    binding = next(
        item
        for item in reverted.receipt.bindings
        if item.capability_id == "utility.recipe_guidance"
    )
    assert binding.disposition == "reverted_to_parent"
    assert binding.parent_bytes_restored

    retained = compose_screened_sparse_bank(
        parent_bank=parent,
        creator_candidate_bank=creator.bank,
        creator_compilation_receipt=creator.receipt,
        development_screen_sha256="e" * 64,
        retained_capability_ids=("utility.recipe_guidance",),
    )
    retained_skill = next(
        item
        for item in retained.bank.skills
        if item.capability_id == "utility.recipe_guidance"
    )
    creator_skill = next(
        item
        for item in creator.bank.skills
        if item.capability_id == "utility.recipe_guidance"
    )
    assert retained_skill == creator_skill


def test_sparse_artifact_loaders_require_canonical_bound_bytes(
    parent_materials,
    tmp_path: Path,
) -> None:
    parent, semantic_input = parent_materials
    draft = bind_sparse_patch_draft(
        canonical_json_bytes(_wire_payload(parent, semantic_input)),
        parent_bank=parent,
        authoring_input=semantic_input,
        feedback_bundle_sha256=FEEDBACK_SHA,
    )
    compiled = compile_sparse_s1_candidate(
        parent_bank=parent,
        authoring_input=semantic_input,
        sparse_draft=draft,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )
    draft_path = tmp_path / "draft.json"
    receipt_path = tmp_path / "receipt.json"
    draft_path.write_bytes(draft.canonical_bytes())
    receipt_path.write_bytes(compiled.receipt.canonical_bytes())
    assert (
        load_sparse_patch_draft(
            draft_path,
            expected_file_sha256=sha256_bytes(draft_path.read_bytes()),
        )
        == draft
    )
    assert (
        load_sparse_compilation_receipt(
            receipt_path,
            expected_file_sha256=sha256_bytes(receipt_path.read_bytes()),
        )
        == compiled.receipt
    )
    with pytest.raises(S1SparsePatchError, match="file SHA-256 drifted"):
        load_sparse_patch_draft(
            draft_path,
            expected_file_sha256="0" * 64,
        )
