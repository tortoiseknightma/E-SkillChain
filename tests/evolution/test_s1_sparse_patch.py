from __future__ import annotations

from pathlib import Path

import pytest

from skillchain.evaluation.portfolio_treatments import (
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
)
from skillchain.evolution.s1_sparse_patch import (
    ENCYCLOPEDIA_CAPABILITY,
    S1SparsePatchError,
    SparseSkillContentPatchV1,
    bind_sparse_patch_draft,
    compile_sparse_s1_candidate,
    compose_screened_sparse_bank,
    load_sparse_compilation_receipt,
    load_sparse_patch_draft,
    sparse_patch_output_json_schema,
    _body_sections,
    _decode_parent_authoring_content,
)
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
CODEX_INPUT = ROOT / "specs/authoring/authoring-packet-codex-high-v5.json"
SEMANTIC_INPUT = (
    ROOT / "specs/authoring/authoring-packet-primary-v5-candidate.json"
)
CODEX_DRAFT = (
    ROOT
    / "runs/formal-authoring/llm-static-codex-primary-20260724-high-v5"
    / "pre-review-draft.json"
)
RUNTIME_SHA = "a" * 64
FEEDBACK_SHA = "f" * 64


def _file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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
        if skill.capability_id == ENCYCLOPEDIA_CAPABILITY:
            source = contents[skill.capability_id]
            patch = SparseSkillContentPatchV1(
                objective=source.objective + " Keep unresolved identity explicit.",
                steps=source.steps,
                fallback_instruction=(
                    source.fallback_instruction
                    + " If there is not enough evidence, use that exact phrase."
                ),
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
    candidate_by_capability = {
        item.capability_id: item for item in first.bank.skills
    }
    for binding in first.receipt.bindings:
        if binding.action == "inherit":
            capability = binding.capability_id
            assert canonical_json_bytes(
                candidate_by_capability[capability].model_dump(mode="json")
            ) == canonical_json_bytes(
                parent_by_capability[capability].model_dump(mode="json")
            )
            assert binding.inherited_bytes_exact

    parent_encyclopedia = parent_by_capability[ENCYCLOPEDIA_CAPABILITY]
    candidate_encyclopedia = candidate_by_capability[ENCYCLOPEDIA_CAPABILITY]
    assert candidate_encyclopedia.skill_sha256 != parent_encyclopedia.skill_sha256
    assert candidate_encyclopedia.operators == parent_encyclopedia.operators
    parent_sections = _body_sections(parent_encyclopedia.body)
    candidate_sections = _body_sections(candidate_encyclopedia.body)
    assert candidate_sections["## Output contract"] == parent_sections[
        "## Output contract"
    ]
    assert "not enough evidence" in candidate_sections[
        "## Authored fallback instruction"
    ].lower()


def test_sparse_bind_rejects_missing_fallback_marker_and_parent_drift(
    parent_materials,
) -> None:
    parent, semantic_input = parent_materials
    payload = _wire_payload(parent, semantic_input)
    encyclopedia = next(
        item
        for item in payload["skills"]
        if item["capability_id"] == ENCYCLOPEDIA_CAPABILITY
    )
    encyclopedia["patch"]["fallback_instruction"] = "State uncertainty."
    with pytest.raises(S1SparsePatchError, match="fallback evidence marker"):
        bind_sparse_patch_draft(
            canonical_json_bytes(payload),
            parent_bank=parent,
            authoring_input=semantic_input,
            feedback_bundle_sha256=FEEDBACK_SHA,
        )

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
        if item["capability_id"] == ENCYCLOPEDIA_CAPABILITY
    )
    encyclopedia["patch"]["steps"] = list(
        reversed(encyclopedia["patch"]["steps"])
    )
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
        if item["capability_id"] == ENCYCLOPEDIA_CAPABILITY
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
        branch["properties"]["capability_id"]["enum"][0]
        for branch in branches
    } == set(capabilities)
    assert all(
        branch["properties"]["parent_skill_sha256"]["enum"]
        == [
            parent_sha_by_capability[
                branch["properties"]["capability_id"]["enum"][0]
            ]
        ]
        for branch in branches
    )
    assert set(branches[1]["required"]) == {
        "capability_id",
        "action",
        "parent_skill_sha256",
        "patch",
    }


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
    screened_by_capability = {
        item.capability_id: item for item in reverted.bank.skills
    }
    assert canonical_json_bytes(
        screened_by_capability[ENCYCLOPEDIA_CAPABILITY].model_dump(mode="json")
    ) == canonical_json_bytes(
        parent_by_capability[ENCYCLOPEDIA_CAPABILITY].model_dump(mode="json")
    )
    binding = next(
        item
        for item in reverted.receipt.bindings
        if item.capability_id == ENCYCLOPEDIA_CAPABILITY
    )
    assert binding.disposition == "reverted_to_parent"
    assert binding.parent_bytes_restored

    retained = compose_screened_sparse_bank(
        parent_bank=parent,
        creator_candidate_bank=creator.bank,
        creator_compilation_receipt=creator.receipt,
        development_screen_sha256="e" * 64,
        retained_capability_ids=(ENCYCLOPEDIA_CAPABILITY,),
    )
    retained_skill = next(
        item
        for item in retained.bank.skills
        if item.capability_id == ENCYCLOPEDIA_CAPABILITY
    )
    creator_skill = next(
        item
        for item in creator.bank.skills
        if item.capability_id == ENCYCLOPEDIA_CAPABILITY
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
    assert load_sparse_patch_draft(
        draft_path,
        expected_file_sha256=sha256_bytes(draft_path.read_bytes()),
    ) == draft
    assert load_sparse_compilation_receipt(
        receipt_path,
        expected_file_sha256=sha256_bytes(receipt_path.read_bytes()),
    ) == compiled.receipt
    with pytest.raises(S1SparsePatchError, match="file SHA-256 drifted"):
        load_sparse_patch_draft(
            draft_path,
            expected_file_sha256="0" * 64,
        )
