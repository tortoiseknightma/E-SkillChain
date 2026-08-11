from pathlib import Path

import pytest

from scripts import refresh_portfolio_llm_static_style_contract as refresh
from skillchain.evaluation.portfolio_treatments import (
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
CODEX_INPUT = ROOT / "specs" / "authoring" / "authoring-packet-codex-high-v5.json"
SEMANTIC_INPUT = (
    ROOT / "specs" / "authoring" / "authoring-packet-primary-v5-candidate.json"
)
SOURCE_DRAFT = (
    ROOT
    / "runs"
    / "formal-authoring"
    / "llm-static-codex-primary-20260724-high-v5"
    / "pre-review-draft.json"
)


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _source_materials():
    verified = load_verified_codex_draft_rebind(
        codex_input_path=CODEX_INPUT,
        expected_codex_input_file_sha256=_sha(CODEX_INPUT),
        semantic_input_path=SEMANTIC_INPUT,
        expected_semantic_input_file_sha256=_sha(SEMANTIC_INPUT),
        draft_path=SOURCE_DRAFT,
        expected_draft_file_sha256=_sha(SOURCE_DRAFT),
    )
    bank = compile_verified_codex_llm_static_bank(
        verified,
        tool_registry_runtime_sha256="a" * 64,
    )
    return verified, bank


def test_contract_refresh_changes_only_style_and_self_hashes_receipt() -> None:
    verified, old_bank = _source_materials()

    semantic, draft, bank, receipt = refresh.build_contract_refresh(
        old_semantic=verified.semantic_input,
        old_draft=verified.source_draft,
        old_semantic_file_sha256=_sha(SEMANTIC_INPUT),
        old_draft_file_sha256=_sha(SOURCE_DRAFT),
        old_bank=old_bank,
        old_bank_file_sha256=sha256_bytes(old_bank.canonical_bytes()),
        tool_registry_runtime_sha256="b" * 64,
    )

    old_skills = {item.capability_id: item for item in old_bank.skills}
    new_skills = {item.capability_id: item for item in bank.skills}
    assert set(old_skills) == set(new_skills)
    assert (
        new_skills["product.style_recommendation"]
        != old_skills["product.style_recommendation"]
    )
    for capability_id in sorted(old_skills):
        if capability_id != "product.style_recommendation":
            assert canonical_json_bytes(
                new_skills[capability_id].model_dump(mode="json")
            ) == canonical_json_bytes(old_skills[capability_id].model_dump(mode="json"))
    assert receipt["provider_calls"] == 0
    assert receipt["algorithm_gain_eligible"] is False
    assert receipt["style_tool_version"] == "2.3.0"
    assert len(receipt["unchanged_skill_sha256s"]) == 5
    assert receipt["new_semantic_authoring_input_sha256"] == semantic.input_sha256
    assert receipt["new_draft_sha256"] == draft.bundle_sha256
    assert receipt["new_bank_sha256"] == bank.bank_sha256
    unsigned = dict(receipt)
    supplied = unsigned.pop("receipt_sha256")
    assert supplied == sha256_bytes(canonical_json_bytes(unsigned))


def test_contract_refresh_rejects_unbound_source_bank() -> None:
    verified, old_bank = _source_materials()

    with pytest.raises(ValueError, match="source LLMStatic lineage"):
        refresh.build_contract_refresh(
            old_semantic=verified.semantic_input,
            old_draft=verified.source_draft,
            old_semantic_file_sha256=_sha(SEMANTIC_INPUT),
            old_draft_file_sha256=_sha(SOURCE_DRAFT),
            old_bank=old_bank.model_copy(
                update={"construction_identity_sha256": "f" * 64}
            ),
            old_bank_file_sha256=sha256_bytes(old_bank.canonical_bytes()),
            tool_registry_runtime_sha256="b" * 64,
        )
