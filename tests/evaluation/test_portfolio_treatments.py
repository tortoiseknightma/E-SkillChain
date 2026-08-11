from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from skillchain.evaluation.packets import VisibleCard
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
    PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION,
    PORTFOLIO_GATE_DECISION_RULE,
    PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION,
    PORTFOLIO_RUNTIME_COMPATIBILITY_REBIND_POLICY_VERSION,
    PORTFOLIO_STAGE_MUTATION_POLICY_VERSION,
    PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
    PortfolioArtifactBinding,
    PortfolioBankBinding,
    PortfolioModelInvocationReceipt,
    PortfolioRuntimeCompatibilityRebind,
    PortfolioRuntimeFileBinding,
    PortfolioRuntimeRebindBankBinding,
    PortfolioSkillMutation,
    PortfolioStageMutation,
    PortfolioStageGateReport,
    PortfolioStageGateResultSet,
    PortfolioTreatmentChainManifest,
    PortfolioTreatmentError,
    PortfolioTreatmentReceipt,
    apply_portfolio_stage_mutation,
    build_portfolio_model_invocation_receipt,
    build_portfolio_execution_artifact_aliases,
    build_portfolio_stage_gate_report,
    build_portfolio_stage_gate_result_set,
    build_portfolio_stage_mutation,
    calculate_portfolio_skill_adherence,
    compile_portfolio_s1_creator_bank,
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
    parse_portfolio_skill_output_contract,
    rebind_portfolio_bank_runtime,
    require_verified_portfolio_treatment_chain,
    validate_portfolio_bank_transition,
    verify_portfolio_stage_gate_evidence,
    verify_portfolio_runtime_compatibility_rebind_chain,
    verify_portfolio_treatment_chain,
)
from skillchain.tools.registry import build_mvp_registry_spec
from skillchain.static_authoring import (
    StaticBankArtifact,
    build_authoring_draft_bundle,
    build_capability_draft,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
CODEX_INPUT = ROOT / "specs" / "authoring" / "authoring-packet-codex-high-v5.json"
SEMANTIC_INPUT = (
    ROOT / "specs" / "authoring" / "authoring-packet-primary-v5-candidate.json"
)
CODEX_DRAFT = (
    ROOT
    / "runs"
    / "formal-authoring"
    / "llm-static-codex-primary-20260724-high-v5"
    / "pre-review-draft.json"
)
RUNTIME_SHA = "a" * 64
IMPLEMENTATION_SHA = "b" * 64
ARTIFACT_SHA = "c" * 64
GATE_SHA = "d" * 64
RAW_SHA = "e" * 64
INVOCATION_SHA = "f" * 64
RUBRIC_SHA = PORTFOLIO_FINAL_RUBRIC_FILE_SHA256
GATE_QUERY_IDS = tuple(f"dm-{index:03d}" for index in range(1, 26))
SHARED_ROUTE_SHAS = tuple(
    sha256_bytes(f"shared-route:{query_id}".encode()) for query_id in GATE_QUERY_IDS
)


def _file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@pytest.fixture(scope="module")
def verified_rebind():
    return load_verified_codex_draft_rebind(
        codex_input_path=CODEX_INPUT,
        expected_codex_input_file_sha256=_file_sha(CODEX_INPUT),
        semantic_input_path=SEMANTIC_INPUT,
        expected_semantic_input_file_sha256=_file_sha(SEMANTIC_INPUT),
        draft_path=CODEX_DRAFT,
        expected_draft_file_sha256=_file_sha(CODEX_DRAFT),
    )


@pytest.fixture(scope="module")
def llm_static_bank(verified_rebind) -> StaticBankArtifact:
    return compile_verified_codex_llm_static_bank(
        verified_rebind,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )


def test_runtime_rebind_changes_only_registry_commitments_and_bank_hash(
    llm_static_bank: StaticBankArtifact,
) -> None:
    registry = build_mvp_registry_spec(include_multi_product=True)

    rebound = rebind_portfolio_bank_runtime(llm_static_bank, registry)

    assert rebound.tool_registry_sha256 == registry.registry_sha256
    assert rebound.tool_registry_runtime_sha256 == registry.registry_runtime_sha256
    assert rebound.bank_sha256 != llm_static_bank.bank_sha256
    assert rebound.skills == llm_static_bank.skills
    assert rebound.capability_map == llm_static_bank.capability_map
    assert rebound.construction_identity_sha256 == (
        llm_static_bank.construction_identity_sha256
    )


def test_runtime_rebind_rejects_removed_operator(
    llm_static_bank: StaticBankArtifact,
) -> None:
    registry = build_mvp_registry_spec(include_multi_product=False)

    with pytest.raises(PortfolioTreatmentError, match="removed Bank operators"):
        rebind_portfolio_bank_runtime(llm_static_bank, registry)


def _binding(kind: str) -> PortfolioArtifactBinding:
    return PortfolioArtifactBinding(
        artifact_kind=kind,
        artifact_file=f"inputs/{kind}.json",
        artifact_file_sha256=ARTIFACT_SHA,
        artifact_content_sha256=ARTIFACT_SHA,
    )


def _s2_mutation(parent: StaticBankArtifact) -> PortfolioStageMutation:
    source = next(
        item for item in parent.skills if item.capability_id == "product.exact_match"
    )
    return build_portfolio_stage_mutation(
        config="s1s2",
        parent_bank_sha256=parent.bank_sha256,
        implementation_id="portfolio-route-optimizer",
        implementation_version="1.0.0",
        implementation_file_sha256=IMPLEMENTATION_SHA,
        input_artifacts=(_binding("route_examples"),),
        changes=(
            PortfolioSkillMutation(
                capability_id=source.capability_id,
                parent_skill_sha256=source.skill_sha256,
                description=(
                    "Use only when the user requests the same catalog product."
                ),
            ),
        ),
    )


def _distinct_s1_bank(verified_rebind) -> StaticBankArtifact:
    source = verified_rebind.rebound_draft.drafts[0]
    changed = build_capability_draft(
        capability_id=source.capability_id,
        objective=source.objective + " Verify catalog evidence before answering.",
        steps=source.steps,
        fallback_instruction=source.fallback_instruction,
        fallback_may_request_clarification=(source.fallback_may_request_clarification),
        fallback_must_state_uncertainty=(source.fallback_must_state_uncertainty),
        citation_source_ids=source.citation_source_ids,
        rule_coverage=source.rule_coverage,
        output_coverage=source.output_coverage,
    )
    bundle = build_authoring_draft_bundle(
        authoring_input_sha256=verified_rebind.semantic_input.input_sha256,
        drafts=(changed, *verified_rebind.rebound_draft.drafts[1:]),
    )
    return compile_portfolio_s1_creator_bank(
        verified_rebind.semantic_input,
        bundle,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )


def _s3_mutation(parent: StaticBankArtifact) -> PortfolioStageMutation:
    source = next(
        item for item in parent.skills if item.capability_id == "product.exact_match"
    )
    return build_portfolio_stage_mutation(
        config="full",
        parent_bank_sha256=parent.bank_sha256,
        implementation_id="portfolio-body-refiner",
        implementation_version="1.0.0",
        implementation_file_sha256=IMPLEMENTATION_SHA,
        input_artifacts=(_binding("body_attribution"),),
        changes=(
            PortfolioSkillMutation(
                capability_id=source.capability_id,
                parent_skill_sha256=source.skill_sha256,
                body=source.body.rstrip()
                + "\n\n## Refined response policy\n\n"
                + "Lead with the evidence-backed result.\n",
            ),
        ),
    )


def _invocation(config: str) -> PortfolioModelInvocationReceipt:
    stage = {
        "llm_static": "static_author",
        "s1": "s1_creator",
        "s1s2": "s2_route_optimizer",
        "full": "s3_body_refiner",
    }[config]
    return build_portfolio_model_invocation_receipt(
        stage=stage,
        requested_model="gpt-5.6-sol",
        effort="high",
        thread_id=f"thread-{config}",
        prompt_sha256="2" * 64,
        output_sha256=RAW_SHA,
        event_log_sha256="3" * 64,
        stderr_sha256="4" * 64,
        input_tokens=100,
        output_tokens=20,
    )


def _gate(
    *,
    config: str,
    parent: StaticBankArtifact,
    candidate: StaticBankArtifact,
    decision: str = "accepted",
) -> PortfolioStageGateReport:
    if decision == "accepted":
        candidate_route_accuracy = 0.6
        candidate_mean_j = 51.0
        candidate_hard_error_count = 0
        candidate_mean_skill_adherence = 0.9
    else:
        candidate_route_accuracy = 0.4
        candidate_mean_j = 49.0
        candidate_hard_error_count = 2
        candidate_mean_skill_adherence = 0.7
    return build_portfolio_stage_gate_report(
        config=config,
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate.bank_sha256,
        adherence_contract_bank_sha256=parent.bank_sha256,
        evaluation_query_ids=GATE_QUERY_IDS,
        rubric_file_sha256=RUBRIC_SHA,
        paired_shared_route_artifact_sha256s=(
            SHARED_ROUTE_SHAS if config == "full" else None
        ),
        parent_route_accuracy=0.5,
        candidate_route_accuracy=candidate_route_accuracy,
        parent_mean_j=50.0,
        candidate_mean_j=candidate_mean_j,
        parent_mean_skill_adherence=0.8,
        candidate_mean_skill_adherence=candidate_mean_skill_adherence,
        parent_hard_error_count=1,
        candidate_hard_error_count=candidate_hard_error_count,
        decision=decision,
        parent_result_file=f"gate/{config}-parent-results.json",
        parent_result_file_sha256="5" * 64,
        candidate_result_file=f"gate/{config}-candidate-results.json",
        candidate_result_file_sha256="6" * 64,
    )


def _receipt(
    *,
    config: str,
    common_input_sha256: str,
    candidate: StaticBankArtifact,
    output: StaticBankArtifact,
    parent: StaticBankArtifact | None = None,
    mutation: PortfolioStageMutation | None = None,
    invocation: PortfolioModelInvocationReceipt | None = None,
    gate: PortfolioStageGateReport | None = None,
    decision: str = "accepted",
) -> PortfolioTreatmentReceipt:
    stage = {
        "llm_static": "static_author",
        "s1": "s1_creator",
        "s1s2": "s2_route_optimizer",
        "full": "s3_body_refiner",
    }[config]
    generation = {
        "llm_static": "codex_static_author",
        "s1": "codex_s1_creator",
        "s1s2": "llm_route_optimizer",
        "full": "llm_body_refiner",
    }[config]
    scope = {
        "llm_static": "create_bank",
        "s1": "create_bank",
        "s1s2": "description_only",
        "full": "body_only",
    }[config]
    input_kinds = {
        "llm_static": ("pre_review_draft",),
        "s1": ("trajectory_bundle", "creator_packet", "engineer_review"),
        "s1s2": ("route_examples",),
        "full": ("body_attribution",),
    }[config]
    candidate_bytes = candidate.canonical_bytes()
    output_bytes = output.canonical_bytes()
    invocation = invocation or _invocation(config)
    gate_report_sha256 = gate.gate_report_sha256 if gate is not None else GATE_SHA
    gate_report_file_sha256 = (
        sha256_bytes(gate.canonical_bytes()) if gate is not None else GATE_SHA
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-treatment-receipt",
        "policy_version": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
        "config": config,
        "stage": stage,
        "generation_kind": generation,
        "edit_scope": scope,
        "common_authoring_input_sha256": common_input_sha256,
        "implementation_id": f"portfolio-{stage}",
        "implementation_version": "1.0.0",
        "implementation_file_sha256": IMPLEMENTATION_SHA,
        "input_artifacts": [
            _binding(kind).model_dump(mode="json") for kind in input_kinds
        ],
        "parent_bank_sha256": parent.bank_sha256 if parent is not None else None,
        "mutation_sha256": mutation.mutation_sha256 if mutation else None,
        "mutation_file": f"mutations/{config}.json" if mutation else None,
        "mutation_file_sha256": (
            sha256_bytes(mutation.canonical_bytes()) if mutation else None
        ),
        "candidate_bank_sha256": candidate.bank_sha256,
        "candidate_bank_file": f"candidates/bank-{config}.json",
        "candidate_bank_file_sha256": sha256_bytes(candidate_bytes),
        "output_bank_sha256": output.bank_sha256,
        "output_bank_file_sha256": sha256_bytes(output_bytes),
        "model_call_count": 1,
        "invocation_receipt_file": f"invocations/{config}.json",
        "invocation_receipt_file_sha256": sha256_bytes(invocation.canonical_bytes()),
        "raw_model_output_file": f"raw/{config}.txt",
        "raw_model_output_file_sha256": RAW_SHA,
        "gate_report_file": f"gate/{config}.json",
        "gate_report_sha256": gate_report_sha256,
        "gate_report_file_sha256": gate_report_file_sha256,
        "decision": decision,
    }
    return PortfolioTreatmentReceipt.model_validate(
        {
            **payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


def _manifest(
    records: tuple[PortfolioTreatmentReceipt, ...],
    banks: tuple[StaticBankArtifact, ...],
    *,
    status: str = "ready_for_matrix",
) -> PortfolioTreatmentChainManifest:
    bindings = tuple(
        PortfolioBankBinding(
            config=record.config,
            bank_file=f"bank-{record.config}.json",
            bank_sha256=bank.bank_sha256,
            bank_file_sha256=sha256_bytes(bank.canonical_bytes()),
        )
        for record, bank in zip(records, banks, strict=True)
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-treatment-chain-manifest",
        "policy_version": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
        "status": status,
        "track": "portfolio",
        "formal_eligible": False,
        "optimization_query_ids": list(GATE_QUERY_IDS),
        "evaluation_query_ids": ["dm-026", "dm-027"],
        "optimization_leakage_group_ids": ["lg-opt-1", "lg-opt-2"],
        "evaluation_leakage_group_ids": ["lg-eval-1", "lg-eval-2"],
        "records": [item.model_dump(mode="json") for item in records],
        "banks": [item.model_dump(mode="json") for item in bindings],
    }
    return PortfolioTreatmentChainManifest.model_validate(
        {
            **payload,
            "chain_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


def test_verified_codex_rebind_changes_only_the_input_binding(
    verified_rebind,
) -> None:
    assert (
        verified_rebind.source_draft.authoring_input_sha256
        == verified_rebind.codex_input.input_sha256
    )
    assert (
        verified_rebind.rebound_draft.authoring_input_sha256
        == verified_rebind.semantic_input.input_sha256
    )
    assert verified_rebind.source_draft.drafts == verified_rebind.rebound_draft.drafts
    assert verified_rebind.codex_input.semantic_source_packet_file_sha256 == _file_sha(
        SEMANTIC_INPUT
    )


def test_verified_codex_rebind_rejects_an_external_digest_mismatch() -> None:
    with pytest.raises(PortfolioTreatmentError, match="digest mismatch"):
        load_verified_codex_draft_rebind(
            codex_input_path=CODEX_INPUT,
            expected_codex_input_file_sha256="0" * 64,
            semantic_input_path=SEMANTIC_INPUT,
            expected_semantic_input_file_sha256=_file_sha(SEMANTIC_INPUT),
            draft_path=CODEX_DRAFT,
            expected_draft_file_sha256=_file_sha(CODEX_DRAFT),
        )


def test_real_llm_static_compiler_uses_the_rebound_bundle(
    verified_rebind,
    llm_static_bank: StaticBankArtifact,
) -> None:
    assert len(llm_static_bank.skills) == 6
    assert (
        llm_static_bank.construction_identity_sha256
        == verified_rebind.rebound_draft.bundle_sha256
    )
    assert llm_static_bank.tool_registry_runtime_sha256 == RUNTIME_SHA
    assert all(
        item.version == 1 and item.parent_skill_sha256 is None
        for item in llm_static_bank.skills
    )
    assert all("## Success criteria" in item.body for item in llm_static_bank.skills)


def test_s1_creator_uses_the_same_compiler_and_requires_semantic_binding(
    verified_rebind,
) -> None:
    bank = compile_portfolio_s1_creator_bank(
        verified_rebind.semantic_input,
        verified_rebind.rebound_draft,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )
    assert len(bank.skills) == 6
    assert all(
        item.version == 1 and item.parent_skill_sha256 is None for item in bank.skills
    )
    with pytest.raises(PortfolioTreatmentError, match="common semantic input"):
        compile_portfolio_s1_creator_bank(
            verified_rebind.semantic_input,
            verified_rebind.source_draft,
            tool_registry_runtime_sha256=RUNTIME_SHA,
        )


def test_s2_and_s3_mutations_enforce_field_scope_and_parent_lineage(
    llm_static_bank: StaticBankArtifact,
) -> None:
    s2_plan = _s2_mutation(llm_static_bank)
    s2 = apply_portfolio_stage_mutation(llm_static_bank, s2_plan)
    validate_portfolio_bank_transition(
        llm_static_bank,
        s2,
        "description_only",
    )
    source_by_capability = {item.capability_id: item for item in llm_static_bank.skills}
    s2_by_capability = {item.capability_id: item for item in s2.skills}
    changed = s2_by_capability["product.exact_match"]
    source = source_by_capability["product.exact_match"]
    assert changed.description != source.description
    assert changed.body == source.body
    assert changed.version == source.version + 1
    assert changed.parent_skill_sha256 == source.skill_sha256

    s3_plan = _s3_mutation(s2)
    full = apply_portfolio_stage_mutation(s2, s3_plan)
    validate_portfolio_bank_transition(s2, full, "body_only")
    full_by_capability = {item.capability_id: item for item in full.skills}
    refined = full_by_capability["product.exact_match"]
    assert refined.description == changed.description
    assert refined.body != changed.body
    assert refined.parent_skill_sha256 == changed.skill_sha256


def test_s2_rejects_a_body_mutation(
    llm_static_bank: StaticBankArtifact,
) -> None:
    source = llm_static_bank.skills[0]
    body_change = PortfolioSkillMutation(
        capability_id=source.capability_id,
        parent_skill_sha256=source.skill_sha256,
        body=source.body.rstrip() + "\n\nChanged.\n",
    )
    with pytest.raises(PortfolioTreatmentError, match="stage mutation is invalid"):
        build_portfolio_stage_mutation(
            config="s1s2",
            parent_bank_sha256=llm_static_bank.bank_sha256,
            implementation_id="route-optimizer",
            implementation_version="1.0.0",
            implementation_file_sha256=IMPLEMENTATION_SHA,
            input_artifacts=(_binding("route_examples"),),
            changes=(body_change,),
        )


def test_receipt_rejects_scaffold_and_missing_engineer_review(
    llm_static_bank: StaticBankArtifact,
) -> None:
    gate = _gate(
        config="s1",
        parent=llm_static_bank,
        candidate=llm_static_bank,
    )
    valid = _receipt(
        config="s1",
        common_input_sha256="1" * 64,
        candidate=llm_static_bank,
        output=llm_static_bank,
        parent=llm_static_bank,
        gate=gate,
    )
    raw = valid.model_dump(mode="json")
    raw["generation_kind"] = "deterministic_scaffold"
    with pytest.raises(ValidationError):
        PortfolioTreatmentReceipt.model_validate(raw, strict=True)

    raw = valid.model_dump(mode="json")
    raw["input_artifacts"] = [
        item
        for item in raw["input_artifacts"]
        if item["artifact_kind"] != "engineer_review"
    ]
    unsigned = dict(raw)
    unsigned.pop("receipt_sha256")
    raw["receipt_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    with pytest.raises(ValidationError, match="Engineer-Gate"):
        PortfolioTreatmentReceipt.model_validate(raw, strict=True)


def test_model_invocation_receipt_requires_one_completed_tool_free_turn() -> None:
    receipt = _invocation("s1")
    assert receipt.model_call_count == 1
    assert receipt.process_exit_code == 0
    assert receipt.turn_completed_count == 1
    assert receipt.agent_message_count == 1
    assert receipt.visible_tool_activity is False

    raw = receipt.model_dump(mode="json")
    raw["visible_tool_activity"] = True
    with pytest.raises(ValidationError):
        PortfolioModelInvocationReceipt.model_validate(raw, strict=True)

    raw = receipt.model_dump(mode="json")
    raw["process_exit_code"] = 1
    with pytest.raises(ValidationError):
        PortfolioModelInvocationReceipt.model_validate(raw, strict=True)


def test_stage_gate_report_derives_accept_or_rollback_from_frozen_rule(
    llm_static_bank: StaticBankArtifact,
) -> None:
    accepted = _gate(
        config="s1",
        parent=llm_static_bank,
        candidate=llm_static_bank,
    )
    assert accepted.decision_rule == PORTFOLIO_GATE_DECISION_RULE
    assert accepted.decision == "accepted"

    with pytest.raises(PortfolioTreatmentError, match="gate report is invalid"):
        build_portfolio_stage_gate_report(
            config="s1",
            parent_bank_sha256=llm_static_bank.bank_sha256,
            candidate_bank_sha256=llm_static_bank.bank_sha256,
            adherence_contract_bank_sha256=llm_static_bank.bank_sha256,
            evaluation_query_ids=GATE_QUERY_IDS,
            rubric_file_sha256=RUBRIC_SHA,
            paired_shared_route_artifact_sha256s=None,
            parent_route_accuracy=0.5,
            candidate_route_accuracy=0.4,
            parent_mean_j=50.0,
            candidate_mean_j=49.0,
            parent_mean_skill_adherence=0.8,
            candidate_mean_skill_adherence=0.7,
            parent_hard_error_count=1,
            candidate_hard_error_count=2,
            decision="accepted",
            parent_result_file="gate/parent.json",
            parent_result_file_sha256="5" * 64,
            candidate_result_file="gate/candidate.json",
            candidate_result_file_sha256="6" * 64,
        )


def test_skill_adherence_uses_parent_contract_without_candidate_slug_bias(
    llm_static_bank: StaticBankArtifact,
) -> None:
    skill = next(
        item
        for item in llm_static_bank.skills
        if item.capability_id == "product.exact_match"
    )
    contract = parse_portfolio_skill_output_contract(skill.body)
    response_text = "\n".join(
        f"{section}: supported" for section in contract.required_sections
    )
    fields = tuple(
        sorted(
            (field, f"visible-{field}")
            for field in contract.card_fields
            if field != "title"
        )
    )
    cards = (VisibleCard(title="Supported product", body="Evidence", fields=fields),)
    assert (
        calculate_portfolio_skill_adherence(
            bank=llm_static_bank,
            selected_capability=skill.capability_id,
            skill_slug=skill.slug,
            response_text=response_text,
            visible_cards=cards,
            hard_error=False,
        )
        == 1.0
    )
    assert (
        calculate_portfolio_skill_adherence(
            bank=llm_static_bank,
            selected_capability=skill.capability_id,
            skill_slug="candidate-renamed-skill",
            response_text=response_text,
            visible_cards=cards,
            hard_error=False,
        )
        == 0.0
    )
    assert (
        calculate_portfolio_skill_adherence(
            bank=llm_static_bank,
            selected_capability=skill.capability_id,
            skill_slug="candidate-renamed-skill",
            response_text=response_text,
            visible_cards=cards,
            hard_error=False,
            require_skill_slug_match=False,
        )
        == 1.0
    )
    assert (
        calculate_portfolio_skill_adherence(
            bank=llm_static_bank,
            selected_capability=skill.capability_id,
            skill_slug=skill.slug,
            response_text=response_text,
            visible_cards=cards,
            hard_error=True,
        )
        == 0.0
    )


def test_skill_adherence_averages_card_fields_across_multiple_cards(
    llm_static_bank: StaticBankArtifact,
) -> None:
    skill = next(
        item
        for item in llm_static_bank.skills
        if item.capability_id == "product.exact_match"
    )
    contract = parse_portfolio_skill_output_contract(skill.body)
    response_text = "\n".join(
        f"{section}: supported" for section in contract.required_sections
    )
    non_title_fields = tuple(
        field for field in contract.card_fields if field != "title"
    )
    assert non_title_fields
    missing_field = non_title_fields[0]

    def card(*, omit: str | None = None) -> VisibleCard:
        fields = tuple(
            sorted(
                (field, f"visible-{field}")
                for field in non_title_fields
                if field != omit
            )
        )
        return VisibleCard(title="Supported product", body="Evidence", fields=fields)

    complete_cards = (card(), card())
    assert (
        calculate_portfolio_skill_adherence(
            bank=llm_static_bank,
            selected_capability=skill.capability_id,
            skill_slug=skill.slug,
            response_text=response_text,
            visible_cards=complete_cards,
            hard_error=False,
        )
        == 1.0
    )

    field_count = len(contract.card_fields)
    expected_card_score = (2 * field_count - 1) / (2 * field_count)
    assert (
        calculate_portfolio_skill_adherence(
            bank=llm_static_bank,
            selected_capability=skill.capability_id,
            skill_slug=skill.slug,
            response_text=response_text,
            visible_cards=(card(), card(omit=missing_field)),
            hard_error=False,
        )
        == (1.0 + expected_card_score) / 2.0
    )


def test_gate_rolls_back_skill_adherence_regression_even_when_j_improves(
    llm_static_bank: StaticBankArtifact,
) -> None:
    arguments = {
        "config": "s1",
        "parent_bank_sha256": llm_static_bank.bank_sha256,
        "candidate_bank_sha256": llm_static_bank.bank_sha256,
        "adherence_contract_bank_sha256": llm_static_bank.bank_sha256,
        "evaluation_query_ids": GATE_QUERY_IDS,
        "rubric_file_sha256": RUBRIC_SHA,
        "paired_shared_route_artifact_sha256s": None,
        "parent_route_accuracy": 0.5,
        "candidate_route_accuracy": 0.5,
        "parent_mean_j": 50.0,
        "candidate_mean_j": 51.0,
        "parent_mean_skill_adherence": 0.8,
        "candidate_mean_skill_adherence": 0.7,
        "parent_hard_error_count": 1,
        "candidate_hard_error_count": 1,
        "parent_result_file": "gate/parent.json",
        "parent_result_file_sha256": "5" * 64,
        "candidate_result_file": "gate/candidate.json",
        "candidate_result_file_sha256": "6" * 64,
    }
    rolled_back = build_portfolio_stage_gate_report(
        **arguments,
        decision="rolled_back",
    )
    assert rolled_back.decision == "rolled_back"
    with pytest.raises(PortfolioTreatmentError, match="gate report is invalid"):
        build_portfolio_stage_gate_report(**arguments, decision="accepted")


def test_gate_is_inconclusive_on_evaluator_anomaly_not_hard_regression(
    llm_static_bank: StaticBankArtifact,
) -> None:
    arguments = {
        "config": "s1",
        "parent_bank_sha256": llm_static_bank.bank_sha256,
        "candidate_bank_sha256": llm_static_bank.bank_sha256,
        "adherence_contract_bank_sha256": llm_static_bank.bank_sha256,
        "evaluation_query_ids": GATE_QUERY_IDS,
        "rubric_file_sha256": RUBRIC_SHA,
        "paired_shared_route_artifact_sha256s": None,
        "parent_route_accuracy": 0.5,
        "candidate_route_accuracy": 0.6,
        "parent_mean_j": 50.0,
        "candidate_mean_j": 51.0,
        "parent_mean_skill_adherence": 0.8,
        "candidate_mean_skill_adherence": 0.9,
        "parent_hard_error_count": 0,
        "candidate_hard_error_count": 0,
        "parent_evaluator_anomaly_count": 0,
        "candidate_evaluator_anomaly_count": 1,
        "parent_result_file": "gate/parent.json",
        "parent_result_file_sha256": "5" * 64,
        "candidate_result_file": "gate/candidate.json",
        "candidate_result_file_sha256": "6" * 64,
    }

    report = build_portfolio_stage_gate_report(
        **arguments,
        decision="inconclusive",
    )

    assert report.decision == "inconclusive"
    assert report.candidate_hard_error_count == 0
    assert report.candidate_evaluator_anomaly_count == 1
    with pytest.raises(PortfolioTreatmentError, match="gate report is invalid"):
        build_portfolio_stage_gate_report(**arguments, decision="rolled_back")


def test_stage_gate_evidence_requires_typed_current_bank_metric_projections(
    llm_static_bank: StaticBankArtifact,
) -> None:
    query_ids = GATE_QUERY_IDS

    def result_set(
        *,
        source_config: str,
        role: str,
        route: float,
        mean_j: float,
        hard: int,
        bank_sha256: str | None = None,
        rubric_file_sha256: str = RUBRIC_SHA,
        shared_routes: tuple[str, ...] | None = None,
    ):
        return build_portfolio_stage_gate_result_set(
            source_config=source_config,
            bank_sha256=bank_sha256 or llm_static_bank.bank_sha256,
            adherence_contract_bank_sha256=llm_static_bank.bank_sha256,
            evaluation_query_ids=query_ids,
            route_accuracy=route,
            mean_j=mean_j,
            mean_skill_adherence=0.8 if role.startswith("parent") else 0.9,
            hard_error_count=hard,
            rubric_file_sha256=rubric_file_sha256,
            shared_route_artifact_sha256s=(
                shared_routes or SHARED_ROUTE_SHAS
                if source_config in {"s1s2", "full"}
                else (None,) * len(query_ids)
            ),
            source_summary_file_sha256=sha256_bytes(f"{role}:summary-file".encode()),
            source_summary_sha256=sha256_bytes(f"{role}:summary".encode()),
            source_results_file_sha256=sha256_bytes(f"{role}:results-file".encode()),
            source_result_sha256s=tuple(
                sha256_bytes(f"{role}:result:{index}".encode())
                for index in range(len(query_ids))
            ),
        )

    parent = result_set(
        source_config="llm_static",
        role="parent",
        route=0.5,
        mean_j=50.0,
        hard=1,
    )
    candidate = result_set(
        source_config="s1",
        role="candidate",
        route=0.6,
        mean_j=51.0,
        hard=0,
    )
    report = build_portfolio_stage_gate_report(
        config="s1",
        parent_bank_sha256=llm_static_bank.bank_sha256,
        candidate_bank_sha256=llm_static_bank.bank_sha256,
        adherence_contract_bank_sha256=llm_static_bank.bank_sha256,
        evaluation_query_ids=query_ids,
        rubric_file_sha256=RUBRIC_SHA,
        paired_shared_route_artifact_sha256s=None,
        parent_route_accuracy=0.5,
        candidate_route_accuracy=0.6,
        parent_mean_j=50.0,
        candidate_mean_j=51.0,
        parent_mean_skill_adherence=0.8,
        candidate_mean_skill_adherence=0.9,
        parent_hard_error_count=1,
        candidate_hard_error_count=0,
        decision="accepted",
        parent_result_file="gate/parent.json",
        parent_result_file_sha256=sha256_bytes(parent.canonical_bytes()),
        candidate_result_file="gate/candidate.json",
        candidate_result_file_sha256=sha256_bytes(candidate.canonical_bytes()),
    )
    verify_portfolio_stage_gate_evidence(report, parent, candidate)

    wrong_result_digest_payload = report.model_dump(
        mode="json",
        exclude={"gate_report_sha256"},
    )
    wrong_result_digest_payload["parent_result_file_sha256"] = "0" * 64
    wrong_result_digest_report = PortfolioStageGateReport.model_validate(
        {
            **wrong_result_digest_payload,
            "gate_report_sha256": sha256_bytes(
                canonical_json_bytes(wrong_result_digest_payload)
            ),
        },
        strict=True,
    )
    with pytest.raises(PortfolioTreatmentError, match="typed parent/candidate"):
        verify_portfolio_stage_gate_evidence(
            wrong_result_digest_report,
            parent,
            candidate,
        )

    stale = result_set(
        source_config="llm_static",
        role="stale-parent",
        route=0.5,
        mean_j=50.0,
        hard=1,
        bank_sha256="0" * 64,
    )
    with pytest.raises(PortfolioTreatmentError, match="typed parent/candidate"):
        verify_portfolio_stage_gate_evidence(report, stale, candidate)

    raw = parent.model_dump(mode="json")
    raw["mean_j"] = 99.0
    with pytest.raises(ValidationError, match="result_set_sha256"):
        PortfolioStageGateResultSet.model_validate(raw, strict=True)

    with pytest.raises(PortfolioTreatmentError, match="result set is invalid"):
        result_set(
            source_config="s1",
            role="wrong-rubric",
            route=0.6,
            mean_j=51.0,
            hard=0,
            rubric_file_sha256="8" * 64,
        )


def test_s3_gate_requires_exact_common_route_per_query(
    llm_static_bank: StaticBankArtifact,
) -> None:
    def result_set(source_config: str, routes: tuple[str, ...]):
        return build_portfolio_stage_gate_result_set(
            source_config=source_config,
            bank_sha256=llm_static_bank.bank_sha256,
            adherence_contract_bank_sha256=llm_static_bank.bank_sha256,
            evaluation_query_ids=GATE_QUERY_IDS,
            route_accuracy=1.0,
            mean_j=50.0 if source_config == "s1s2" else 51.0,
            mean_skill_adherence=0.8 if source_config == "s1s2" else 0.9,
            hard_error_count=0,
            rubric_file_sha256=RUBRIC_SHA,
            shared_route_artifact_sha256s=routes,
            source_summary_file_sha256=sha256_bytes(
                f"{source_config}:summary-file".encode()
            ),
            source_summary_sha256=sha256_bytes(f"{source_config}:summary".encode()),
            source_results_file_sha256=sha256_bytes(
                f"{source_config}:results-file".encode()
            ),
            source_result_sha256s=tuple(
                sha256_bytes(f"{source_config}:result:{index}".encode())
                for index in range(len(GATE_QUERY_IDS))
            ),
        )

    parent = result_set("s1s2", SHARED_ROUTE_SHAS)
    candidate = result_set("full", SHARED_ROUTE_SHAS)
    report = build_portfolio_stage_gate_report(
        config="full",
        parent_bank_sha256=llm_static_bank.bank_sha256,
        candidate_bank_sha256=llm_static_bank.bank_sha256,
        adherence_contract_bank_sha256=llm_static_bank.bank_sha256,
        evaluation_query_ids=GATE_QUERY_IDS,
        rubric_file_sha256=RUBRIC_SHA,
        paired_shared_route_artifact_sha256s=SHARED_ROUTE_SHAS,
        parent_route_accuracy=1.0,
        candidate_route_accuracy=1.0,
        parent_mean_j=50.0,
        candidate_mean_j=51.0,
        parent_mean_skill_adherence=0.8,
        candidate_mean_skill_adherence=0.9,
        parent_hard_error_count=0,
        candidate_hard_error_count=0,
        decision="accepted",
        parent_result_file="gate/parent.json",
        parent_result_file_sha256=sha256_bytes(parent.canonical_bytes()),
        candidate_result_file="gate/candidate.json",
        candidate_result_file_sha256=sha256_bytes(candidate.canonical_bytes()),
    )
    verify_portfolio_stage_gate_evidence(report, parent, candidate)

    drifted_routes = list(SHARED_ROUTE_SHAS)
    drifted_routes[-1] = "9" * 64
    drifted = result_set("full", tuple(drifted_routes))
    drift_report_payload = report.model_dump(
        mode="json",
        exclude={"gate_report_sha256"},
    )
    drift_report_payload["candidate_result_file_sha256"] = sha256_bytes(
        drifted.canonical_bytes()
    )
    drift_report = PortfolioStageGateReport.model_validate(
        {
            **drift_report_payload,
            "gate_report_sha256": sha256_bytes(
                canonical_json_bytes(drift_report_payload)
            ),
        },
        strict=True,
    )
    with pytest.raises(PortfolioTreatmentError, match="exact per-query routes"):
        verify_portfolio_stage_gate_evidence(drift_report, parent, drifted)


def test_ready_chain_reproduces_every_candidate_and_transition(
    verified_rebind,
    llm_static_bank: StaticBankArtifact,
) -> None:
    s1 = _distinct_s1_bank(verified_rebind)
    s2_plan = _s2_mutation(s1)
    s2 = apply_portfolio_stage_mutation(s1, s2_plan)
    s3_plan = _s3_mutation(s2)
    full = apply_portfolio_stage_mutation(s2, s3_plan)
    common = verified_rebind.semantic_input.input_sha256
    invocations = {
        config: _invocation(config) for config in ("llm_static", "s1", "s1s2", "full")
    }
    gates = {
        "s1": _gate(config="s1", parent=llm_static_bank, candidate=s1),
        "s1s2": _gate(config="s1s2", parent=s1, candidate=s2),
        "full": _gate(config="full", parent=s2, candidate=full),
    }
    records = (
        _receipt(
            config="llm_static",
            common_input_sha256=common,
            candidate=llm_static_bank,
            output=llm_static_bank,
            invocation=invocations["llm_static"],
        ),
        _receipt(
            config="s1",
            common_input_sha256=common,
            candidate=s1,
            output=s1,
            parent=llm_static_bank,
            invocation=invocations["s1"],
            gate=gates["s1"],
        ),
        _receipt(
            config="s1s2",
            common_input_sha256=common,
            candidate=s2,
            output=s2,
            parent=s1,
            mutation=s2_plan,
            invocation=invocations["s1s2"],
            gate=gates["s1s2"],
        ),
        _receipt(
            config="full",
            common_input_sha256=common,
            candidate=full,
            output=full,
            parent=s2,
            mutation=s3_plan,
            invocation=invocations["full"],
            gate=gates["full"],
        ),
    )
    banks = (llm_static_bank, s1, s2, full)
    manifest = _manifest(records, banks)
    verified = verify_portfolio_treatment_chain(
        manifest,
        output_banks=dict(zip(("llm_static", "s1", "s1s2", "full"), banks)),
        candidate_banks=dict(zip(("llm_static", "s1", "s1s2", "full"), banks)),
        mutations={"s1s2": s2_plan, "full": s3_plan},
        invocation_receipts=invocations,
        gate_reports=gates,
    )
    assert verified.manifest.status == "ready_for_matrix"
    assert verified.output_banks["full"] == full


def test_s1_gate_can_roll_back_to_the_llm_static_baseline(
    verified_rebind,
    llm_static_bank: StaticBankArtifact,
) -> None:
    s1_candidate = _distinct_s1_bank(verified_rebind)
    s1_output = llm_static_bank
    s2_plan = _s2_mutation(s1_output)
    s2 = apply_portfolio_stage_mutation(s1_output, s2_plan)
    s3_plan = _s3_mutation(s2)
    full = apply_portfolio_stage_mutation(s2, s3_plan)
    common = verified_rebind.semantic_input.input_sha256
    invocations = {
        config: _invocation(config) for config in ("llm_static", "s1", "s1s2", "full")
    }
    gates = {
        "s1": _gate(
            config="s1",
            parent=llm_static_bank,
            candidate=s1_candidate,
            decision="rolled_back",
        ),
        "s1s2": _gate(config="s1s2", parent=s1_output, candidate=s2),
        "full": _gate(config="full", parent=s2, candidate=full),
    }
    records = (
        _receipt(
            config="llm_static",
            common_input_sha256=common,
            candidate=llm_static_bank,
            output=llm_static_bank,
            invocation=invocations["llm_static"],
        ),
        _receipt(
            config="s1",
            common_input_sha256=common,
            candidate=s1_candidate,
            output=s1_output,
            parent=llm_static_bank,
            invocation=invocations["s1"],
            gate=gates["s1"],
            decision="rolled_back",
        ),
        _receipt(
            config="s1s2",
            common_input_sha256=common,
            candidate=s2,
            output=s2,
            parent=s1_output,
            mutation=s2_plan,
            invocation=invocations["s1s2"],
            gate=gates["s1s2"],
        ),
        _receipt(
            config="full",
            common_input_sha256=common,
            candidate=full,
            output=full,
            parent=s2,
            mutation=s3_plan,
            invocation=invocations["full"],
            gate=gates["full"],
        ),
    )
    outputs = (llm_static_bank, s1_output, s2, full)
    candidates = (llm_static_bank, s1_candidate, s2, full)
    manifest = _manifest(records, outputs)
    verified = verify_portfolio_treatment_chain(
        manifest,
        output_banks=dict(zip(("llm_static", "s1", "s1s2", "full"), outputs)),
        candidate_banks=dict(zip(("llm_static", "s1", "s1s2", "full"), candidates)),
        mutations={"s1s2": s2_plan, "full": s3_plan},
        invocation_receipts=invocations,
        gate_reports=gates,
    )
    assert verified.output_banks["s1"] == llm_static_bank
    assert verified.candidate_banks["s1"] == s1_candidate
    assert verified.manifest.records[1].decision == "rolled_back"


def test_s3_rollback_derives_exact_s1s2_execution_artifact_alias(
    verified_rebind,
    llm_static_bank: StaticBankArtifact,
) -> None:
    s1 = _distinct_s1_bank(verified_rebind)
    s2_plan = _s2_mutation(s1)
    s2 = apply_portfolio_stage_mutation(s1, s2_plan)
    s3_plan = _s3_mutation(s2)
    rejected_full = apply_portfolio_stage_mutation(s2, s3_plan)
    common = verified_rebind.semantic_input.input_sha256
    invocations = {
        config: _invocation(config) for config in ("llm_static", "s1", "s1s2", "full")
    }
    gates = {
        "s1": _gate(config="s1", parent=llm_static_bank, candidate=s1),
        "s1s2": _gate(config="s1s2", parent=s1, candidate=s2),
        "full": _gate(
            config="full",
            parent=s2,
            candidate=rejected_full,
            decision="rolled_back",
        ),
    }
    records = (
        _receipt(
            config="llm_static",
            common_input_sha256=common,
            candidate=llm_static_bank,
            output=llm_static_bank,
            invocation=invocations["llm_static"],
        ),
        _receipt(
            config="s1",
            common_input_sha256=common,
            candidate=s1,
            output=s1,
            parent=llm_static_bank,
            invocation=invocations["s1"],
            gate=gates["s1"],
        ),
        _receipt(
            config="s1s2",
            common_input_sha256=common,
            candidate=s2,
            output=s2,
            parent=s1,
            mutation=s2_plan,
            invocation=invocations["s1s2"],
            gate=gates["s1s2"],
        ),
        _receipt(
            config="full",
            common_input_sha256=common,
            candidate=rejected_full,
            output=s2,
            parent=s2,
            mutation=s3_plan,
            invocation=invocations["full"],
            gate=gates["full"],
            decision="rolled_back",
        ),
    )
    manifest = _manifest(records, (llm_static_bank, s1, s2, s2))

    aliases = build_portfolio_execution_artifact_aliases(manifest)

    assert len(aliases) == 1
    alias = aliases[0]
    assert alias.policy_version == PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION
    assert alias.target_config == "full"
    assert alias.source_config == "s1s2"
    assert alias.stage_decision == "rolled_back"
    assert alias.source_bank_sha256 == alias.target_bank_sha256 == s2.bank_sha256
    assert alias.source_bank_file_sha256 == alias.target_bank_file_sha256
    assert alias.provider_model_call_count == 0
    assert alias.rejected_candidate_use == "diagnostic_only"

    tampered_full = records[-1].model_dump(mode="json")
    tampered_full["output_bank_file_sha256"] = "f" * 64
    unsigned_receipt = dict(tampered_full)
    unsigned_receipt.pop("receipt_sha256")
    tampered_full["receipt_sha256"] = sha256_bytes(
        canonical_json_bytes(unsigned_receipt)
    )
    tampered_records = (
        *records[:-1],
        PortfolioTreatmentReceipt.model_validate(
            tampered_full,
            strict=True,
        ),
    )
    with pytest.raises(
        ValidationError,
        match="Bank binding mismatch|exact parent artifact alias",
    ):
        _manifest(tampered_records, (llm_static_bank, s1, s2, s2))


def test_runtime_compatibility_rebind_preserves_chain_and_rejects_semantic_tamper(
    verified_rebind,
    llm_static_bank: StaticBankArtifact,
) -> None:
    s1 = _distinct_s1_bank(verified_rebind)
    s2_plan = _s2_mutation(s1)
    s2 = apply_portfolio_stage_mutation(s1, s2_plan)
    s3_plan = _s3_mutation(s2)
    rejected_full = apply_portfolio_stage_mutation(s2, s3_plan)
    common = verified_rebind.semantic_input.input_sha256
    invocations = {
        config: _invocation(config) for config in ("llm_static", "s1", "s1s2", "full")
    }
    gates = {
        "s1": _gate(config="s1", parent=llm_static_bank, candidate=s1),
        "s1s2": _gate(config="s1s2", parent=s1, candidate=s2),
        "full": _gate(
            config="full",
            parent=s2,
            candidate=rejected_full,
            decision="rolled_back",
        ),
    }
    records = (
        _receipt(
            config="llm_static",
            common_input_sha256=common,
            candidate=llm_static_bank,
            output=llm_static_bank,
            invocation=invocations["llm_static"],
        ),
        _receipt(
            config="s1",
            common_input_sha256=common,
            candidate=s1,
            output=s1,
            parent=llm_static_bank,
            invocation=invocations["s1"],
            gate=gates["s1"],
        ),
        _receipt(
            config="s1s2",
            common_input_sha256=common,
            candidate=s2,
            output=s2,
            parent=s1,
            mutation=s2_plan,
            invocation=invocations["s1s2"],
            gate=gates["s1s2"],
        ),
        _receipt(
            config="full",
            common_input_sha256=common,
            candidate=rejected_full,
            output=s2,
            parent=s2,
            mutation=s3_plan,
            invocation=invocations["full"],
            gate=gates["full"],
            decision="rolled_back",
        ),
    )
    source_outputs = {
        "llm_static": llm_static_bank,
        "s1": s1,
        "s1s2": s2,
        "full": s2,
    }
    source_candidates = {
        "llm_static": llm_static_bank,
        "s1": s1,
        "s1s2": s2,
        "full": rejected_full,
    }
    source = verify_portfolio_treatment_chain(
        _manifest(records, tuple(source_outputs.values())),
        output_banks=source_outputs,
        candidate_banks=source_candidates,
        mutations={"s1s2": s2_plan, "full": s3_plan},
        invocation_receipts=invocations,
        gate_reports=gates,
    )
    target_registry = build_mvp_registry_spec(include_multi_product=True)
    outputs = {
        config: rebind_portfolio_bank_runtime(bank, target_registry)
        for config, bank in source_outputs.items()
    }
    candidates = {
        config: rebind_portfolio_bank_runtime(bank, target_registry)
        for config, bank in source_candidates.items()
    }
    bindings = []
    for config in ("llm_static", "s1", "s1s2", "full"):
        for role, source_map, rebound_map in (
            ("output", source_outputs, outputs),
            ("candidate", source_candidates, candidates),
        ):
            source_bank = source_map[config]
            rebound = rebound_map[config]
            bindings.append(
                PortfolioRuntimeRebindBankBinding(
                    config=config,
                    role=role,
                    source_bank_sha256=source_bank.bank_sha256,
                    source_bank_file_sha256=sha256_bytes(source_bank.canonical_bytes()),
                    rebound_bank_file=f"{role}s/bank-{config}.json",
                    rebound_bank_sha256=rebound.bank_sha256,
                    rebound_bank_file_sha256=sha256_bytes(rebound.canonical_bytes()),
                )
            )
    receipt_payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-runtime-compatibility-rebind",
        "policy_version": PORTFOLIO_RUNTIME_COMPATIBILITY_REBIND_POLICY_VERSION,
        "source_runtime_dir": "parent-runtime",
        "source_runtime_lock_file_sha256": "1" * 64,
        "source_runtime_lock_sha256": "2" * 64,
        "source_treatment_manifest_file_sha256": "3" * 64,
        "source_treatment_chain_sha256": source.manifest.chain_sha256,
        "source_optimization_query_count": len(source.manifest.optimization_query_ids),
        "source_evaluation_query_count": len(source.manifest.evaluation_query_ids),
        "source_optimization_query_ids_sha256": sha256_bytes(
            canonical_json_bytes(list(source.manifest.optimization_query_ids))
        ),
        "source_evaluation_query_ids_sha256": sha256_bytes(
            canonical_json_bytes(list(source.manifest.evaluation_query_ids))
        ),
        "source_runtime_files": [
            PortfolioRuntimeFileBinding(
                relative_path="runtime-lock.json",
                file_sha256="1" * 64,
            ).model_dump(mode="json")
        ],
        "core_runtime_sources_dir": "core-runtime-sources",
        "core_runtime_sources_receipt_file_sha256": "4" * 64,
        "core_runtime_sources_receipt_sha256": "5" * 64,
        "target_tool_registry_sha256": target_registry.registry_sha256,
        "target_tool_registry_runtime_sha256": (
            target_registry.registry_runtime_sha256
        ),
        "target_runtime_data_sha256": "6" * 64,
        "target_source_sha256s": [str(index) * 64 for index in range(1, 7)],
        "semantic_fields_preserved": [
            "schema_version",
            "baseline_kind",
            "construction_identity_sha256",
            "construction_identity_policy",
            "runtime_binding_policy",
            "compiler",
            "skills",
            "capability_map",
        ],
        "compatibility_changes": [
            "tool_registry_sha256",
            "tool_registry_runtime_sha256",
            "bank_sha256",
        ],
        "algorithm_decisions_preserved": True,
        "model_evidence_bytes_preserved": True,
        "provider_model_call_count": 0,
        "bank_bindings": [item.model_dump(mode="json") for item in bindings],
    }
    receipt = PortfolioRuntimeCompatibilityRebind.model_validate(
        {
            **receipt_payload,
            "rebind_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
        },
        strict=True,
    )
    with pytest.raises(ValidationError):
        PortfolioRuntimeCompatibilityRebind.model_validate(
            {
                **receipt.model_dump(mode="json"),
                "provider_model_call_count": 1,
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="compatibility change set"):
        changed_allowlist = receipt.model_dump(mode="json")
        changed_allowlist["compatibility_changes"] = [
            *changed_allowlist["compatibility_changes"],
            "skills",
        ]
        unsigned_allowlist = dict(changed_allowlist)
        unsigned_allowlist.pop("rebind_sha256")
        changed_allowlist["rebind_sha256"] = sha256_bytes(
            canonical_json_bytes(unsigned_allowlist)
        )
        PortfolioRuntimeCompatibilityRebind.model_validate(
            changed_allowlist,
            strict=True,
        )

    verified = verify_portfolio_runtime_compatibility_rebind_chain(
        rebind=receipt,
        source_chain=source,
        output_banks=outputs,
        candidate_banks=candidates,
    )

    assert verified.manifest == source.manifest
    assert verified.compatibility_rebind == receipt
    assert verified.compatibility_rebind.provider_model_call_count == 0
    assert verified.output_banks["full"] == verified.output_banks["s1s2"]
    assert len(verified.execution_artifact_aliases) == 1
    assert verified.execution_artifact_aliases[0].target_config == "full"
    assert require_verified_portfolio_treatment_chain(verified) is verified

    tampered_payload = outputs["s1"].model_dump(mode="json", exclude={"bank_sha256"})
    tampered_payload["construction_identity_sha256"] = "f" * 64
    tampered = StaticBankArtifact.model_validate(
        {
            **tampered_payload,
            "bank_sha256": sha256_bytes(canonical_json_bytes(tampered_payload)),
        },
        strict=True,
    )
    tampered_outputs = {**outputs, "s1": tampered}
    with pytest.raises(
        PortfolioTreatmentError,
        match="binding differs|outside the compatibility allowlist",
    ):
        verify_portfolio_runtime_compatibility_rebind_chain(
            rebind=receipt,
            source_chain=source,
            output_banks=tampered_outputs,
            candidate_banks=candidates,
        )


def test_development_chain_is_parseable_but_not_matrix_verifiable(
    verified_rebind,
    llm_static_bank: StaticBankArtifact,
) -> None:
    s1 = _distinct_s1_bank(verified_rebind)
    s2_plan = _s2_mutation(s1)
    s2 = apply_portfolio_stage_mutation(s1, s2_plan)
    s3_plan = _s3_mutation(s2)
    full = apply_portfolio_stage_mutation(s2, s3_plan)
    common = verified_rebind.semantic_input.input_sha256
    invocations = {
        config: _invocation(config) for config in ("llm_static", "s1", "s1s2", "full")
    }
    gates = {
        "s1": _gate(config="s1", parent=llm_static_bank, candidate=s1),
        "s1s2": _gate(config="s1s2", parent=s1, candidate=s2),
        "full": _gate(config="full", parent=s2, candidate=full),
    }
    records = (
        _receipt(
            config="llm_static",
            common_input_sha256=common,
            candidate=llm_static_bank,
            output=llm_static_bank,
            invocation=invocations["llm_static"],
        ),
        _receipt(
            config="s1",
            common_input_sha256=common,
            candidate=s1,
            output=s1,
            parent=llm_static_bank,
            invocation=invocations["s1"],
            gate=gates["s1"],
        ),
        _receipt(
            config="s1s2",
            common_input_sha256=common,
            candidate=s2,
            output=s2,
            parent=s1,
            mutation=s2_plan,
            invocation=invocations["s1s2"],
            gate=gates["s1s2"],
        ),
        _receipt(
            config="full",
            common_input_sha256=common,
            candidate=full,
            output=full,
            parent=s2,
            mutation=s3_plan,
            invocation=invocations["full"],
            gate=gates["full"],
        ),
    )
    banks = (llm_static_bank, s1, s2, full)
    manifest = _manifest(records, banks, status="development")
    with pytest.raises(PortfolioTreatmentError, match="ready_for_matrix"):
        verify_portfolio_treatment_chain(
            manifest,
            output_banks=dict(zip(("llm_static", "s1", "s1s2", "full"), banks)),
            candidate_banks=dict(zip(("llm_static", "s1", "s1s2", "full"), banks)),
            mutations={"s1s2": s2_plan, "full": s3_plan},
            invocation_receipts=invocations,
            gate_reports=gates,
        )


def test_policy_constants_are_stable() -> None:
    assert (
        PORTFOLIO_GATE_DECISION_RULE
        == "causal-pareto-routing-quality-assistant-hard-error-skill-adherence-v3"
    )
    assert PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION == "portfolio-stage-gate-evidence-v4"
    assert (
        PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION == "portfolio-real-treatment-chain-v1"
    )
    assert PORTFOLIO_STAGE_MUTATION_POLICY_VERSION == "portfolio-stage-mutation-v1"


def test_gate_v4_emission_keeps_v3_artifacts_byte_loadable(
    llm_static_bank: StaticBankArtifact,
) -> None:
    v4_result = build_portfolio_stage_gate_result_set(
        source_config="s1",
        bank_sha256=llm_static_bank.bank_sha256,
        adherence_contract_bank_sha256=llm_static_bank.bank_sha256,
        evaluation_query_ids=GATE_QUERY_IDS,
        route_accuracy=0.6,
        mean_j=51.0,
        mean_skill_adherence=0.9,
        hard_error_count=0,
        rubric_file_sha256=RUBRIC_SHA,
        shared_route_artifact_sha256s=(None,) * len(GATE_QUERY_IDS),
        source_summary_file_sha256="1" * 64,
        source_summary_sha256="2" * 64,
        source_results_file_sha256="3" * 64,
        source_result_sha256s=tuple(
            sha256_bytes(f"legacy:{query_id}".encode()) for query_id in GATE_QUERY_IDS
        ),
    )
    legacy_result_payload = v4_result.model_dump(
        mode="json",
        exclude={"result_set_sha256"},
    )
    legacy_result_payload["schema_version"] = 3
    legacy_result_payload["gate_evidence_policy_version"] = (
        "portfolio-stage-gate-evidence-v3"
    )
    for field in (
        "evaluator_anomaly_count",
        "causal_reuse_unaffected",
        "row_provenance",
    ):
        legacy_result_payload.pop(field)
    legacy_result_raw = {
        **legacy_result_payload,
        "result_set_sha256": sha256_bytes(canonical_json_bytes(legacy_result_payload)),
    }
    legacy_result = PortfolioStageGateResultSet.model_validate(
        legacy_result_raw,
        strict=True,
    )
    assert legacy_result.canonical_bytes() == canonical_json_bytes(legacy_result_raw)

    v4_report = _gate(
        config="s1",
        parent=llm_static_bank,
        candidate=llm_static_bank,
    )
    legacy_report_payload = v4_report.model_dump(
        mode="json",
        exclude={"gate_report_sha256"},
    )
    legacy_report_payload.update(
        {
            "schema_version": 3,
            "gate_evidence_policy_version": "portfolio-stage-gate-evidence-v3",
            "decision_rule": "pareto-routing-quality-hard-error-skill-adherence-v2",
        }
    )
    for field in (
        "parent_evaluator_anomaly_count",
        "candidate_evaluator_anomaly_count",
        "causal_reuse_unaffected",
    ):
        legacy_report_payload.pop(field)
    legacy_report_raw = {
        **legacy_report_payload,
        "gate_report_sha256": sha256_bytes(canonical_json_bytes(legacy_report_payload)),
    }
    legacy_report = PortfolioStageGateReport.model_validate(
        legacy_report_raw,
        strict=True,
    )
    assert legacy_report.canonical_bytes() == canonical_json_bytes(legacy_report_raw)
