from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from skillchain.codex_authoring import validate_codex_cli_output_schema
from skillchain.evaluation.portfolio_s3_textopt import (
    PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION,
    PortfolioS3TextOptError,
    append_portfolio_s3_rejected_edit,
    build_empty_portfolio_s3_rejected_edit_buffer,
    build_portfolio_s3_editable_rule_catalog,
    build_portfolio_s3_failure_cluster_binding,
    compile_portfolio_s3_text_patch,
    normalize_portfolio_s3_text_patch,
    parse_portfolio_s3_rejected_edit_buffer,
    parse_portfolio_s3_text_patch_artifact,
    parse_portfolio_s3_text_patch_proposal,
    portfolio_s3_text_patch_proposal_json_schema,
)
from skillchain.evaluation.portfolio_treatments import (
    PortfolioArtifactBinding,
    PortfolioSkillMutation,
    apply_portfolio_stage_mutation,
    build_portfolio_stage_mutation,
)
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
PARENT_BANK = (
    ROOT
    / "runs"
    / "portfolio"
    / "portfolio-public-data-runtime-v15"
    / "bank-s1s2.json"
)
PARENT_BANK_FILE_SHA256 = "346963df7f0fc55fa935b86fd11c208bde38bc849fc30ba2852180f28838be7e"
IMPLEMENTATION_SHA = "b" * 64
ARTIFACT_SHA = "c" * 64
CAPABILITY = "knowledge.visual_encyclopedia"
SUCCESS_SCOPE = "encyclopedia.success.scope"
SUCCESS_CITATION = "encyclopedia.success.citation"
FAILURE_RULE = "encyclopedia.failure.ungrounded"
SAFETY_RULE = "encyclopedia.safety.no-identity-overclaim"


@pytest.fixture(scope="module")
def parent_bank() -> StaticBankArtifact:
    content = PARENT_BANK.read_bytes()
    assert sha256_bytes(content) == PARENT_BANK_FILE_SHA256
    return StaticBankArtifact.model_validate_json(
        content,
        strict=True,
    )


def _skill(bank: StaticBankArtifact):
    return next(item for item in bank.skills if item.capability_id == CAPABILITY)


def _line(bank: StaticBankArtifact, rule_id: str) -> str:
    prefix = f"- [{rule_id}] "
    return next(line for line in _skill(bank).body.splitlines() if line.startswith(prefix))


def _cluster(
    *,
    cluster_id: str = "cluster-grounding",
    classification: str = "SKILL_DEFECT",
    evidence_ids: tuple[str, ...] = ("evidence-1", "evidence-2"),
    support_count: int | None = None,
):
    return build_portfolio_s3_failure_cluster_binding(
        capability_id=CAPABILITY,
        failure_cluster_id=cluster_id,
        classification=classification,
        evidence_ids=evidence_ids,
        addressed_dimensions=("grounding",),
        support_count=support_count or len(evidence_ids),
        minimum_support_count=2,
    )


def _proposal(
    bank: StaticBankArtifact,
    cluster,
    *,
    op: str = "replace_rule",
    rule_id: str = SUCCESS_SCOPE,
    replacement: str | None = (
        "The answer must directly address the request and explicitly label every "
        "claim that remains unresolved."
    ),
    evidence_ids: tuple[str, ...] | None = None,
    support_count: int | None = None,
    reconsideration_rationale: str | None = None,
    edits: list[dict[str, object]] | None = None,
):
    evidence = evidence_ids or cluster.evidence_ids
    raw_edit = {
        "op": op,
        "rule_id": rule_id,
        "expected_text_sha256": sha256_bytes(_line(bank, rule_id).encode("utf-8")),
        "replacement": replacement,
        "evidence_ids": list(evidence),
        "support_count": support_count or len(evidence),
        "addressed_dimensions": ["grounding"],
        "rationale": "Recurring failures justify this narrow rule-level correction.",
    }
    return parse_portfolio_s3_text_patch_proposal(
        {
            "schema_version": 1,
            "artifact_kind": "portfolio-s3-text-patch-proposal",
            "policy_version": PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION,
            "capability_id": CAPABILITY,
            "failure_cluster_id": cluster.failure_cluster_id,
            "failure_cluster_sha256": cluster.cluster_sha256,
            "edits": edits or [raw_edit],
            "reconsideration_rationale": reconsideration_rationale,
        }
    )


def _bank_with_body(bank: StaticBankArtifact, body: str) -> StaticBankArtifact:
    source = _skill(bank)
    mutation = build_portfolio_stage_mutation(
        config="full",
        parent_bank_sha256=bank.bank_sha256,
        implementation_id="s3-textopt-test",
        implementation_version="1.0.0",
        implementation_file_sha256=IMPLEMENTATION_SHA,
        input_artifacts=(
            PortfolioArtifactBinding(
                artifact_kind="body_attribution",
                artifact_file="inputs/body-attribution.json",
                artifact_file_sha256=ARTIFACT_SHA,
                artifact_content_sha256=ARTIFACT_SHA,
            ),
        ),
        changes=(
            PortfolioSkillMutation(
                capability_id=CAPABILITY,
                parent_skill_sha256=source.skill_sha256,
                body=body,
            ),
        ),
    )
    return apply_portfolio_stage_mutation(bank, mutation)


def _output_contract(body: str) -> str:
    return body.split("## Output contract", 1)[1].split("## Failure conditions", 1)[0]


def test_replace_compiles_to_existing_body_mutation_and_preserves_contract(
    parent_bank: StaticBankArtifact,
) -> None:
    cluster = _cluster()
    proposal = _proposal(parent_bank, cluster)

    patch = normalize_portfolio_s3_text_patch(
        parent_bank=parent_bank, failure_cluster=cluster, proposal=proposal
    )
    mutation = compile_portfolio_s3_text_patch(
        parent_bank=parent_bank,
        failure_cluster=cluster,
        proposal_or_patch=patch,
    )

    source = _skill(parent_bank)
    assert mutation.capability_id == CAPABILITY
    assert mutation.parent_skill_sha256 == source.skill_sha256
    assert mutation.description is None
    assert proposal.edits[0].replacement in mutation.body
    assert _output_contract(mutation.body) == _output_contract(source.body)
    assert patch.canonical_bytes() == normalize_portfolio_s3_text_patch(
        parent_bank=parent_bank, failure_cluster=cluster, proposal=proposal
    ).canonical_bytes()


def test_insert_and_delete_only_compiler_inserted_rules(
    parent_bank: StaticBankArtifact,
) -> None:
    insert_cluster = _cluster(cluster_id="cluster-insert")
    insert_proposal = _proposal(
        parent_bank,
        insert_cluster,
        op="insert_after_rule",
        replacement="State the most important unresolved evidence before concluding.",
    )
    insert_patch = normalize_portfolio_s3_text_patch(
        parent_bank=parent_bank,
        failure_cluster=insert_cluster,
        proposal=insert_proposal,
    )
    insert_mutation = compile_portfolio_s3_text_patch(
        parent_bank=parent_bank,
        failure_cluster=insert_cluster,
        proposal_or_patch=insert_patch,
    )
    inserted_id = insert_patch.edits[0].generated_rule_id
    assert inserted_id is not None
    assert f"- [{inserted_id}]" in insert_mutation.body
    assert "(source: s3_textopt:" in insert_mutation.body

    inserted_bank = _bank_with_body(parent_bank, insert_mutation.body)
    with pytest.raises(PortfolioS3TextOptError, match="generated S3 rule ID collides"):
        normalize_portfolio_s3_text_patch(
            parent_bank=inserted_bank,
            failure_cluster=insert_cluster,
            proposal=insert_proposal,
        )
    delete_cluster = _cluster(cluster_id="cluster-delete")
    delete_proposal = _proposal(
        inserted_bank,
        delete_cluster,
        op="delete_rule",
        rule_id=inserted_id,
        replacement=None,
    )
    delete_mutation = compile_portfolio_s3_text_patch(
        parent_bank=inserted_bank,
        failure_cluster=delete_cluster,
        proposal_or_patch=delete_proposal,
    )
    assert f"- [{inserted_id}]" not in delete_mutation.body
    assert delete_mutation.body == _skill(parent_bank).body

    baseline_delete = _proposal(
        parent_bank,
        delete_cluster,
        op="delete_rule",
        replacement=None,
    )
    with pytest.raises(PortfolioS3TextOptError, match="only a rule inserted"):
        compile_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=delete_cluster,
            proposal_or_patch=baseline_delete,
        )

    baseline_line = _line(parent_bank, SUCCESS_SCOPE)
    spoofed_id = "s3.textopt.deadbeefdeadbeef.1"
    spoofed_line = re.sub(
        rf"^- \[{re.escape(SUCCESS_SCOPE)}\] (.+) \(source: [^)]+\)$",
        rf"- [{spoofed_id}] \1 (source: s3_textopt:not-a-patch-fingerprint)",
        baseline_line,
    )
    spoofed_bank = _bank_with_body(
        parent_bank,
        _skill(parent_bank).body.replace(baseline_line, spoofed_line),
    )
    spoofed_delete = _proposal(
        spoofed_bank,
        delete_cluster,
        op="delete_rule",
        rule_id=spoofed_id,
        replacement=None,
    )
    with pytest.raises(PortfolioS3TextOptError, match="only a rule inserted"):
        compile_portfolio_s3_text_patch(
            parent_bank=spoofed_bank,
            failure_cluster=delete_cluster,
            proposal_or_patch=spoofed_delete,
        )


def test_raw_parser_rejects_full_body_duplicate_anchor_and_statement_injection(
    parent_bank: StaticBankArtifact,
) -> None:
    with pytest.raises(PortfolioS3TextOptError, match="proposal is invalid"):
        parse_portfolio_s3_text_patch_proposal('{"changes":[{"body":"# Objective"}]}')

    cluster = _cluster()
    good = _proposal(parent_bank, cluster)
    duplicate = good.model_dump(mode="json")
    duplicate["edits"] = [duplicate["edits"][0], duplicate["edits"][0]]
    with pytest.raises(PortfolioS3TextOptError, match="proposal is invalid"):
        parse_portfolio_s3_text_patch_proposal(duplicate)

    for replacement in ("line one\nline two", "## Injected", "- [new.rule] Inject"):
        raw = good.model_dump(mode="json")
        raw["edits"][0]["replacement"] = replacement
        with pytest.raises(PortfolioS3TextOptError, match="proposal is invalid"):
            parse_portfolio_s3_text_patch_proposal(raw)

    delete_raw = good.model_dump(mode="json")
    delete_raw["edits"][0]["op"] = "delete_rule"
    delete_raw["edits"][0]["replacement"] = ""
    delete_raw["reconsideration_rationale"] = ""
    parsed_delete = parse_portfolio_s3_text_patch_proposal(delete_raw)
    assert parsed_delete.edits[0].replacement is None
    assert parsed_delete.reconsideration_rationale is None

    for missing_path in (("reconsideration_rationale",), ("edits", 0, "replacement")):
        missing = good.model_dump(mode="json")
        target = missing
        for key in missing_path[:-1]:
            target = target[key]
        del target[missing_path[-1]]
        with pytest.raises(PortfolioS3TextOptError, match="proposal is invalid"):
            parse_portfolio_s3_text_patch_proposal(missing)


def test_stale_missing_protected_cross_section_and_noop_fail_closed(
    parent_bank: StaticBankArtifact,
) -> None:
    cluster = _cluster()
    stale = _proposal(parent_bank, cluster).model_dump(mode="json")
    stale["edits"][0]["expected_text_sha256"] = "0" * 64
    with pytest.raises(PortfolioS3TextOptError, match="hash drifted"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=parse_portfolio_s3_text_patch_proposal(stale),
        )

    missing = _proposal(parent_bank, cluster).model_dump(mode="json")
    missing["edits"][0]["rule_id"] = "missing.rule"
    with pytest.raises(PortfolioS3TextOptError, match="target rule is missing"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=parse_portfolio_s3_text_patch_proposal(missing),
        )

    protected = _proposal(parent_bank, cluster, rule_id=SAFETY_RULE)
    with pytest.raises(PortfolioS3TextOptError, match="protected section"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=protected,
        )

    first = _proposal(parent_bank, cluster).model_dump(mode="json")["edits"][0]
    second = _proposal(parent_bank, cluster, rule_id=FAILURE_RULE).model_dump(
        mode="json"
    )["edits"][0]
    cross_section = _proposal(parent_bank, cluster, edits=[first, second])
    with pytest.raises(PortfolioS3TextOptError, match="same editable section"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=cross_section,
            optimization_mode="confirmed_l3",
            origin_bank=parent_bank,
        )

    source_statement = _line(parent_bank, SUCCESS_SCOPE).split("] ", 1)[1].rsplit(
        " (source:", 1
    )[0]
    noop = _proposal(parent_bank, cluster, replacement=source_statement)
    with pytest.raises(PortfolioS3TextOptError, match="must change"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=noop,
        )


@pytest.mark.parametrize(
    "body_transform,error",
    [
        (lambda body: body.replace("## Output contract\n\n", ""), "output contract"),
        (
            lambda body: body.replace(
                "- evidence_requirement: cited_knowledge_evidence",
                "- unexpected_field: x\n\n- evidence_requirement: cited_knowledge_evidence",
            ),
            "frozen five fields",
        ),
        (
            lambda body: body.replace(
                "- card_requirement: forbidden", "- card_requirement: required"
            ),
            "semantics are invalid",
        ),
        (lambda body: body.replace("\n", "\r\n"), "must use LF"),
        (
            lambda body: body.replace(
                "- [encyclopedia.success.scope]",
                "- [encyclopedia.success.citation]",
            ),
            "globally unique",
        ),
    ],
)
def test_invalid_parent_body_contract_and_rule_structure_is_rejected(
    parent_bank: StaticBankArtifact,
    body_transform,
    error: str,
) -> None:
    malformed_bank = _bank_with_body(parent_bank, body_transform(_skill(parent_bank).body))
    cluster = _cluster()
    proposal = _proposal(parent_bank, cluster)
    with pytest.raises(PortfolioS3TextOptError, match=error):
        normalize_portfolio_s3_text_patch(
            parent_bank=malformed_bank,
            failure_cluster=cluster,
            proposal=proposal,
        )


def test_modes_enforce_edit_count_origin_and_growth_budgets(
    parent_bank: StaticBankArtifact,
) -> None:
    cluster = _cluster()
    first = _proposal(parent_bank, cluster).model_dump(mode="json")["edits"][0]
    second = _proposal(parent_bank, cluster, rule_id=SUCCESS_CITATION).model_dump(
        mode="json"
    )["edits"][0]
    two_edits = _proposal(parent_bank, cluster, edits=[first, second])
    with pytest.raises(PortfolioS3TextOptError, match="exactly one edit"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=two_edits,
        )
    with pytest.raises(PortfolioS3TextOptError, match="origin Bank"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=two_edits,
            optimization_mode="confirmed_l3",
        )
    confirmed = normalize_portfolio_s3_text_patch(
        parent_bank=parent_bank,
        failure_cluster=cluster,
        proposal=two_edits,
        optimization_mode="confirmed_l3",
        origin_bank=parent_bank,
    )
    assert len(confirmed.edits) == 2

    protected_drift_bank = _bank_with_body(
        parent_bank,
        _skill(parent_bank).body.replace(
            "# Objective\n\n",
            "# Objective\n\nProtected origin text was changed. ",
            1,
        ),
    )
    drifted_proposal = _proposal(protected_drift_bank, cluster)
    with pytest.raises(PortfolioS3TextOptError, match="frozen origin"):
        normalize_portfolio_s3_text_patch(
            parent_bank=protected_drift_bank,
            failure_cluster=cluster,
            proposal=drifted_proposal,
            optimization_mode="confirmed_l3",
            origin_bank=parent_bank,
        )

    huge = _proposal(parent_bank, cluster, replacement="x" * 1000)
    with pytest.raises(PortfolioS3TextOptError, match="exceeds 10 percent"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=huge,
        )
    with pytest.raises(PortfolioS3TextOptError, match="exceeds 20 percent"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=cluster,
            proposal=huge,
            optimization_mode="confirmed_l3",
            origin_bank=parent_bank,
        )


def test_execution_lapse_under_support_and_cluster_drift_never_mutate(
    parent_bank: StaticBankArtifact,
) -> None:
    lapse = _cluster(classification="EXECUTION_LAPSE")
    with pytest.raises(PortfolioS3TextOptError, match="execution lapse"):
        compile_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=lapse,
            proposal_or_patch=_proposal(parent_bank, lapse),
        )

    under_supported = build_portfolio_s3_failure_cluster_binding(
        capability_id=CAPABILITY,
        failure_cluster_id="under-supported",
        classification="SKILL_DEFECT",
        evidence_ids=("evidence-1",),
        addressed_dimensions=("grounding",),
        support_count=1,
        minimum_support_count=2,
    )
    with pytest.raises(PortfolioS3TextOptError, match="under-supported"):
        compile_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=under_supported,
            proposal_or_patch=_proposal(
                parent_bank,
                under_supported,
                evidence_ids=("evidence-1",),
                support_count=1,
            ),
        )

    cluster = _cluster()
    other = _cluster(cluster_id="other-cluster")
    with pytest.raises(PortfolioS3TextOptError, match="cluster identity"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=other,
            proposal=_proposal(parent_bank, cluster),
        )


def test_rejected_fingerprint_ignores_evidence_cluster_and_rationale_but_gates_retry(
    parent_bank: StaticBankArtifact,
) -> None:
    first_cluster = _cluster()
    first_proposal = _proposal(parent_bank, first_cluster)
    first_patch = normalize_portfolio_s3_text_patch(
        parent_bank=parent_bank,
        failure_cluster=first_cluster,
        proposal=first_proposal,
    )
    buffer = append_portfolio_s3_rejected_edit(
        build_empty_portfolio_s3_rejected_edit_buffer(),
        patch=first_patch,
        rejection_reason="Body gate regressed grounding.",
    )
    assert parse_portfolio_s3_rejected_edit_buffer(buffer.canonical_bytes()) == buffer

    with pytest.raises(PortfolioS3TextOptError, match="strictly more evidence"):
        normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=first_cluster,
            proposal=first_proposal,
            rejected_buffer=buffer,
        )

    expanded_cluster = _cluster(
        cluster_id="different-cluster",
        evidence_ids=("evidence-1", "evidence-2", "evidence-3"),
    )
    expanded = _proposal(
        parent_bank,
        expanded_cluster,
        evidence_ids=expanded_cluster.evidence_ids,
        reconsideration_rationale=(
            "A newly observed third failure supports reconsidering the same edit."
        ),
    )
    retried = normalize_portfolio_s3_text_patch(
        parent_bank=parent_bank,
        failure_cluster=expanded_cluster,
        proposal=expanded,
        rejected_buffer=buffer,
    )
    assert retried.patch_fingerprint == first_patch.patch_fingerprint
    assert retried.patch_sha256 != first_patch.patch_sha256


def test_normalized_artifact_is_canonical_self_hashed_and_schema_is_strict(
    parent_bank: StaticBankArtifact,
) -> None:
    cluster = _cluster()
    artifact = normalize_portfolio_s3_text_patch(
        parent_bank=parent_bank,
        failure_cluster=cluster,
        proposal=_proposal(parent_bank, cluster),
    )
    assert parse_portfolio_s3_text_patch_artifact(artifact.canonical_bytes()) == artifact
    catalog = build_portfolio_s3_editable_rule_catalog(
        parent_bank=parent_bank,
        capability_ids=(CAPABILITY,),
    )
    catalog_rules = catalog[0]["rules"]
    assert isinstance(catalog_rules, list)
    catalog_rule = next(
        item for item in catalog_rules if item["rule_id"] == SUCCESS_SCOPE
    )
    assert catalog_rule["full_line"] == _line(parent_bank, SUCCESS_SCOPE)
    assert catalog_rule["expected_text_sha256"] == sha256_bytes(
        _line(parent_bank, SUCCESS_SCOPE).encode("utf-8")
    )
    assert all(item["rule_id"] != SAFETY_RULE for item in catalog_rules)
    with pytest.raises(PortfolioS3TextOptError, match="not canonical"):
        parse_portfolio_s3_text_patch_artifact(
            artifact.model_dump_json(indent=2).encode("utf-8")
        )
    incomplete_patch = artifact.model_dump(mode="json")
    incomplete_patch.pop("reconsideration_rationale")
    with pytest.raises(PortfolioS3TextOptError, match="canonical model bytes"):
        parse_portfolio_s3_text_patch_artifact(
            canonical_json_bytes(incomplete_patch)
        )

    empty_buffer = build_empty_portfolio_s3_rejected_edit_buffer()
    incomplete_buffer = empty_buffer.model_dump(mode="json")
    incomplete_buffer.pop("schema_version")
    with pytest.raises(PortfolioS3TextOptError, match="canonical model bytes"):
        parse_portfolio_s3_rejected_edit_buffer(
            canonical_json_bytes(incomplete_buffer)
        )

    schema = portfolio_s3_text_patch_proposal_json_schema()
    validate_codex_cli_output_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "schema_version",
        "artifact_kind",
        "policy_version",
        "capability_id",
        "failure_cluster_id",
        "failure_cluster_sha256",
        "edits",
        "reconsideration_rationale",
    }
    l1_schema = portfolio_s3_text_patch_proposal_json_schema(max_edits=1)
    validate_codex_cli_output_schema(l1_schema)
    assert l1_schema["properties"]["edits"]["maxItems"] == 1
    with pytest.raises(PortfolioS3TextOptError, match="must be 1 or 3"):
        portfolio_s3_text_patch_proposal_json_schema(max_edits=2)  # type: ignore[arg-type]


def test_self_hashes_and_sorted_evidence_fail_closed() -> None:
    cluster = _cluster()
    with pytest.raises(ValidationError, match="cluster_sha256 mismatch"):
        cluster.model_copy(update={"cluster_sha256": "0" * 64}).__class__.model_validate(
            cluster.model_copy(update={"cluster_sha256": "0" * 64}).model_dump(
                mode="python"
            ),
            strict=True,
        )
    with pytest.raises(PortfolioS3TextOptError, match="binding is invalid"):
        build_portfolio_s3_failure_cluster_binding(
            capability_id=CAPABILITY,
            failure_cluster_id="unsorted",
            classification="SKILL_DEFECT",
            evidence_ids=("z", "a"),
            addressed_dimensions=("grounding",),
            support_count=2,
            minimum_support_count=2,
        )
