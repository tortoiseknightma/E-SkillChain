from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from scripts import analyze_portfolio_s1_population as analyzer_cli
from scripts import run_portfolio_s1_gcs_gate as gate_cli
from skillchain.evaluation import portfolio_s1_gcs_artifacts as artifact_exporter
from skillchain.evaluation.portfolio_gcs import (
    GCS_CAPABILITY_ORDER,
    GCSQueryScoreV2,
    build_gcs_population_v2,
)
from skillchain.evaluation.portfolio_s1_experiment_runtime import (
    HISTORICAL_S1_RUNTIME_V5_FILE_SHA256,
    load_verified_portfolio_s1_experiment_launch,
    load_verified_portfolio_s1_experiment_runtime_evidence,
)
from skillchain.evaluation.portfolio_treatments import (
    compile_portfolio_s1_creator_bank,
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
)
from skillchain.evaluation.portfolio_s1_gcs_artifacts import (
    PortfolioS1GCSExportManifestV1,
    S1_GCS_ARTIFACT_POLICY_VERSION,
)
from skillchain.evolution.s1_gcs_gate import (
    S1GCSGateError,
    S1_GCS_GATE_POLICY_SHA256,
    S1_GCS_GATE_POLICY_VERSION,
    S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS,
    S1Round2DevelopmentScreen,
    S1Round2ResponseContractDiagnostic,
    build_s1_bank_disposition,
    build_s1_round2_bank_disposition,
    evaluate_s1_development_composite,
    evaluate_s1_body_gate,
    evaluate_s1_replay,
    evaluate_s1_round2_body_gate,
    make_s1_gcs_evidence_binding,
    make_s1_round2_candidate_freeze,
    make_s1_round2_response_contract_diagnostics,
    screen_s1_development_patches,
    screen_s1_round2_development_patches,
)
from skillchain.evolution.s1_sparse_patch import (
    S1_SPARSE_COMPILATION_POLICY_VERSION,
    SparseCompilationReceiptV1,
    SparseScreenedBankReceiptV1,
)
from skillchain.schemas import LabelDecision, Query
from skillchain.static_authoring import (
    StaticBankArtifact,
    build_authoring_draft_bundle,
    build_capability_draft,
)
from skillchain.synthesis.store import canonical_jsonl_bytes
from skillchain.taxonomy import TAXONOMY_VERSION
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
REAL_S1_REPLAY_ROOT_V5 = (
    ROOT.parent.parent
    / "b5cd"
    / "ECommerceSkillChain"
    / "runs"
    / "portfolio"
    / "core-s1"
)
_REAL_S1_REPLAY_V5_AVAILABLE = all(
    path.is_dir()
    for path in (
        REAL_S1_REPLAY_ROOT_V5 / "s1-runtime-v5",
        REAL_S1_REPLAY_ROOT_V5 / "s1-replay-launch-v5",
        REAL_S1_REPLAY_ROOT_V5 / "s1-replay-execution-v5",
    )
)
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
_CAPABILITY_META = {
    "knowledge.visual_encyclopedia": ("encyclopedia", False),
    "product.exact_match": ("exact_match", True),
    "product.multi_search": ("multi_product", True),
    "product.style_recommendation": ("divergent_rec", True),
    "utility.document_reading": ("utility", False),
    "utility.recipe_guidance": ("utility", False),
}


@pytest.fixture(scope="module")
def banks() -> tuple[StaticBankArtifact, StaticBankArtifact]:
    def file_sha(path: Path) -> str:
        return sha256_bytes(path.read_bytes())

    rebind = load_verified_codex_draft_rebind(
        codex_input_path=CODEX_INPUT,
        expected_codex_input_file_sha256=file_sha(CODEX_INPUT),
        semantic_input_path=SEMANTIC_INPUT,
        expected_semantic_input_file_sha256=file_sha(SEMANTIC_INPUT),
        draft_path=CODEX_DRAFT,
        expected_draft_file_sha256=file_sha(CODEX_DRAFT),
    )
    parent = compile_verified_codex_llm_static_bank(
        rebind, tool_registry_runtime_sha256=RUNTIME_SHA
    )
    source = rebind.rebound_draft.drafts[0]
    changed = build_capability_draft(
        capability_id=source.capability_id,
        objective=source.objective + " Verify evidence before answering.",
        steps=source.steps,
        fallback_instruction=source.fallback_instruction,
        fallback_may_request_clarification=source.fallback_may_request_clarification,
        fallback_must_state_uncertainty=source.fallback_must_state_uncertainty,
        citation_source_ids=source.citation_source_ids,
        rule_coverage=source.rule_coverage,
        output_coverage=source.output_coverage,
    )
    bundle = build_authoring_draft_bundle(
        authoring_input_sha256=rebind.semantic_input.input_sha256,
        drafts=(changed, *rebind.rebound_draft.drafts[1:]),
    )
    candidate = compile_portfolio_s1_creator_bank(
        rebind.semantic_input,
        bundle,
        tool_registry_runtime_sha256=RUNTIME_SHA,
    )
    assert candidate.bank_sha256 != parent.bank_sha256
    return parent, candidate


def _queries(
    *,
    count: int,
    split: Literal["opt_pool", "val", "test_frozen"],
    prefix: str,
    required_query_ids: tuple[str, ...] = (),
) -> tuple[Query, ...]:
    rows: list[Query] = []
    for index in range(count):
        capability = GCS_CAPABILITY_ORDER[index % len(GCS_CAPABILITY_ORDER)]
        intent, requires_card = _CAPABILITY_META[capability]
        query_id = (
            required_query_ids[index]
            if index < len(required_query_ids)
            else f"{prefix}-{index:03d}"
        )
        text = f"Evaluate {query_id}"
        rows.append(
            Query(
                schema_version=2,
                taxonomy_version=TAXONOMY_VERSION,
                task_spec_version="ecommerce-task-spec-v1",
                query_id=query_id,
                asset_id=f"asset.{query_id}",
                image_path=f"fixtures/{query_id}.jpg",
                leakage_group_id=f"leakage-{query_id}",
                boundary_group_id=None,
                template_family=f"template-{query_id}",
                generator_batch_id=f"batch-{index // 25:02d}",
                text=text,
                turns=[{"role": "user", "content": text}],
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
                is_boundary=False,
                boundary_strategy=None,
                requires_card=requires_card,
                split=split,
                label_status="auto",
                label_provenance=[
                    LabelDecision(
                        decision_type="constructed",
                        annotator_kind="planner",
                        annotator_id="s1-gcs-gate-test",
                        canonical_intent=intent,
                        canonical_capability=capability,
                        acceptable_capabilities=[capability],
                    )
                ],
            )
        )
    return tuple(rows)


def _sparse_receipt(
    parent: StaticBankArtifact, candidate: StaticBankArtifact
) -> SparseCompilationReceiptV1:
    parent_by_capability = {item.capability_id: item for item in parent.skills}
    candidate_by_capability = {item.capability_id: item for item in candidate.skills}
    bindings = []
    for capability_id in sorted(parent_by_capability):
        parent_skill = parent_by_capability[capability_id]
        candidate_skill = candidate_by_capability[capability_id]
        parent_bytes = canonical_json_bytes(parent_skill.model_dump(mode="json"))
        candidate_bytes = canonical_json_bytes(candidate_skill.model_dump(mode="json"))
        action = (
            "inherit"
            if candidate_skill.skill_sha256 == parent_skill.skill_sha256
            else "patch"
        )
        bindings.append(
            {
                "capability_id": capability_id,
                "action": action,
                "parent_skill_sha256": parent_skill.skill_sha256,
                "parent_skill_bytes_sha256": sha256_bytes(parent_bytes),
                "candidate_skill_sha256": candidate_skill.skill_sha256,
                "candidate_skill_bytes_sha256": sha256_bytes(candidate_bytes),
                "inherited_bytes_exact": action == "inherit",
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-sparse-compilation-receipt",
        "policy_version": S1_SPARSE_COMPILATION_POLICY_VERSION,
        "sparse_draft_sha256": "1" * 64,
        "sparse_draft_file_sha256": "2" * 64,
        "parent_bank_sha256": parent.bank_sha256,
        "feedback_bundle_sha256": "3" * 64,
        "authoring_input_sha256": "4" * 64,
        "tool_registry_sha256": parent.tool_registry_sha256,
        "tool_registry_runtime_sha256": parent.tool_registry_runtime_sha256,
        "compiler_identity_sha256": "5" * 64,
        "sparse_compiler_file_sha256": "6" * 64,
        "compiler_owned_sections_sha256": "7" * 64,
        "bindings": bindings,
        "candidate_bank_sha256": candidate.bank_sha256,
    }
    return SparseCompilationReceiptV1.model_validate(
        {
            **payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


def _scores(
    queries: tuple[Query, ...],
    *,
    config: Literal["llm_static", "s1"],
    success: bool,
    coverage: bool = True,
) -> tuple[GCSQueryScoreV2, ...]:
    population = build_gcs_population_v2(queries)
    component_by_id = {item.query_id: item.component_id for item in population.bindings}
    return tuple(
        GCSQueryScoreV2(
            query_id=query.query_id,
            config=config,
            canonical_capability=query.canonical_capability,
            component_id=component_by_id[query.query_id],
            route_disposition="pass",
            answer_mode="supported",
            oracle_available=coverage,
            semantic_claim_support_resolved=True,
            route_acceptable=1,
            no_hard_error=1,
            tool_contract_pass=1,
            evidence_grounded=1 if success else 0,
            output_contract_pass=1,
            hard_error=0,
            gcs=1 if success else 0,
            reason_codes=() if success else ("material_claim_uncited",),
            evaluated_capability=query.canonical_capability,
            style_support_status=(
                "candidates"
                if query.canonical_capability == "product.style_recommendation"
                else None
            ),
        )
        for query in queries
    )


def _failed_score(
    score: GCSQueryScoreV2,
    *,
    reason_code: str = "material_claim_uncited",
    hard_error: bool = False,
) -> GCSQueryScoreV2:
    payload = score.model_dump(mode="python")
    payload.update(
        {
            "no_hard_error": 0 if hard_error else 1,
            "hard_error": 1 if hard_error else 0,
            "evidence_grounded": 1 if hard_error else 0,
            "output_contract_pass": (
                0 if reason_code == "output_section_invalid" else 1
            ),
            "gcs": 0,
            "reason_codes": (reason_code,),
        }
    )
    return GCSQueryScoreV2.model_validate(payload, strict=True)


def _evidence(phase: Literal["replay", "body_gate"], banks):
    parent, candidate = banks
    return make_s1_gcs_evidence_binding(
        phase=phase,
        population_binding_file_sha256="1" * 64,
        queries_file_sha256="2" * 64,
        baseline_scores_file_sha256="3" * 64,
        candidate_scores_file_sha256="4" * 64,
        parent_bank_file_sha256=sha256_bytes(parent.canonical_bytes()),
        candidate_bank_file_sha256=sha256_bytes(candidate.canonical_bytes()),
    )


def _contract_diagnostics(
    queries: tuple[Query, ...],
    *,
    mapped_by_key: dict[tuple[str, str], tuple[str, ...]] | None = None,
):
    mapped_by_key = mapped_by_key or {}
    rows = []
    for query in queries:
        for config in ("llm_static", "s1"):
            mapped = mapped_by_key.get((query.query_id, config), ())
            terminal = bool(mapped)
            payload = {
                "schema_version": 1,
                "artifact_kind": (
                    "portfolio-s1-round2-response-contract-diagnostic"
                ),
                "policy_version": (
                    "portfolio-s1-round2-response-contract-diagnostics-v1"
                ),
                "query_id": query.query_id,
                "config": config,
                "checkpoint_file_sha256": sha256_bytes(
                    f"{query.query_id}:{config}:file".encode()
                ),
                "checkpoint_row_sha256": sha256_bytes(
                    f"{query.query_id}:{config}:row".encode()
                ),
                "assistant_response_sha256": sha256_bytes(
                    f"{query.query_id}:{config}:response".encode()
                ),
                "assistant_receipt_sha256": sha256_bytes(
                    f"{query.query_id}:{config}:receipt".encode()
                ),
                "response_error_code": (
                    "response_contract_error" if terminal else None
                ),
                "repair_status": "not_attempted",
                "repair_attempt_count": 0,
                "repair_receipt_sha256": None,
                "repair_call_index": None,
                "repair_input_tokens": 0,
                "repair_output_tokens": 0,
                "initial_reason_codes": [],
                "final_reason_codes": [],
                "mapped_contract_reason_codes": list(mapped),
                "mapping_disposition": (
                    "unclassified_terminal_mapped_fail_closed"
                    if terminal
                    else "not_applicable"
                ),
            }
            rows.append(
                S1Round2ResponseContractDiagnostic.model_validate(
                    {
                        **payload,
                        "diagnostic_sha256": sha256_bytes(
                            canonical_json_bytes(payload)
                        ),
                    },
                    strict=True,
                )
            )
    return make_s1_round2_response_contract_diagnostics(
        query_ids=tuple(query.query_id for query in queries), rows=tuple(rows)
    )


def _write_verified_gate_export(
    root: Path,
    *,
    queries: tuple[Query, ...],
    baseline: tuple[GCSQueryScoreV2, ...],
    candidate_scores: tuple[GCSQueryScoreV2, ...],
    parent: StaticBankArtifact,
    candidate_bank: StaticBankArtifact,
    phase: Literal["replay", "body_gate"] = "replay",
) -> str:
    execution_scope = "s1_opt_replay" if phase == "replay" else "s1_body_gate"
    source_split = "opt_pool" if phase == "replay" else "val"
    population = build_gcs_population_v2(queries)
    contract_diagnostics = _contract_diagnostics(queries)
    source_hashes = {
        "exporter_file_sha256": sha256_bytes(
            Path(artifact_exporter.__file__).read_bytes()
        ),
        "analyzer_cli_file_sha256": sha256_bytes(
            (ROOT / "scripts" / "analyze_portfolio_s1_population.py").read_bytes()
        ),
        "gate_cli_file_sha256": sha256_bytes(Path(gate_cli.__file__).read_bytes()),
        "gate_module_file_sha256": sha256_bytes(
            (ROOT / "src" / "skillchain" / "evolution" / "s1_gcs_gate.py").read_bytes()
        ),
        "gate_policy_sha256": S1_GCS_GATE_POLICY_SHA256,
    }
    identity = {
        "execution_control_file_sha256": "5" * 64,
        "execution_control_sha256": "6" * 64,
        "launch_plan_file_sha256": "7" * 64,
        "launch_plan_sha256": "8" * 64,
        "runtime_lock_file_sha256": "9" * 64,
        "runtime_lock_sha256": "a" * 64,
    }
    binding_payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-gcs-population-binding",
        "policy_version": S1_GCS_ARTIFACT_POLICY_VERSION,
        "phase": phase,
        "execution_scope": execution_scope,
        "source_split": source_split,
        "query_count": len(queries),
        "query_ids": [query.query_id for query in queries],
        "population": population.model_dump(mode="json"),
        **identity,
        "s1_gcs_artifacts_file_sha256": source_hashes["exporter_file_sha256"],
        "analyzer_cli_file_sha256": source_hashes["analyzer_cli_file_sha256"],
        "gate_cli_file_sha256": source_hashes["gate_cli_file_sha256"],
        "gate_module_file_sha256": source_hashes["gate_module_file_sha256"],
        "gate_policy_sha256": source_hashes["gate_policy_sha256"],
        "parent_bank_sha256": parent.bank_sha256,
        "candidate_bank_sha256": candidate_bank.bank_sha256,
        "round2_response_contract_diagnostics": (
            contract_diagnostics.model_dump(mode="json")
        ),
    }
    population_binding = canonical_json_bytes(
        {
            **binding_payload,
            "binding_sha256": sha256_bytes(canonical_json_bytes(binding_payload)),
        }
    )
    file_bytes = {
        "queries.jsonl": canonical_jsonl_bytes(queries),
        "baseline-scores.jsonl": canonical_jsonl_bytes(baseline),
        "candidate-scores.jsonl": canonical_jsonl_bytes(candidate_scores),
        "population-binding.json": population_binding,
        "parent-bank.json": parent.canonical_bytes(),
        "candidate-bank.json": candidate_bank.canonical_bytes(),
    }
    evidence = make_s1_gcs_evidence_binding(
        phase=phase,
        population_binding_file_sha256=sha256_bytes(population_binding),
        queries_file_sha256=sha256_bytes(file_bytes["queries.jsonl"]),
        baseline_scores_file_sha256=sha256_bytes(file_bytes["baseline-scores.jsonl"]),
        candidate_scores_file_sha256=sha256_bytes(file_bytes["candidate-scores.jsonl"]),
        parent_bank_file_sha256=sha256_bytes(file_bytes["parent-bank.json"]),
        candidate_bank_file_sha256=sha256_bytes(file_bytes["candidate-bank.json"]),
    )
    file_bytes["evidence-binding.json"] = evidence.canonical_bytes()
    manifest_payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-gcs-gate-export-manifest",
        "policy_version": S1_GCS_ARTIFACT_POLICY_VERSION,
        "gate_policy_version": S1_GCS_GATE_POLICY_VERSION,
        "phase": phase,
        "execution_scope": execution_scope,
        "source_split": source_split,
        "query_count": len(queries),
        **identity,
        **source_hashes,
        "files": {
            name: {"file_sha256": sha256_bytes(content), "size_bytes": len(content)}
            for name, content in sorted(file_bytes.items())
        },
    }
    manifest = PortfolioS1GCSExportManifestV1.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest_payload)),
        },
        strict=True,
    )
    root.mkdir()
    for name, content in file_bytes.items():
        (root / name).write_bytes(content)
    manifest_bytes = manifest.canonical_bytes()
    (root / "export-manifest.json").write_bytes(manifest_bytes)
    return sha256_bytes(manifest_bytes)


@pytest.fixture(scope="module")
def passed_reports(banks):
    parent, candidate = banks
    replay_queries = _queries(count=200, split="opt_pool", prefix="replay-pass")
    replay = evaluate_s1_replay(
        queries=replay_queries,
        baseline_scores=_scores(replay_queries, config="llm_static", success=False),
        candidate_scores=_scores(replay_queries, config="s1", success=True),
        evidence=_evidence("replay", banks),
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate.bank_sha256,
    )
    body_queries = _queries(count=75, split="val", prefix="body-pass")
    body = evaluate_s1_body_gate(
        replay_report=replay,
        queries=body_queries,
        baseline_scores=_scores(body_queries, config="llm_static", success=False),
        candidate_scores=_scores(body_queries, config="s1", success=True),
        evidence=_evidence("body_gate", banks),
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate.bank_sha256,
    )
    return replay, body


@pytest.fixture(scope="module")
def round2_development_freeze(banks):
    parent, candidate = banks
    queries = _queries(
        count=200,
        split="opt_pool",
        prefix="round2-development",
        required_query_ids=S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS,
    )
    baseline = _scores(queries, config="llm_static", success=True)
    candidate_scores = _scores(queries, config="s1", success=True)
    diagnostics = _contract_diagnostics(queries)
    screen = screen_s1_round2_development_patches(
        queries=queries,
        baseline_scores=baseline,
        raw_candidate_scores=candidate_scores,
        response_contract_diagnostics=diagnostics,
        source_evidence=_evidence("replay", banks),
        parent_bank_sha256=parent.bank_sha256,
        raw_candidate_bank_sha256=candidate.bank_sha256,
    )
    development = evaluate_s1_development_composite(
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=candidate_scores,
        response_contract_diagnostics=diagnostics,
    )
    freeze = make_s1_round2_candidate_freeze(
        development_report=development,
        development_screen=screen,
        development_screen_file_sha256=sha256_bytes(screen.canonical_bytes()),
        screened_bank_receipt_sha256="d" * 64,
        screened_bank_receipt_file_sha256="e" * 64,
        retained_capability_ids=(candidate.skills[0].capability_id,),
        parent_bank_sha256=parent.bank_sha256,
        parent_bank_file_sha256=sha256_bytes(parent.canonical_bytes()),
        candidate_bank_sha256=candidate.bank_sha256,
        candidate_bank_file_sha256=sha256_bytes(candidate.canonical_bytes()),
        runtime_lock_sha256="b" * 64,
        runtime_lock_file_sha256="c" * 64,
    )
    return development, freeze


def test_replay_body_gate_and_accepted_bank_are_deterministic(
    banks, passed_reports
) -> None:
    parent, candidate = banks
    replay, body = passed_reports
    assert S1_GCS_GATE_POLICY_VERSION == "portfolio-s1-gcs-replay-body-gate-v1"
    assert S1_GCS_GATE_POLICY_SHA256 == (
        "9d0dc6d291203dc4d2eefff0f6abea65c4355964eb7696b14b1543034a556e5d"
    )
    assert "paired_safety" not in replay.model_dump(mode="json")
    assert tuple(item.name for item in body.criteria) == (
        "macro_delta_pp",
        "macro_ci95_low_pp",
        "hard_error_delta_pp",
        *(f"capability_delta_pp:{capability}" for capability in GCS_CAPABILITY_ORDER),
    )
    assert replay.status == "passed"
    assert body.status == "passed"
    assert body.contrast.macro_ci95_low_pp == 100.0
    publication = build_s1_bank_disposition(
        replay_report=replay,
        body_gate_report=body,
        parent_bank=parent,
        parent_bank_bytes=parent.canonical_bytes(),
        candidate_bank=candidate,
        candidate_bank_bytes=candidate.canonical_bytes(),
    )
    assert publication.receipt.decision == "accepted"
    assert publication.output_bank_bytes == candidate.canonical_bytes()
    assert publication.receipt.provider_model_call_count == 0
    assert publication.receipt.pairwise_judge_call_count == 0
    assert publication.receipt.legacy_final_judge_call_count == 0


def test_round2_forward_body_gate_accepts_frozen_candidate_bytes(
    banks, round2_development_freeze
) -> None:
    parent, candidate = banks
    development, freeze = round2_development_freeze
    queries = _queries(count=75, split="val", prefix="round2-body-pass")

    with pytest.raises(S1GCSGateError, match="runtime identity changed"):
        evaluate_s1_round2_body_gate(
            development_report=development,
            candidate_freeze=freeze,
            queries=queries,
            baseline_scores=_scores(queries, config="llm_static", success=False),
            candidate_scores=_scores(queries, config="s1", success=True),
            response_contract_diagnostics=_contract_diagnostics(queries),
            evidence=_evidence("body_gate", banks),
            parent_bank_sha256=parent.bank_sha256,
            candidate_bank_sha256=candidate.bank_sha256,
            runtime_lock_sha256="d" * 64,
            runtime_lock_file_sha256="c" * 64,
        )

    report = evaluate_s1_round2_body_gate(
        development_report=development,
        candidate_freeze=freeze,
        queries=queries,
        baseline_scores=_scores(queries, config="llm_static", success=False),
        candidate_scores=_scores(queries, config="s1", success=True),
        response_contract_diagnostics=_contract_diagnostics(queries),
        evidence=_evidence("body_gate", banks),
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate.bank_sha256,
        runtime_lock_sha256="b" * 64,
        runtime_lock_file_sha256="c" * 64,
    )

    assert report.status == "passed"
    assert report.body_gate_report.contrast.macro_ci95_low_pp == 100.0
    publication = build_s1_round2_bank_disposition(
        body_gate_report=report,
        parent_bank=parent,
        parent_bank_bytes=parent.canonical_bytes(),
        candidate_bank=candidate,
        candidate_bank_bytes=candidate.canonical_bytes(),
    )
    assert publication.receipt.decision == "accepted"
    assert publication.output_bank_bytes == candidate.canonical_bytes()
    assert publication.receipt.provider_model_call_count == 0


def test_round2_failed_body_gate_rolls_back_byte_exact(
    banks, round2_development_freeze
) -> None:
    parent, candidate = banks
    development, freeze = round2_development_freeze
    queries = _queries(count=75, split="val", prefix="round2-body-fail")
    report = evaluate_s1_round2_body_gate(
        development_report=development,
        candidate_freeze=freeze,
        queries=queries,
        baseline_scores=_scores(queries, config="llm_static", success=True),
        candidate_scores=_scores(queries, config="s1", success=False),
        response_contract_diagnostics=_contract_diagnostics(queries),
        evidence=_evidence("body_gate", banks),
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate.bank_sha256,
        runtime_lock_sha256="b" * 64,
        runtime_lock_file_sha256="c" * 64,
    )

    assert report.status == "failed"
    publication = build_s1_round2_bank_disposition(
        body_gate_report=report,
        parent_bank=parent,
        parent_bank_bytes=parent.canonical_bytes(),
        candidate_bank=candidate,
        candidate_bank_bytes=candidate.canonical_bytes(),
    )
    assert publication.receipt.decision == "rolled_back"
    assert publication.output_bank_bytes == parent.canonical_bytes()
    assert sha256_bytes(publication.output_bank_bytes) == sha256_bytes(
        parent.canonical_bytes()
    )


def test_failed_replay_rolls_back_to_exact_parent_bytes(banks) -> None:
    parent, candidate = banks
    queries = _queries(count=200, split="opt_pool", prefix="replay-fail")
    replay = evaluate_s1_replay(
        queries=queries,
        baseline_scores=_scores(queries, config="llm_static", success=True),
        candidate_scores=_scores(queries, config="s1", success=False),
        evidence=_evidence("replay", banks),
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate.bank_sha256,
    )
    assert replay.status == "failed"
    assert "threshold_failed:macro_delta_pp" in replay.reason_codes
    publication = build_s1_bank_disposition(
        replay_report=replay,
        body_gate_report=None,
        parent_bank=parent,
        parent_bank_bytes=parent.canonical_bytes(),
        candidate_bank=candidate,
        candidate_bank_bytes=candidate.canonical_bytes(),
    )
    assert publication.receipt.decision == "rolled_back"
    assert publication.output_bank_bytes == parent.canonical_bytes()


def test_development_screen_inherits_only_regressing_capability() -> None:
    queries = _queries(count=24, split="opt_pool", prefix="development-screen")
    baseline = _scores(queries, config="llm_static", success=True)
    candidate = list(_scores(queries, config="s1", success=True))
    encyclopedia_index = next(
        index
        for index, query in enumerate(queries)
        if query.canonical_capability == "knowledge.visual_encyclopedia"
    )
    candidate[encyclopedia_index] = _failed_score(
        candidate[encyclopedia_index],
        reason_code="output_section_invalid",
    )

    screen = screen_s1_development_patches(
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=tuple(candidate),
        response_contract_diagnostics=_contract_diagnostics(queries),
    )

    by_capability = {item.capability_id: item for item in screen.decisions}
    encyclopedia = by_capability["knowledge.visual_encyclopedia"]
    assert encyclopedia.decision == "inherit_parent"
    assert encyclopedia.static_success_to_candidate_failure_count == 1
    assert encyclopedia.reason_codes == (
        "candidate_success_count_decreased",
        "contract_reason_regressed",
        "paired_static_success_regressed",
    )
    assert all(
        item.decision == "retain_patch"
        for capability, item in by_capability.items()
        if capability != "knowledge.visual_encyclopedia"
    )


def test_round2_screen_blocks_checkpoint_contract_regression_when_scores_are_zero(
    banks,
) -> None:
    parent, candidate_bank = banks
    queries = _queries(
        count=200,
        split="opt_pool",
        prefix="round2-contract-artifact",
        required_query_ids=S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS,
    )
    baseline = _scores(queries, config="llm_static", success=False)
    candidate = _scores(queries, config="s1", success=False)
    encyclopedia_query = next(
        query
        for query in queries
        if query.canonical_capability == "knowledge.visual_encyclopedia"
    )
    diagnostics = _contract_diagnostics(
        queries,
        mapped_by_key={
            (encyclopedia_query.query_id, "s1"): ("output_section_invalid",)
        },
    )

    screen = screen_s1_round2_development_patches(
        queries=queries,
        baseline_scores=baseline,
        raw_candidate_scores=candidate,
        response_contract_diagnostics=diagnostics,
        source_evidence=_evidence("replay", banks),
        parent_bank_sha256=parent.bank_sha256,
        raw_candidate_bank_sha256=candidate_bank.bank_sha256,
    )

    # The scorer could only report the pre-existing evidence failure. The
    # checkpoint-bound runner diagnostic must still make the new contract
    # failure visible to the gate and force this capability to inherit.
    assert all(
        "output_section_invalid" not in score.reason_codes
        for score in (*baseline, *candidate)
    )
    encyclopedia = next(
        item
        for item in screen.decisions
        if item.capability_id == "knowledge.visual_encyclopedia"
    )
    output_reason = next(
        item
        for item in encyclopedia.contract_reasons
        if item.reason_code == "output_section_invalid"
    )
    assert encyclopedia.decision == "inherit_parent"
    assert encyclopedia.static_success_to_candidate_failure_count == 0
    assert encyclopedia.reason_codes == ("contract_reason_regressed",)
    assert output_reason.baseline_count == 0
    assert output_reason.candidate_count == 1
    assert output_reason.new_occurrence_count == 1


def test_development_composite_requires_macro_caps_and_hard_error_safety() -> None:
    queries = _queries(count=24, split="opt_pool", prefix="development-composite")
    baseline = _scores(queries, config="llm_static", success=True)
    candidate = list(_scores(queries, config="s1", success=True))
    candidate[0] = _failed_score(
        candidate[0],
        reason_code="assistant_hard_error",
        hard_error=True,
    )

    report = evaluate_s1_development_composite(
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=tuple(candidate),
    )

    assert report.purpose == "development_only_not_acceptance"
    assert report.passed is False
    assert report.macro_nonnegative is False
    assert report.all_capabilities_nonnegative is False
    assert report.hard_error_nonincreasing is False
    assert report.reason_codes == (
        "capability_negative",
        "hard_error_increased",
        "macro_negative",
        "paired_static_success_regressed",
    )


@pytest.mark.parametrize(
    "contract_reason",
    (
        "fallback_contract_failed",
        "output_section_invalid",
        "tool_contract_failed",
    ),
)
def test_round2_body_rejects_paired_and_contract_regressions(
    banks, round2_development_freeze, contract_reason: str
) -> None:
    parent, candidate_bank = banks
    development, freeze = round2_development_freeze
    queries = _queries(count=75, split="val", prefix="round2-paired-flip")
    baseline = list(_scores(queries, config="llm_static", success=False))
    candidate = list(_scores(queries, config="s1", success=True))
    capability_indices = [
        index
        for index, query in enumerate(queries)
        if query.canonical_capability == "knowledge.visual_encyclopedia"
    ]
    first = capability_indices[0]
    successful_payload = baseline[first].model_dump(mode="python")
    successful_payload.update({"evidence_grounded": 1, "gcs": 1, "reason_codes": ()})
    baseline[first] = GCSQueryScoreV2.model_validate(successful_payload, strict=True)
    candidate[first] = _failed_score(candidate[first], reason_code=contract_reason)

    report = evaluate_s1_round2_body_gate(
        development_report=development,
        candidate_freeze=freeze,
        queries=queries,
        baseline_scores=tuple(baseline),
        candidate_scores=tuple(candidate),
        response_contract_diagnostics=_contract_diagnostics(queries),
        evidence=_evidence("body_gate", banks),
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate_bank.bank_sha256,
        runtime_lock_sha256="b" * 64,
        runtime_lock_file_sha256="c" * 64,
    )

    assert report.body_gate_report.status == "passed"
    assert report.status == "failed"
    assert report.paired_safety.static_success_to_candidate_failure_count == 1
    assert report.paired_safety.new_contract_reason_occurrence_count == 1
    assert (
        "threshold_failed:static_success_to_candidate_failure_count"
        in report.reason_codes
    )
    assert "threshold_failed:new_contract_reason_occurrence_count" in (
        report.reason_codes
    )


def test_round2_body_rejects_new_checkpoint_contract_failure_on_existing_gcs_zero(
    banks, round2_development_freeze
) -> None:
    parent, candidate_bank = banks
    development, freeze = round2_development_freeze
    queries = _queries(count=75, split="val", prefix="round2-hidden-contract")
    baseline = _scores(queries, config="llm_static", success=False)
    candidate = list(_scores(queries, config="s1", success=True))
    encyclopedia_index = next(
        index
        for index, query in enumerate(queries)
        if query.canonical_capability == "knowledge.visual_encyclopedia"
    )
    candidate[encyclopedia_index] = _failed_score(candidate[encyclopedia_index])
    affected_query = queries[encyclopedia_index]
    diagnostics = _contract_diagnostics(
        queries,
        mapped_by_key={
            (affected_query.query_id, "s1"): ("output_section_invalid",)
        },
    )

    report = evaluate_s1_round2_body_gate(
        development_report=development,
        candidate_freeze=freeze,
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=tuple(candidate),
        response_contract_diagnostics=diagnostics,
        evidence=_evidence("body_gate", banks),
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate_bank.bank_sha256,
        runtime_lock_sha256="b" * 64,
        runtime_lock_file_sha256="c" * 64,
    )

    assert baseline[encyclopedia_index].gcs == candidate[encyclopedia_index].gcs == 0
    assert report.body_gate_report.status == "passed"
    assert report.status == "failed"
    assert report.paired_safety.static_success_to_candidate_failure_count == 0
    assert report.paired_safety.new_contract_reason_occurrence_count == 1
    assert report.reason_codes == (
        "threshold_failed:new_contract_reason_occurrence_count",
    )


def test_coverage_is_unavailable_and_test_frozen_is_forbidden(banks) -> None:
    parent, candidate = banks
    queries = _queries(count=200, split="opt_pool", prefix="coverage")
    report = evaluate_s1_replay(
        queries=queries,
        baseline_scores=_scores(queries, config="llm_static", success=False),
        candidate_scores=_scores(queries, config="s1", success=True, coverage=False),
        evidence=_evidence("replay", banks),
        parent_bank_sha256=parent.bank_sha256,
        candidate_bank_sha256=candidate.bank_sha256,
    )
    assert report.status == "unavailable"
    assert "candidate_oracle_coverage_incomplete" in report.reason_codes

    test_queries = _queries(count=200, split="test_frozen", prefix="forbidden")
    with pytest.raises(S1GCSGateError, match="never consume test_frozen"):
        evaluate_s1_replay(
            queries=test_queries,
            baseline_scores=_scores(test_queries, config="llm_static", success=False),
            candidate_scores=_scores(test_queries, config="s1", success=True),
            evidence=_evidence("replay", banks),
            parent_bank_sha256=parent.bank_sha256,
            candidate_bank_sha256=candidate.bank_sha256,
        )


def test_cli_publishes_create_only_replay_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, banks
) -> None:
    parent, candidate = banks
    queries = _queries(count=200, split="opt_pool", prefix="cli")
    export_root = tmp_path / "export"
    manifest_sha256 = _write_verified_gate_export(
        export_root,
        queries=queries,
        baseline=_scores(queries, config="llm_static", success=False),
        candidate_scores=_scores(queries, config="s1", success=True),
        parent=parent,
        candidate_bank=candidate,
    )
    output = tmp_path / "out"
    argv = [
        "run_portfolio_s1_gcs_gate.py",
        "replay",
        "--export-root",
        str(export_root),
        "--export-manifest-file-sha256",
        manifest_sha256,
        "--output-dir",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gate_cli.main() == 0
    assert (output / "replay-report.json").is_file()
    assert (output / "replay-evidence-binding.json").is_file()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        gate_cli.main()


def test_round2_screen_cli_composes_failed_capabilities_from_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, banks
) -> None:
    parent, candidate = banks
    queries = _queries(
        count=200,
        split="opt_pool",
        prefix="round2-screen",
        required_query_ids=S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS,
    )
    baseline = list(_scores(queries, config="llm_static", success=True))
    candidate_scores = list(_scores(queries, config="s1", success=True))
    patched_capability = next(
        item.capability_id
        for item in candidate.skills
        if item.skill_sha256
        != next(
            parent_item.skill_sha256
            for parent_item in parent.skills
            if parent_item.capability_id == item.capability_id
        )
    )
    regressed_index = next(
        index
        for index, query in enumerate(queries)
        if query.canonical_capability == patched_capability
    )
    candidate_scores[regressed_index] = _failed_score(
        candidate_scores[regressed_index],
        reason_code="output_section_invalid",
    )
    export_root = tmp_path / "export"
    manifest_sha256 = _write_verified_gate_export(
        export_root,
        queries=queries,
        baseline=tuple(baseline),
        candidate_scores=tuple(candidate_scores),
        parent=parent,
        candidate_bank=candidate,
    )
    receipt = _sparse_receipt(parent, candidate)
    receipt_path = tmp_path / "sparse-compilation-receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    output = tmp_path / "screen"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_portfolio_s1_gcs_gate.py",
            "round2-screen",
            "--export-root",
            str(export_root),
            "--export-manifest-file-sha256",
            manifest_sha256,
            "--output-dir",
            str(output),
            "--sparse-compilation-receipt",
            str(receipt_path),
            "--sparse-compilation-receipt-file-sha256",
            sha256_bytes(receipt.canonical_bytes()),
        ],
    )

    assert gate_cli.main() == 0
    screen = parse_canonical_json(
        (output / "development-screen.json").read_bytes(),
        label="Round 2 development screen",
    )
    decisions = {
        item["capability_id"]: item["decision"] for item in screen["decisions"]
    }
    assert screen["schema_version"] == 2
    assert screen["query_count"] == 200
    assert tuple(screen["required_regression_query_ids"]) == (
        S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS
    )
    assert screen["baseline_scores_sha256"]
    assert screen["raw_candidate_scores_sha256"]
    tampered_screen = dict(screen)
    tampered_screen["raw_candidate_scores_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="self hash mismatch"):
        S1Round2DevelopmentScreen.model_validate(tampered_screen, strict=True)
    assert decisions[patched_capability] == "inherit_parent"
    screened = StaticBankArtifact.model_validate_json(
        (output / "screened-bank.json").read_bytes(), strict=True
    )
    screened_skill = next(
        item for item in screened.skills if item.capability_id == patched_capability
    )
    parent_skill = next(
        item for item in parent.skills if item.capability_id == patched_capability
    )
    assert canonical_json_bytes(screened_skill.model_dump(mode="json")) == (
        canonical_json_bytes(parent_skill.model_dump(mode="json"))
    )
    with pytest.raises(FileExistsError):
        gate_cli.main()


def test_round2_screen_cli_rejects_missing_required_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, banks
) -> None:
    parent, candidate = banks
    queries = _queries(count=200, split="opt_pool", prefix="missing-regression")
    export_root = tmp_path / "export"
    manifest_sha256 = _write_verified_gate_export(
        export_root,
        queries=queries,
        baseline=_scores(queries, config="llm_static", success=True),
        candidate_scores=_scores(queries, config="s1", success=True),
        parent=parent,
        candidate_bank=candidate,
    )
    receipt = _sparse_receipt(parent, candidate)
    receipt_path = tmp_path / "sparse-compilation-receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    output = tmp_path / "screen"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_portfolio_s1_gcs_gate.py",
            "round2-screen",
            "--export-root",
            str(export_root),
            "--export-manifest-file-sha256",
            manifest_sha256,
            "--output-dir",
            str(output),
            "--sparse-compilation-receipt",
            str(receipt_path),
            "--sparse-compilation-receipt-file-sha256",
            sha256_bytes(receipt.canonical_bytes()),
        ],
    )

    with pytest.raises(S1GCSGateError, match="missing required regressions"):
        gate_cli.main()
    assert not output.exists()


def test_round2_freeze_and_body_cli_close_screened_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, banks
) -> None:
    parent, raw_candidate = banks
    development_queries = _queries(
        count=200,
        split="opt_pool",
        prefix="round2-cli-development",
        required_query_ids=S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS,
    )
    baseline = _scores(development_queries, config="llm_static", success=True)
    raw_scores = _scores(development_queries, config="s1", success=True)
    raw_export = tmp_path / "raw-export"
    raw_manifest_sha256 = _write_verified_gate_export(
        raw_export,
        queries=development_queries,
        baseline=baseline,
        candidate_scores=raw_scores,
        parent=parent,
        candidate_bank=raw_candidate,
    )
    compilation_receipt = _sparse_receipt(parent, raw_candidate)
    compilation_receipt_path = tmp_path / "sparse-compilation-receipt.json"
    compilation_receipt_path.write_bytes(compilation_receipt.canonical_bytes())
    screen_output = tmp_path / "screen"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_portfolio_s1_gcs_gate.py",
            "round2-screen",
            "--export-root",
            str(raw_export),
            "--export-manifest-file-sha256",
            raw_manifest_sha256,
            "--output-dir",
            str(screen_output),
            "--sparse-compilation-receipt",
            str(compilation_receipt_path),
            "--sparse-compilation-receipt-file-sha256",
            sha256_bytes(compilation_receipt.canonical_bytes()),
        ],
    )
    assert gate_cli.main() == 0

    screen_path = screen_output / "development-screen.json"
    screened_bank_path = screen_output / "screened-bank.json"
    screened_receipt_path = screen_output / "screened-bank-receipt.json"
    screened_bank = StaticBankArtifact.model_validate_json(
        screened_bank_path.read_bytes(), strict=True
    )
    screened_receipt = SparseScreenedBankReceiptV1.model_validate_json(
        screened_receipt_path.read_bytes(), strict=True
    )
    assert screened_receipt.retained_capability_ids

    composite_export = tmp_path / "composite-export"
    composite_manifest_sha256 = _write_verified_gate_export(
        composite_export,
        queries=development_queries,
        baseline=baseline,
        candidate_scores=raw_scores,
        parent=parent,
        candidate_bank=screened_bank,
    )
    runtime = SimpleNamespace(runtime_lock_file_sha256="c" * 64)
    monkeypatch.setattr(
        gate_cli,
        "_verified_round2_runtime",
        lambda *args, **kwargs: (runtime, "b" * 64),
    )
    freeze_output = tmp_path / "freeze"
    freeze_argv = [
        "run_portfolio_s1_gcs_gate.py",
        "round2-freeze",
        "--export-root",
        str(composite_export),
        "--export-manifest-file-sha256",
        composite_manifest_sha256,
        "--output-dir",
        str(freeze_output),
        "--development-screen",
        str(screen_path),
        "--development-screen-file-sha256",
        sha256_bytes(screen_path.read_bytes()),
        "--screened-bank-receipt",
        str(screened_receipt_path),
        "--screened-bank-receipt-file-sha256",
        sha256_bytes(screened_receipt_path.read_bytes()),
        "--runtime-root",
        str(tmp_path / "runtime"),
        "--runtime-lock-file-sha256",
        "c" * 64,
    ]
    monkeypatch.setattr(sys, "argv", freeze_argv)
    assert gate_cli.main() == 0
    freeze = parse_canonical_json(
        (freeze_output / "candidate-freeze.json").read_bytes(),
        label="Round 2 candidate freeze",
    )
    assert freeze["schema_version"] == 2
    assert freeze["development_screen_sha256"]
    assert freeze["screened_bank_receipt_sha256"] == (screened_receipt.receipt_sha256)
    assert tuple(freeze["retained_capability_ids"]) == (
        screened_receipt.retained_capability_ids
    )
    monkeypatch.setattr(sys, "argv", freeze_argv)
    with pytest.raises(FileExistsError):
        gate_cli.main()

    body_queries = _queries(count=75, split="val", prefix="round2-cli-fresh-body")
    body_export = tmp_path / "body-export"
    body_manifest_sha256 = _write_verified_gate_export(
        body_export,
        queries=body_queries,
        baseline=_scores(body_queries, config="llm_static", success=False),
        candidate_scores=_scores(body_queries, config="s1", success=True),
        parent=parent,
        candidate_bank=screened_bank,
        phase="body_gate",
    )
    development_path = freeze_output / "development-report.json"
    freeze_path = freeze_output / "candidate-freeze.json"
    body_output = tmp_path / "body"
    body_argv = [
        "run_portfolio_s1_gcs_gate.py",
        "round2-body-gate",
        "--export-root",
        str(body_export),
        "--export-manifest-file-sha256",
        body_manifest_sha256,
        "--output-dir",
        str(body_output),
        "--development-report",
        str(development_path),
        "--development-report-file-sha256",
        sha256_bytes(development_path.read_bytes()),
        "--candidate-freeze",
        str(freeze_path),
        "--candidate-freeze-file-sha256",
        sha256_bytes(freeze_path.read_bytes()),
        "--runtime-root",
        str(tmp_path / "runtime"),
        "--runtime-lock-file-sha256",
        "c" * 64,
    ]
    monkeypatch.setattr(sys, "argv", body_argv)
    assert gate_cli.main() == 0
    disposition = parse_canonical_json(
        (body_output / "disposition-receipt.json").read_bytes(),
        label="Round 2 disposition",
    )
    assert disposition["decision"] == "accepted"
    assert (body_output / "output-bank.json").read_bytes() == (
        screened_bank.canonical_bytes()
    )

    wrong_sha_argv = body_argv.copy()
    wrong_sha_argv[wrong_sha_argv.index("--development-report-file-sha256") + 1] = (
        "f" * 64
    )
    wrong_sha_argv[wrong_sha_argv.index("--output-dir") + 1] = str(
        tmp_path / "body-wrong-sha"
    )
    monkeypatch.setattr(sys, "argv", wrong_sha_argv)
    with pytest.raises(S1GCSGateError, match="file SHA-256 drifted"):
        gate_cli.main()


def test_round2_runtime_rejects_scored_population_lock_drift(
    monkeypatch: pytest.MonkeyPatch, banks
) -> None:
    parent, candidate = banks
    parent_bytes = parent.canonical_bytes()
    candidate_bytes = candidate.canonical_bytes()
    runtime = SimpleNamespace(
        banks={"llm_static": parent, "s1": candidate},
        bank_file_sha256s={
            "llm_static": sha256_bytes(parent_bytes),
            "s1": sha256_bytes(candidate_bytes),
        },
        runtime_lock={"runtime_lock_sha256": "b" * 64},
        runtime_lock_file_sha256="c" * 64,
    )
    monkeypatch.setattr(
        gate_cli,
        "load_verified_portfolio_s1_experiment_runtime",
        lambda *args, **kwargs: runtime,
    )
    arguments = SimpleNamespace(
        runtime_root=Path("unused"), runtime_lock_file_sha256="c" * 64
    )
    with pytest.raises(S1GCSGateError, match="scored population differs"):
        gate_cli._verified_round2_runtime(
            arguments,
            parent_bytes=parent_bytes,
            parent=parent,
            candidate_bytes=candidate_bytes,
            candidate=candidate,
            population_binding={
                "runtime_lock_sha256": "d" * 64,
                "runtime_lock_file_sha256": "c" * 64,
            },
        )


def test_gate_export_rejects_wrong_root_sha_and_tampered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, banks
) -> None:
    parent, candidate = banks
    queries = _queries(count=200, split="opt_pool", prefix="tamper")
    export_root = tmp_path / "export"
    manifest_sha256 = _write_verified_gate_export(
        export_root,
        queries=queries,
        baseline=_scores(queries, config="llm_static", success=False),
        candidate_scores=_scores(queries, config="s1", success=True),
        parent=parent,
        candidate_bank=candidate,
    )
    argv = [
        "run_portfolio_s1_gcs_gate.py",
        "replay",
        "--export-root",
        str(export_root),
        "--export-manifest-file-sha256",
        "f" * 64,
        "--output-dir",
        str(tmp_path / "wrong-sha"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(S1GCSGateError, match="export verification failed"):
        gate_cli.main()

    (export_root / "candidate-scores.jsonl").write_bytes(b"{}\n")
    argv[5] = manifest_sha256
    argv[-1] = str(tmp_path / "tampered")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(S1GCSGateError, match="export verification failed"):
        gate_cli.main()


def test_analyzer_reports_external_manifest_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_sha256 = "e" * 64
    monkeypatch.setattr(
        analyzer_cli,
        "export_portfolio_s1_gcs_gate_inputs",
        lambda *args, **kwargs: SimpleNamespace(
            phase="replay",
            query_count=200,
            root=tmp_path / "export",
            population_binding_sha256="1" * 64,
            evidence_binding_sha256="2" * 64,
            baseline_scores_file_sha256="3" * 64,
            candidate_scores_file_sha256="4" * 64,
            export_manifest_file_sha256=manifest_sha256,
        ),
    )
    assert (
        analyzer_cli.main(
            [
                "--execution-root",
                str(tmp_path),
                "--execution-control-file-sha256",
                "5" * 64,
                "--artifact-repository-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "export"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert f'"export_manifest_file_sha256": "{manifest_sha256}"' in output
    assert '"export_manifest_path":' in output


def test_gate_cli_direct_script_bootstraps_repository_imports() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_portfolio_s1_gcs_gate.py"),
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_checkpoint_request_uses_strict_json_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StrictTupleRequest(BaseModel):
        model_config = ConfigDict(strict=True)

        tools: tuple[str, ...]

    payload = {"tools": ["catalog_image_search"]}
    with pytest.raises(ValidationError):
        StrictTupleRequest.model_validate(payload, strict=True)

    monkeypatch.setattr(
        artifact_exporter, "AssistantRequestSnapshot", StrictTupleRequest
    )
    request = artifact_exporter._checkpoint_request(payload)
    assert request.tools == ("catalog_image_search",)


def test_checkpoint_inventory_hash_normalizes_internal_tuple() -> None:
    inventory = (
        {
            "query_id": "replay-001",
            "config": "s1",
            "checkpoint_file_sha256": "1" * 64,
        },
    )
    expected = sha256_bytes(canonical_json_bytes(list(inventory)))
    assert artifact_exporter._checkpoint_inventory_sha256(inventory) == expected
    assert artifact_exporter._checkpoint_inventory_sha256(list(inventory)) == expected


def test_round2_sparse_replay_uses_fresh_paired_static_scores() -> None:
    queries = _queries(count=6, split="opt_pool", prefix="paired-baseline")
    baseline = _scores(queries, config="llm_static", success=True)
    candidate = _scores(queries, config="s1", success=False)
    control_sha256 = "a" * 64

    assert artifact_exporter._expected_config_order(
        "replay", {"sparse_development_paired": True}
    ) == ("llm_static", "s1")
    selected = artifact_exporter._fresh_execution_baseline(
        {"llm_static": baseline, "s1": candidate},
        phase="replay",
        sparse_development_paired=True,
        control_sha256=control_sha256,
        static_execution_root=None,
        static_expected_control_file_sha256=None,
    )
    assert selected == (
        baseline,
        {
            "source_kind": ("same-development-replay-execution-physical-llm-static"),
            "source_execution_control_sha256": control_sha256,
            "provider_model_call_count": 0,
        },
    )


def test_round2_sparse_replay_rejects_historical_static_baseline() -> None:
    with pytest.raises(
        artifact_exporter.PortfolioS1GCSArtifactError,
        match="forbids a historical Static baseline",
    ):
        artifact_exporter._fresh_execution_baseline(
            {},
            phase="replay",
            sparse_development_paired=True,
            control_sha256="a" * 64,
            static_execution_root=Path("legacy-static"),
            static_expected_control_file_sha256="b" * 64,
        )


def test_legacy_replay_keeps_static_corpus_geometry() -> None:
    assert artifact_exporter._expected_config_order("replay", {}) == ("s1",)
    assert (
        artifact_exporter._fresh_execution_baseline(
            {},
            phase="replay",
            sparse_development_paired=False,
            control_sha256="a" * 64,
            static_execution_root=Path("legacy-static"),
            static_expected_control_file_sha256="b" * 64,
        )
        is None
    )


@pytest.mark.skipif(
    not _REAL_S1_REPLAY_V5_AVAILABLE,
    reason="the immutable S1 runtime-v5 replay is local smoke evidence",
)
def test_historical_replay_export_manifest_binds_current_analyzer_sources() -> None:
    runtime_root = REAL_S1_REPLAY_ROOT_V5 / "s1-runtime-v5"
    launch_root = REAL_S1_REPLAY_ROOT_V5 / "s1-replay-launch-v5"
    execution_root = REAL_S1_REPLAY_ROOT_V5 / "s1-replay-execution-v5"
    runtime = load_verified_portfolio_s1_experiment_runtime_evidence(
        runtime_root,
        expected_runtime_lock_file_sha256=(HISTORICAL_S1_RUNTIME_V5_FILE_SHA256),
    )
    launch_plan_path = launch_root / "launch-plan.json"
    launch = load_verified_portfolio_s1_experiment_launch(
        launch_root,
        expected_plan_file_sha256=sha256_bytes(launch_plan_path.read_bytes()),
    )
    control = parse_canonical_json(
        (execution_root / "execution-control.json").read_bytes(),
        label="historical replay control",
    )
    assert isinstance(control, dict)
    files = {
        name: canonical_json_bytes({"fixture": name})
        for name in artifact_exporter._EXPORT_FILE_NAMES
    }

    manifest = artifact_exporter._build_export_manifest(
        phase="replay",
        execution_scope="s1_opt_replay",
        source_split="opt_pool",
        query_count=200,
        control=control,
        control_file_sha256=sha256_bytes(
            (execution_root / "execution-control.json").read_bytes()
        ),
        launch=launch,
        runtime=runtime,
        files=files,
    )

    assert manifest.runtime_lock_file_sha256 == (HISTORICAL_S1_RUNTIME_V5_FILE_SHA256)
    assert manifest.exporter_file_sha256 == sha256_bytes(
        Path(artifact_exporter.__file__).read_bytes()
    )
    assert manifest.analyzer_cli_file_sha256 == sha256_bytes(
        artifact_exporter._ANALYZER_CLI_PATH.read_bytes()
    )


def test_gate_export_rejects_active_gate_policy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, banks
) -> None:
    parent, candidate = banks
    queries = _queries(count=200, split="opt_pool", prefix="policy-drift")
    export_root = tmp_path / "export"
    manifest_sha256 = _write_verified_gate_export(
        export_root,
        queries=queries,
        baseline=_scores(queries, config="llm_static", success=False),
        candidate_scores=_scores(queries, config="s1", success=True),
        parent=parent,
        candidate_bank=candidate,
    )
    monkeypatch.setattr(artifact_exporter, "S1_GCS_GATE_POLICY_SHA256", "f" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_portfolio_s1_gcs_gate.py",
            "replay",
            "--export-root",
            str(export_root),
            "--export-manifest-file-sha256",
            manifest_sha256,
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(S1GCSGateError, match="export verification failed"):
        gate_cli.main()
