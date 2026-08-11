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
    build_s1_bank_disposition,
    evaluate_s1_body_gate,
    evaluate_s1_replay,
    make_s1_gcs_evidence_binding,
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
    *, count: int, split: Literal["opt_pool", "val", "test_frozen"], prefix: str
) -> tuple[Query, ...]:
    rows: list[Query] = []
    for index in range(count):
        capability = GCS_CAPABILITY_ORDER[index % len(GCS_CAPABILITY_ORDER)]
        intent, requires_card = _CAPABILITY_META[capability]
        query_id = f"{prefix}-{index:03d}"
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


def _write_verified_gate_export(
    root: Path,
    *,
    queries: tuple[Query, ...],
    baseline: tuple[GCSQueryScoreV2, ...],
    candidate_scores: tuple[GCSQueryScoreV2, ...],
    parent: StaticBankArtifact,
    candidate_bank: StaticBankArtifact,
) -> str:
    population = build_gcs_population_v2(queries)
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
        "phase": "replay",
        "execution_scope": "s1_opt_replay",
        "source_split": "opt_pool",
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
        phase="replay",
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
        "phase": "replay",
        "execution_scope": "s1_opt_replay",
        "source_split": "opt_pool",
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


def test_replay_body_gate_and_accepted_bank_are_deterministic(
    banks, passed_reports
) -> None:
    parent, candidate = banks
    replay, body = passed_reports
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
