from __future__ import annotations

from decimal import Decimal
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from scripts import refresh_portfolio_llm_static_style_contract as refresh
from scripts import prepare_portfolio_s1_experiment as prepare_s1_cli
from scripts.run_portfolio_evolution_model import (
    CleanTurnAudit,
    CodexProcessResult,
    InvocationFileBinding,
    _build_receipt,
)
from skillchain.codex_authoring import (
    CODEX_COMMAND_SHAPE,
    build_codex_authoring_input,
    load_codex_model_access_evidence,
)
from skillchain.evaluation.assistant_runs import build_assistant_query_input
from skillchain.evaluation.portfolio_core_inputs import (
    PortfolioCoreBatch,
    VerifiedPortfolioCoreInputs,
)
from skillchain.evaluation.evaluator_outputs import VisualFeedbackOutput
from skillchain.evaluation.portfolio_gcs import (
    GCS_CAPABILITY_ORDER,
    GCS_V2_POLICY_SHA256,
    GCSQueryScoreV2,
    build_gcs_population_v2,
)
from skillchain.evaluation import portfolio_s1_experiment_runtime as s1_runtime_module
from skillchain.evaluation.portfolio_inputs import PortfolioQueryAssetBinding
from skillchain.evaluation.portfolio_s1_experiment_runtime import (
    HISTORICAL_S1_REPLAY_V5_CONTROL_FILE_SHA256,
    HISTORICAL_S1_RUNTIME_V5_FILE_SHA256,
    PortfolioS1ExperimentError,
    S1_BODY_GATE_SCOPE,
    S1_OPT_REPLAY_SCOPE,
    create_portfolio_s1_experiment_execution_control,
    create_portfolio_s1_experiment_launch_package,
    create_portfolio_s1_experiment_runtime,
    create_portfolio_s1_screened_sparse_runtime,
    load_verified_portfolio_s1_experiment_launch,
    load_verified_portfolio_s1_experiment_runtime,
    load_verified_portfolio_s1_experiment_runtime_evidence,
    load_verified_portfolio_s1_screened_sparse_runtime,
    require_verified_portfolio_s1_experiment_runtime,
    validate_portfolio_s1_experiment_evidence_control,
)
from skillchain.evolution.s1_gcs_gate import (
    S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION,
    S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS,
    S1Round2ResponseContractDiagnostic,
    evaluate_s1_development_composite,
    make_s1_gcs_evidence_binding,
    make_s1_round2_candidate_freeze,
    make_s1_round2_response_contract_diagnostics,
    screen_s1_round2_development_patches,
)
from skillchain.evolution.s1_sparse_patch import (
    S1_SPARSE_COMPILATION_POLICY_VERSION,
    SparseCompilationReceiptV1,
    compose_screened_sparse_bank,
)
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackBundleEntryV1,
    PortfolioS1FeedbackBundleV1,
    PortfolioS1FeedbackClusterSummaryV1,
    PortfolioS1FeedbackModelEntryV1,
    PortfolioS1FeedbackModelProjectionV1,
    PortfolioS1FeedbackRepresentativeExampleV1,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (
    _active_execution_contract as _active_static_opt_execution_contract,
    build_portfolio_static_opt_runtime_lock,
    load_verified_portfolio_static_opt_runtime_evidence,
    require_verified_portfolio_static_opt_runtime,
)
from skillchain.evaluation.portfolio_treatments import (
    build_portfolio_model_invocation_receipt,
    compile_verified_codex_llm_static_bank,
    load_verified_codex_draft_rebind,
)
from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.portfolio_runtime import (
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
CODEX_INPUT = ROOT / "specs/authoring/authoring-packet-codex-high-v5.json"
SEMANTIC_INPUT = ROOT / "specs/authoring/authoring-packet-primary-v5-candidate.json"
SOURCE_DRAFT = (
    ROOT
    / "runs/formal-authoring/llm-static-codex-primary-20260724-high-v5"
    / "pre-review-draft.json"
)
MODEL_ACCESS_EVIDENCE = ROOT / "specs/authoring/codex-cli-model-access-evidence-v2.json"
REAL_RUNTIME_V8 = (
    ROOT.parent.parent
    / "b5cd"
    / "ECommerceSkillChain"
    / "runs"
    / "portfolio"
    / "core-static-opt"
    / "static-opt-runtime-v8"
)
REAL_CREATOR_V5 = ROOT / "runs/portfolio/core-s1/s1-creator-v5"
REAL_S1_RUNTIME_V5 = (
    ROOT.parent.parent
    / "b5cd"
    / "ECommerceSkillChain"
    / "runs"
    / "portfolio"
    / "core-s1"
    / "s1-runtime-v5"
)
REAL_S1_REPLAY_LAUNCH_V5 = REAL_S1_RUNTIME_V5.parent / "s1-replay-launch-v5"
REAL_S1_REPLAY_EXECUTION_V5 = REAL_S1_RUNTIME_V5.parent / "s1-replay-execution-v5"
_REAL_V8_CREATOR_SMOKE_AVAILABLE = all(
    path.is_dir() for path in (REAL_RUNTIME_V8, REAL_CREATOR_V5)
)
_REAL_S1_REPLAY_V5_AVAILABLE = all(
    path.is_dir()
    for path in (
        REAL_S1_RUNTIME_V5,
        REAL_S1_REPLAY_LAUNCH_V5,
        REAL_S1_REPLAY_EXECUTION_V5,
    )
)


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _candidate_bank(parent: StaticBankArtifact) -> StaticBankArtifact:
    payload = parent.model_dump(mode="json")
    first = dict(payload["skills"][0])
    parent_skill_sha256 = first.pop("skill_sha256")
    first["version"] += 1
    first["parent_skill_sha256"] = parent_skill_sha256
    first["body"] = (
        first["body"].rstrip()
        + "\n\n## S1 replay improvement\nClose every public evidence handle.\n"
    )
    first["skill_sha256"] = sha256_bytes(canonical_json_bytes(first))
    payload["skills"][0] = first
    payload.pop("bank_sha256")
    payload["construction_identity_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {"parent_bank_sha256": parent.bank_sha256, "stage": "s1-test"}
        )
    )
    payload["bank_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return StaticBankArtifact.model_validate(payload, strict=True)


def _different_compatible_candidate(
    candidate: StaticBankArtifact,
) -> StaticBankArtifact:
    payload = candidate.model_dump(mode="json")
    second = dict(payload["skills"][1])
    parent_skill_sha256 = second.pop("skill_sha256")
    second["version"] += 1
    second["parent_skill_sha256"] = parent_skill_sha256
    second["body"] = second["body"].rstrip() + "\n\nReject invented handles.\n"
    second["skill_sha256"] = sha256_bytes(canonical_json_bytes(second))
    payload["skills"][1] = second
    payload.pop("bank_sha256")
    payload["construction_identity_sha256"] = "9" * 64
    payload["bank_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return StaticBankArtifact.model_validate(payload, strict=True)


def _sparse_receipt_for_banks(
    parent: StaticBankArtifact,
    candidate: StaticBankArtifact,
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


def _fake_sparse_creator_runtime(
    root: Path,
    runtime,
):
    fake_root = root / "raw-sparse-runtime"
    shutil.copytree(runtime.root, fake_root)
    receipt = _sparse_receipt_for_banks(
        runtime.banks["llm_static"], runtime.banks["s1"]
    )
    receipt_path = fake_root / "fixture-sparse-receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes())
    lock = dict(runtime.runtime_lock)
    lock.update(
        {
            "s1_sparse_compilation_receipt_file": receipt_path.name,
            "s1_sparse_compilation_receipt_file_sha256": _sha(receipt_path),
            "s1_sparse_compilation_receipt_sha256": receipt.receipt_sha256,
            "s1_sparse_candidate_bank_sha256": runtime.banks["s1"].bank_sha256,
        }
    )
    lock.pop("runtime_lock_sha256", None)
    lock["runtime_lock_sha256"] = sha256_bytes(canonical_json_bytes(lock))
    lock_bytes = canonical_json_bytes(lock)
    (fake_root / "runtime-lock.json").write_bytes(lock_bytes)
    return (
        replace(
            runtime,
            root=fake_root,
            runtime_lock=lock,
            runtime_lock_file_sha256=sha256_bytes(lock_bytes),
        ),
        receipt,
    )


def _typed_feedback_bundle(parent_bank_sha256: str) -> PortfolioS1FeedbackBundleV1:
    feedback = VisualFeedbackOutput(
        schema_version=1,
        summary="Follow the verified evidence and output contract.",
        rule_violations=(),
        ideal_response_gaps=(),
        skill_suggestions=("Make evidence and fallback steps explicit.",),
    )
    capabilities = (
        "knowledge.visual_encyclopedia",
        "product.exact_match",
        "product.multi_search",
        "product.style_recommendation",
        "utility.document_reading",
        "utility.recipe_guidance",
    )
    entries = tuple(
        PortfolioS1FeedbackBundleEntryV1(
            selection_ordinal=index,
            query_id=f"feedback-{index:03d}",
            capability=capabilities[(index - 1) % 6],
            role="failure" if index <= 36 else "success_anchor",
            primary_cluster="other" if index <= 36 else "success",
            leakage_group_id=f"leakage-{index:03d}",
            gcs=0 if index <= 36 else 1,
            reason_codes=("output_contract_failed",) if index <= 36 else (),
            bound_artifact_sha256=f"{index:064x}",
            feedback_result_sha256=f"{index + 100:064x}",
            feedback=feedback,
        )
        for index in range(1, 49)
    )
    model_entries = tuple(
        PortfolioS1FeedbackModelEntryV1(
            selection_ordinal=item.selection_ordinal,
            capability=item.capability,
            role=item.role,
            primary_cluster=item.primary_cluster,
            gcs=item.gcs,
            reason_codes=item.reason_codes,
            feedback=item.feedback,
        )
        for item in entries
    )
    representatives = tuple(
        PortfolioS1FeedbackRepresentativeExampleV1(
            selection_ordinal=item.selection_ordinal,
            capability=item.capability,
            role=item.role,
            primary_cluster=item.primary_cluster,
            turns=(ConversationTurn(role="user", content="Help with this item."),),
            response_text="Here is the current grounded response.",
            cards=(),
            tool_evidence=(),
        )
        for item in entries
        if item.selection_ordinal in (*range(1, 7), *range(37, 49))
    )
    projection = PortfolioS1FeedbackModelProjectionV1(
        feedback_entries=model_entries,
        representative_examples=representatives,
        cluster_summaries=(
            PortfolioS1FeedbackClusterSummaryV1(
                cluster_id="all",
                selected_count=48,
                selection_ordinals=tuple(range(1, 49)),
                rule_violation_count=0,
                ideal_response_gap_count=0,
                skill_suggestion_count=48,
            ),
        ),
    )
    payload = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-bundle",
        "policy_version": "portfolio-s1-feedback-bundle-v1",
        "selection_sha256": "1" * 64,
        "control_sha256": "2" * 64,
        "run_sha256": "6" * 64,
        "authorization_sha256": "3" * 64,
        "corpus_sha256": "4" * 64,
        "parent_static_bank_sha256": parent_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "selected_count": 48,
        "parsed_count": 48,
        "provider_call_count": 48,
        "excluded_execution_lapse_query_ids": ("private-execution-lapse",),
        "bound_artifact_set_sha256": "5" * 64,
        "entries": entries,
        "model_projection": projection,
    }
    draft = PortfolioS1FeedbackBundleV1.model_construct(
        **payload, bundle_sha256="0" * 64
    )
    unsigned = draft.model_dump(mode="json", exclude={"bundle_sha256"})
    return PortfolioS1FeedbackBundleV1.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "bundle_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        ),
        strict=True,
    )


def _creator_output(
    root: Path,
    *,
    parent: StaticBankArtifact,
    semantic,
    candidate: StaticBankArtifact,
) -> Path:
    creator = root / "creator"
    inputs = root / "creator-inputs"
    creator.mkdir()
    inputs.mkdir()
    feedback = _typed_feedback_bundle(parent.bank_sha256)
    access = load_codex_model_access_evidence(
        MODEL_ACCESS_EVIDENCE,
        expected_file_sha256=_sha(MODEL_ACCESS_EVIDENCE),
    )
    codex = build_codex_authoring_input(
        semantic_source=semantic,
        semantic_source_packet_file_sha256=sha256_bytes(semantic.canonical_bytes()),
        model_access_evidence=access,
    )
    bound_inputs = {
        "codex_authoring_input": (inputs / "codex.json", codex.canonical_bytes()),
        "feedback_bundle": (inputs / "feedback.json", feedback.canonical_bytes()),
        "parent_static_bank": (inputs / "parent.json", parent.canonical_bytes()),
        "semantic_authoring_input": (
            inputs / "semantic.json",
            semantic.canonical_bytes(),
        ),
    }
    for path, content in bound_inputs.values():
        path.write_bytes(content)
    input_files = tuple(
        InvocationFileBinding(
            role=role,
            path=str(path.absolute()),
            file_sha256=sha256_bytes(content),
            content_sha256=sha256_bytes(content),
        )
        for role, (path, content) in sorted(bound_inputs.items())
    )
    prompt = b"frozen prompt\n"
    schema = b'{"type":"object"}\n'
    events = b'{"type":"turn.completed"}\n'
    stderr = b""
    raw = b"{}"
    for name, content in {
        "prompt.txt": prompt,
        "output-schema.json": schema,
        "codex-events.jsonl": events,
        "codex-stderr.bin": stderr,
        "raw-model-output.json": raw,
        "candidate-bank.json": candidate.canonical_bytes(),
    }.items():
        (creator / name).write_bytes(content)
    executable = Path(sys.executable)
    process = CodexProcessResult(returncode=0, stdout=events, stderr=stderr)
    audit = CleanTurnAudit(
        thread_id="test-s1-thread",
        agent_message="{}",
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=50,
    )
    receipt = _build_receipt(
        stage="s1_creator",
        status="completed",
        implementation_file_sha256=_sha(
            ROOT / "scripts/run_portfolio_evolution_model.py"
        ),
        s3_textopt_compiler_file_sha256=None,
        executable=executable,
        executable_sha256=_sha(executable),
        normalized_command=CODEX_COMMAND_SHAPE,
        session_mode="ephemeral",
        session_turn_index=1,
        resume_thread_id=None,
        session_scratch_path=None,
        session_scratch_device=None,
        session_scratch_inode=None,
        prior_invocation_receipt_file_sha256=None,
        prior_gate_report_file_sha256=None,
        input_files=input_files,
        prompt_bytes=prompt,
        schema_bytes=schema,
        process=process,
        elapsed_ms=1,
        audit=audit,
        raw_final=raw,
        parent_bank_sha256=parent.bank_sha256,
        s1_feedback_bundle_sha256=feedback.bundle_sha256,
        timeout_seconds=600,
        max_final_output_bytes=65536,
        max_event_log_bytes=16777216,
        max_stderr_bytes=1048576,
        candidate_bank=candidate,
        mutation=None,
        output_dir=creator,
        error=None,
    )
    (creator / "invocation-receipt.json").write_bytes(receipt.canonical_bytes())
    model_receipt = build_portfolio_model_invocation_receipt(
        stage="s1_creator",
        requested_model="gpt-5.6-sol",
        effort="high",
        thread_id=audit.thread_id,
        prompt_sha256=sha256_bytes(prompt),
        output_sha256=sha256_bytes(raw),
        event_log_sha256=sha256_bytes(events),
        stderr_sha256=sha256_bytes(stderr),
        input_tokens=audit.input_tokens,
        output_tokens=audit.output_tokens,
    )
    (creator / "model-invocation-receipt.json").write_bytes(
        model_receipt.canonical_bytes()
    )
    return creator


@pytest.fixture(scope="module")
def experiment_runtime(tmp_path_factory):
    root = tmp_path_factory.mktemp("s1-experiment")
    verified = load_verified_codex_draft_rebind(
        codex_input_path=CODEX_INPUT,
        expected_codex_input_file_sha256=_sha(CODEX_INPUT),
        semantic_input_path=SEMANTIC_INPUT,
        expected_semantic_input_file_sha256=_sha(SEMANTIC_INPUT),
        draft_path=SOURCE_DRAFT,
        expected_draft_file_sha256=_sha(SOURCE_DRAFT),
    )
    old_bank = compile_verified_codex_llm_static_bank(
        verified, tool_registry_runtime_sha256="a" * 64
    )
    semantic, _draft, static_bank, receipt = refresh.build_contract_refresh(
        old_semantic=verified.semantic_input,
        old_draft=verified.source_draft,
        old_semantic_file_sha256=_sha(SEMANTIC_INPUT),
        old_draft_file_sha256=_sha(SOURCE_DRAFT),
        old_bank=old_bank,
        old_bank_file_sha256=sha256_bytes(old_bank.canonical_bytes()),
        tool_registry_runtime_sha256="b" * 64,
    )
    parent = root / "parent"
    core = parent / "core-runtime-sources"
    core.mkdir(parents=True)
    (parent / "bank-llm_static.json").write_bytes(static_bank.canonical_bytes())
    (parent / "semantic-authoring-input.json").write_bytes(semantic.canonical_bytes())
    (parent / "static-contract-refresh-receipt.json").write_bytes(
        canonical_json_bytes(receipt)
    )
    (parent / "system-prompt.txt").write_text(
        PORTFOLIO_SYSTEM_PROMPT, encoding="utf-8", newline=""
    )
    source = b'{"source":"fixture"}\n'
    (core / "fixture.jsonl").write_bytes(source)
    source_sha = sha256_bytes(source)
    core_payload = {
        "schema_version": 1,
        "kind": "portfolio-core-runtime-sources-receipt",
        "policy_version": "portfolio-core-runtime-sources-v1",
        "provider_call_count": 0,
        "formal_eligible": False,
        "outputs": {
            "fixture.jsonl": {
                "bytes": len(source),
                "rows": 1,
                "sha256": source_sha,
            }
        },
        "runtime_source_sha256s": [source_sha],
        "runtime_data_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
                    "source_sha256s": [source_sha],
                }
            )
        ),
    }
    core_receipt = {
        **core_payload,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(core_payload)),
    }
    (core / "receipt.json").write_bytes(canonical_json_bytes(core_receipt))
    parent_lock = build_portfolio_static_opt_runtime_lock(parent, active_contract={})
    (parent / "runtime-lock.json").write_bytes(canonical_json_bytes(parent_lock))
    candidate = _candidate_bank(static_bank)
    creator = _creator_output(
        root, parent=static_bank, semantic=semantic, candidate=candidate
    )
    return create_portfolio_s1_experiment_runtime(
        parent_static_runtime_root=parent,
        expected_parent_runtime_lock_file_sha256=_sha(parent / "runtime-lock.json"),
        creator_output_root=creator,
        expected_creator_invocation_receipt_file_sha256=_sha(
            creator / "invocation-receipt.json"
        ),
        expected_creator_model_invocation_receipt_file_sha256=_sha(
            creator / "model-invocation-receipt.json"
        ),
        expected_candidate_bank_file_sha256=_sha(creator / "candidate-bank.json"),
        output_dir=root / "runtime",
    )


@pytest.fixture(scope="module")
def synthetic_inputs() -> VerifiedPortfolioCoreInputs:
    queries = []
    assistant_queries = []
    assets = []
    batches = []
    ordinal = 0
    for split, batch_count in (("opt_pool", 32), ("val", 8)):
        for _ in range(batch_count):
            batch_id = f"core-{len(batches) + 1:03d}"
            query_ids = []
            for _position in range(25):
                ordinal += 1
                query_id = f"r2-core-{ordinal:04d}"
                text = f"Public query {ordinal}"
                query = Query(
                    schema_version=2,
                    taxonomy_version="taxonomy-test",
                    task_spec_version="task-test",
                    query_id=query_id,
                    asset_id=f"private-asset-{ordinal:04d}",
                    image_path=f"private/{ordinal:04d}.jpg",
                    leakage_group_id=f"leak-{ordinal:04d}",
                    template_family="private-template",
                    generator_batch_id=batch_id,
                    text=text,
                    turns=[ConversationTurn(role="user", content=text)],
                    canonical_intent="utility",
                    canonical_capability="utility.document_reading",
                    acceptable_capabilities=["utility.document_reading"],
                    requires_card=False,
                    split=split,
                    label_provenance=[
                        LabelDecision(
                            decision_type="constructed",
                            annotator_kind="planner",
                            annotator_id="test",
                            canonical_intent="utility",
                            canonical_capability="utility.document_reading",
                            acceptable_capabilities=["utility.document_reading"],
                        )
                    ],
                )
                assistant_queries.append(build_assistant_query_input(query))
                queries.append(query)
                assets.append(
                    PortfolioQueryAssetBinding(
                        query_id=query_id,
                        asset_id=query.asset_id,
                        image_path=query.image_path,
                        image_sha256=f"{ordinal:064x}",
                    )
                )
                query_ids.append(query_id)
            batches.append(
                PortfolioCoreBatch(
                    batch_id=batch_id, split=split, query_ids=tuple(query_ids)
                )
            )
    files = SimpleNamespace(
        expected_plan_sha256="1" * 64,
        expected_query_artifact_sha256="2" * 64,
        expected_capability_assignments_sha256="3" * 64,
        expected_base_catalog_sha256="4" * 64,
        expected_runtime_catalog_sha256="5" * 64,
    )
    return VerifiedPortfolioCoreInputs(
        files=files,  # type: ignore[arg-type]
        plan=None,  # type: ignore[arg-type]
        queries=tuple(queries),
        assistant_queries=tuple(assistant_queries),
        query_assets=tuple(assets),
        batches=tuple(batches),
        remote_runtimes=(),
        _snapshots=(),
        _marker=object(),
    )


def _passing_contract_diagnostics(queries: tuple[Query, ...]):
    rows = []
    for query in sorted(queries, key=lambda item: item.query_id):
        for config in ("llm_static", "s1"):
            prefix = f"{query.query_id}:{config}".encode()
            payload = {
                "schema_version": 1,
                "artifact_kind": ("portfolio-s1-round2-response-contract-diagnostic"),
                "policy_version": (
                    S1_ROUND2_RESPONSE_CONTRACT_DIAGNOSTIC_POLICY_VERSION
                ),
                "query_id": query.query_id,
                "config": config,
                "checkpoint_file_sha256": sha256_bytes(prefix + b":checkpoint"),
                "checkpoint_row_sha256": sha256_bytes(prefix + b":row"),
                "assistant_response_sha256": sha256_bytes(prefix + b":response"),
                "assistant_receipt_sha256": sha256_bytes(prefix + b":receipt"),
                "response_error_code": None,
                "repair_status": "not_attempted",
                "repair_attempt_count": 0,
                "repair_receipt_sha256": None,
                "repair_call_index": None,
                "repair_input_tokens": 0,
                "repair_output_tokens": 0,
                "initial_reason_codes": [],
                "final_reason_codes": [],
                "mapped_contract_reason_codes": [],
                "mapping_disposition": "not_applicable",
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
        query_ids=tuple(item.query_id for item in queries), rows=tuple(rows)
    )


def _round2_development_artifacts(
    inputs: VerifiedPortfolioCoreInputs,
    parent: StaticBankArtifact,
    candidate: StaticBankArtifact,
):
    opt_queries = [item for item in inputs.queries if item.split == "opt_pool"]
    required = set(S1_ROUND2_REQUIRED_REGRESSION_QUERY_IDS)
    source = [item for item in opt_queries if item.query_id in required]
    source.extend(
        item
        for item in opt_queries
        if item.query_id not in required and len(source) < 200
    )
    source = sorted(source, key=lambda item: item.query_id)
    assert len(source) == 200
    metadata = {
        "knowledge.visual_encyclopedia": ("encyclopedia", False),
        "product.exact_match": ("exact_match", True),
        "product.multi_search": ("multi_product", True),
        "product.style_recommendation": ("divergent_rec", True),
        "utility.document_reading": ("utility", False),
        "utility.recipe_guidance": ("utility", False),
    }
    rewritten = []
    for index, item in enumerate(source):
        capability = GCS_CAPABILITY_ORDER[index % 6]
        intent, requires_card = metadata[capability]
        payload = item.model_dump(mode="json")
        payload.update(
            {
                "canonical_capability": capability,
                "acceptable_capabilities": [capability],
                "canonical_intent": intent,
                "requires_card": requires_card,
            }
        )
        payload["label_provenance"] = [
            {
                **decision,
                "canonical_intent": intent,
                "canonical_capability": capability,
                "acceptable_capabilities": [capability],
            }
            for decision in payload["label_provenance"]
        ]
        rewritten.append(Query.model_validate(payload, strict=True))
    queries = tuple(rewritten)
    population = build_gcs_population_v2(queries)
    component_by_id = {item.query_id: item.component_id for item in population.bindings}

    def scores(config: str) -> tuple[GCSQueryScoreV2, ...]:
        return tuple(
            GCSQueryScoreV2(
                query_id=query.query_id,
                config=config,
                canonical_capability=query.canonical_capability,
                component_id=component_by_id[query.query_id],
                route_disposition="pass",
                answer_mode="supported",
                oracle_available=True,
                semantic_claim_support_resolved=True,
                route_acceptable=1,
                no_hard_error=1,
                tool_contract_pass=1,
                evidence_grounded=1,
                output_contract_pass=1,
                hard_error=0,
                gcs=1,
                reason_codes=(),
                evaluated_capability=query.canonical_capability,
                style_support_status=(
                    "candidates"
                    if query.canonical_capability == "product.style_recommendation"
                    else None
                ),
            )
            for query in queries
        )

    baseline = scores("llm_static")
    candidate_scores = scores("s1")
    diagnostics = _passing_contract_diagnostics(queries)
    evidence = make_s1_gcs_evidence_binding(
        phase="replay",
        population_binding_file_sha256="1" * 64,
        queries_file_sha256="2" * 64,
        baseline_scores_file_sha256="3" * 64,
        candidate_scores_file_sha256="4" * 64,
        parent_bank_file_sha256=sha256_bytes(parent.canonical_bytes()),
        candidate_bank_file_sha256=sha256_bytes(candidate.canonical_bytes()),
    )
    screen = screen_s1_round2_development_patches(
        queries=queries,
        baseline_scores=baseline,
        raw_candidate_scores=candidate_scores,
        response_contract_diagnostics=diagnostics,
        source_evidence=evidence,
        parent_bank_sha256=parent.bank_sha256,
        raw_candidate_bank_sha256=candidate.bank_sha256,
    )
    development = evaluate_s1_development_composite(
        queries=queries,
        baseline_scores=baseline,
        candidate_scores=candidate_scores,
        response_contract_diagnostics=diagnostics,
    )
    return development, screen


def _selection_files(tmp_path: Path, inputs: VerifiedPortfolioCoreInputs):
    opt_batches = tuple(batch for batch in inputs.batches if batch.split == "opt_pool")
    rows = []
    replay_ids = []
    for batch_index, batch in enumerate(opt_batches):
        role = "replay" if batch_index < 8 else "discovery"
        for query_id in batch.query_ids:
            rows.append(
                {
                    "schema_version": 1,
                    "query_id": query_id,
                    "atomic_batch_id": batch.batch_id,
                    "fold_id": f"fold-{batch_index + 1:02d}",
                    "role": role,
                }
            )
            if role == "replay":
                replay_ids.append(query_id)
    mapping_path = tmp_path / "fold-mapping.jsonl"
    mapping_path.write_bytes(canonical_jsonl_bytes(tuple(rows)))
    manifest = {
        "schema_version": 1,
        "policy_version": "portfolio-core-opt800-group-folds-v1",
        "plan_sha256": inputs.expected_plan_sha256,
        "source_queries_sha256": inputs.expected_query_artifact_sha256,
        "mapping_sha256": _sha(mapping_path),
        "replay_query_ids_sha256": sha256_bytes(canonical_json_bytes(replay_ids)),
    }
    manifest_path = tmp_path / "fold-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    val_queries = [query for query in inputs.queries if query.split == "val"]
    gates = {}
    for index, query in enumerate(val_queries):
        gates[query.query_id] = (
            "body_gate" if index < 75 else "route_gate" if index < 150 else "shadow_val"
        )
    gate_artifact = {
        "schema_version": 1,
        "manifest": {
            "schema_version": 1,
            "policy_version": "portfolio-core-r2-text-free-validation-gates-v1",
            "plan_sha256": inputs.expected_plan_sha256,
            "capability_assignments_sha256": (
                inputs.expected_capability_assignments_sha256
            ),
        },
        "audit": {
            "source_split": "val",
            "gate_sizes": {"route_gate": 75, "body_gate": 75, "shadow_val": 50},
            "query_id_to_gate": gates,
        },
    }
    gate_path = tmp_path / "validation-gates.json"
    gate_path.write_bytes(canonical_json_bytes(gate_artifact))
    return manifest_path, mapping_path, gate_path


def test_two_bank_runtime_is_independent_and_judge_free(experiment_runtime) -> None:
    lock = experiment_runtime.runtime_lock
    assert lock["eligible_configs"] == ["llm_static", "s1"]
    assert lock["bank_sha256s"]["llm_static"] != lock["bank_sha256s"]["s1"]
    assert lock["evaluation_stages"] == ["assistant", "gcs_v2"]
    assert lock["provider_calls"] == 0
    assert lock["pairwise_judge_enabled"] is False
    assert lock["legacy_final_judge_enabled"] is False
    assert "s1_population_runner_file_sha256" in lock["active_source_file_sha256s"]
    assert "s1_population_analyzer_file_sha256" in lock["active_source_file_sha256s"]
    assert (
        "assistant_response_contract_file_sha256" in lock["active_source_file_sha256s"]
    )
    assert lock["s1_creator_invocation_receipt_file_sha256"] == _sha(
        experiment_runtime.root / "creator-invocation-receipt.json"
    )
    assert lock["s1_creator_model_invocation_receipt_file_sha256"] == _sha(
        experiment_runtime.root / "creator-model-invocation-receipt.json"
    )
    assert lock["s1_feedback_bundle_file_sha256"] == _sha(
        experiment_runtime.root / "creator-feedback-bundle.json"
    )
    assert lock["s1_feedback_bundle_schema_version"] == 1
    assert lock["s1_feedback_bundle_policy_version"] == (
        "portfolio-s1-feedback-bundle-v1"
    )
    assert lock["s1_feedback_bundle_selected_count"] == 48
    assert lock["s1_feedback_bundle_parsed_count"] == 48
    assert lock["s1_feedback_bundle_provider_call_count"] == 48


def test_sparse_creator_replay_freshly_runs_static_and_s1(
    tmp_path: Path,
    experiment_runtime,
    synthetic_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sparse_runtime, _receipt = _fake_sparse_creator_runtime(
        tmp_path, experiment_runtime
    )
    monkeypatch.setattr(
        s1_runtime_module,
        "require_verified_portfolio_s1_experiment_runtime",
        lambda _value: sparse_runtime,
    )
    monkeypatch.setattr(
        s1_runtime_module,
        "require_verified_portfolio_core_inputs",
        lambda value: value,
    )
    manifest, mapping, _gate = _selection_files(tmp_path, synthetic_inputs)
    launch = create_portfolio_s1_experiment_launch_package(
        synthetic_inputs,
        sparse_runtime,
        execution_scope=S1_OPT_REPLAY_SCOPE,
        matrix_run_id="round2-raw-sparse-development",
        output_dir=tmp_path / "raw-sparse-launch",
        replay_manifest_path=manifest,
        expected_replay_manifest_file_sha256=_sha(manifest),
        replay_mapping_path=mapping,
        expected_replay_mapping_file_sha256=_sha(mapping),
    )
    assert launch.plan["sparse_development_paired"] is True
    assert launch.plan["config_order"] == ["llm_static", "s1"]
    assert launch.plan["query_count"] == 200
    assert launch.plan["shard_count"] == 16
    assert launch.plan["instance_count"] == 400


def test_screened_sparse_runtime_and_body_freeze_are_forward_bound(
    tmp_path: Path,
    experiment_runtime,
    synthetic_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sparse_runtime, sparse_receipt = _fake_sparse_creator_runtime(
        tmp_path, experiment_runtime
    )
    parent = sparse_runtime.banks["llm_static"]
    creator_candidate = sparse_runtime.banks["s1"]
    development, screen = _round2_development_artifacts(
        synthetic_inputs, parent, creator_candidate
    )
    screen_bytes = screen.canonical_bytes()
    retained = tuple(
        item.capability_id for item in sparse_receipt.bindings if item.action == "patch"
    )
    composed = compose_screened_sparse_bank(
        parent_bank=parent,
        creator_candidate_bank=creator_candidate,
        creator_compilation_receipt=sparse_receipt,
        development_screen_sha256=sha256_bytes(screen_bytes),
        retained_capability_ids=retained,
    )
    artifact_root = tmp_path / "screen-artifacts"
    artifact_root.mkdir()
    creator_bank_path = artifact_root / "creator-bank.json"
    sparse_receipt_path = artifact_root / "sparse-receipt.json"
    screen_path = artifact_root / "screen.json"
    screened_bank_path = artifact_root / "screened-bank.json"
    screened_receipt_path = artifact_root / "screened-receipt.json"
    creator_bank_path.write_bytes(creator_candidate.canonical_bytes())
    sparse_receipt_path.write_bytes(sparse_receipt.canonical_bytes())
    screen_path.write_bytes(screen_bytes)
    screened_bank_path.write_bytes(composed.bank.canonical_bytes())
    screened_receipt_path.write_bytes(composed.receipt.canonical_bytes())

    screened_output = tmp_path / "screened-runtime"
    original_screened_loader = load_verified_portfolio_s1_screened_sparse_runtime

    def load_runtime(root, *, expected_runtime_lock_file_sha256):
        resolved = Path(root).absolute()
        if resolved == sparse_runtime.root or resolved.name == (
            "creator-sparse-runtime-evidence"
        ):
            return sparse_runtime
        return original_screened_loader(
            resolved,
            expected_runtime_lock_file_sha256=(expected_runtime_lock_file_sha256),
        )

    monkeypatch.setattr(
        s1_runtime_module,
        "load_verified_portfolio_s1_experiment_runtime",
        load_runtime,
    )
    monkeypatch.setattr(
        s1_runtime_module,
        "require_verified_portfolio_core_inputs",
        lambda value: value,
    )
    screened_runtime = create_portfolio_s1_screened_sparse_runtime(
        creator_sparse_runtime_root=sparse_runtime.root,
        expected_creator_sparse_runtime_lock_file_sha256=(
            sparse_runtime.runtime_lock_file_sha256
        ),
        creator_candidate_bank_path=creator_bank_path,
        expected_creator_candidate_bank_file_sha256=_sha(creator_bank_path),
        creator_sparse_compilation_receipt_path=sparse_receipt_path,
        expected_creator_sparse_compilation_receipt_file_sha256=(
            _sha(sparse_receipt_path)
        ),
        development_screen_path=screen_path,
        expected_development_screen_file_sha256=_sha(screen_path),
        screened_bank_path=screened_bank_path,
        expected_screened_bank_file_sha256=_sha(screened_bank_path),
        screened_bank_receipt_path=screened_receipt_path,
        expected_screened_bank_receipt_file_sha256=_sha(screened_receipt_path),
        output_dir=screened_output,
    )
    assert screened_runtime.banks["s1"].canonical_bytes() == (
        composed.bank.canonical_bytes()
    )
    assert screened_runtime.runtime_lock["creator_candidate_bank_sha256"] == (
        creator_candidate.bank_sha256
    )
    assert screened_runtime.runtime_lock["screened_bank_receipt_sha256"] == (
        composed.receipt.receipt_sha256
    )
    with pytest.raises(PortfolioS1ExperimentError, match="output exists"):
        create_portfolio_s1_screened_sparse_runtime(
            creator_sparse_runtime_root=sparse_runtime.root,
            expected_creator_sparse_runtime_lock_file_sha256=(
                sparse_runtime.runtime_lock_file_sha256
            ),
            creator_candidate_bank_path=creator_bank_path,
            expected_creator_candidate_bank_file_sha256=_sha(creator_bank_path),
            creator_sparse_compilation_receipt_path=sparse_receipt_path,
            expected_creator_sparse_compilation_receipt_file_sha256=(
                _sha(sparse_receipt_path)
            ),
            development_screen_path=screen_path,
            expected_development_screen_file_sha256=_sha(screen_path),
            screened_bank_path=screened_bank_path,
            expected_screened_bank_file_sha256=_sha(screened_bank_path),
            screened_bank_receipt_path=screened_receipt_path,
            expected_screened_bank_receipt_file_sha256=(_sha(screened_receipt_path)),
            output_dir=screened_output,
        )

    _manifest, _mapping, gate = _selection_files(tmp_path, synthetic_inputs)
    with pytest.raises(PortfolioS1ExperimentError, match="pre-existing"):
        create_portfolio_s1_experiment_launch_package(
            synthetic_inputs,
            screened_runtime,
            execution_scope=S1_BODY_GATE_SCOPE,
            matrix_run_id="round2-body-without-freeze",
            output_dir=tmp_path / "body-without-freeze",
            validation_gate_path=tmp_path / "must-not-be-read.json",
            expected_validation_gate_file_sha256="0" * 64,
        )

    development_path = tmp_path / "development-report.json"
    development_path.write_bytes(development.canonical_bytes())
    freeze = make_s1_round2_candidate_freeze(
        development_report=development,
        development_screen=screen,
        development_screen_file_sha256=sha256_bytes(screen_bytes),
        screened_bank_receipt_sha256=composed.receipt.receipt_sha256,
        screened_bank_receipt_file_sha256=sha256_bytes(
            composed.receipt.canonical_bytes()
        ),
        retained_capability_ids=retained,
        parent_bank_sha256=screened_runtime.banks["llm_static"].bank_sha256,
        parent_bank_file_sha256=screened_runtime.bank_file_sha256s["llm_static"],
        candidate_bank_sha256=screened_runtime.banks["s1"].bank_sha256,
        candidate_bank_file_sha256=screened_runtime.bank_file_sha256s["s1"],
        runtime_lock_sha256=screened_runtime.runtime_lock["runtime_lock_sha256"],
        runtime_lock_file_sha256=screened_runtime.runtime_lock_file_sha256,
    )
    freeze_path = tmp_path / "candidate-freeze.json"
    freeze_path.write_bytes(freeze.canonical_bytes())
    body = create_portfolio_s1_experiment_launch_package(
        synthetic_inputs,
        screened_runtime,
        execution_scope=S1_BODY_GATE_SCOPE,
        matrix_run_id="round2-frozen-body",
        output_dir=tmp_path / "round2-body-launch",
        validation_gate_path=gate,
        expected_validation_gate_file_sha256=_sha(gate),
        round2_candidate_freeze_path=freeze_path,
        expected_round2_candidate_freeze_file_sha256=_sha(freeze_path),
        round2_development_report_path=development_path,
        expected_round2_development_report_file_sha256=_sha(development_path),
    )
    assert body.plan["instance_count"] == 150
    assert body.plan["round2_candidate_freeze_sha256"] == freeze.freeze_sha256
    assert body.plan["round2_development_report_sha256"] == (development.report_sha256)
    control = create_portfolio_s1_experiment_execution_control(
        body,
        screened_runtime,
        output_dir=tmp_path / "round2-body-control",
        approved_dashscope_budget_cny=Decimal("1.000000000000"),
        phase_cumulative_cap_cny=Decimal("1.000000000000"),
    )
    assert control["round2_candidate_freeze_sha256"] == freeze.freeze_sha256


def test_prepare_cli_exposes_screened_runtime_and_round2_freeze_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def create_screened(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            root=Path(kwargs["output_dir"]).absolute(),
            runtime_lock_file_sha256="a" * 64,
            runtime_lock={
                "runtime_lock_sha256": "b" * 64,
                "bank_sha256s": {"llm_static": "c" * 64, "s1": "d" * 64},
                "creator_candidate_bank_sha256": "e" * 64,
                "screened_bank_receipt_sha256": "f" * 64,
            },
        )

    monkeypatch.setattr(
        prepare_s1_cli,
        "create_portfolio_s1_screened_sparse_runtime",
        create_screened,
    )
    code = prepare_s1_cli.main(
        [
            "screened-runtime",
            "--creator-sparse-runtime-root",
            str(tmp_path / "creator-runtime"),
            "--creator-sparse-runtime-lock-file-sha256",
            "1" * 64,
            "--creator-candidate-bank",
            str(tmp_path / "creator-bank.json"),
            "--creator-candidate-bank-file-sha256",
            "2" * 64,
            "--creator-sparse-compilation-receipt",
            str(tmp_path / "sparse-receipt.json"),
            "--creator-sparse-compilation-receipt-file-sha256",
            "3" * 64,
            "--development-screen",
            str(tmp_path / "screen.json"),
            "--development-screen-file-sha256",
            "4" * 64,
            "--screened-bank",
            str(tmp_path / "screened-bank.json"),
            "--screened-bank-file-sha256",
            "5" * 64,
            "--screened-bank-receipt",
            str(tmp_path / "screened-receipt.json"),
            "--screened-bank-receipt-file-sha256",
            "6" * 64,
            "--output-dir",
            str(tmp_path / "screened-runtime"),
        ]
    )
    assert code == 0
    assert captured["development_screen_path"] == tmp_path / "screen.json"
    assert captured["screened_bank_receipt_path"] == (
        tmp_path / "screened-receipt.json"
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "verified_ready"
    assert output["provider_calls"] == 0

    parsed = prepare_s1_cli.build_parser().parse_args(
        [
            "population",
            "--scope",
            S1_BODY_GATE_SCOPE,
            "--matrix-run-id",
            "round2-body",
            "--core-input-binding-launch-root",
            str(tmp_path / "core"),
            "--core-input-binding-launch-plan-file-sha256",
            "7" * 64,
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--runtime-lock-file-sha256",
            "8" * 64,
            "--validation-gates",
            str(tmp_path / "gates.json"),
            "--validation-gates-file-sha256",
            "9" * 64,
            "--round2-candidate-freeze",
            str(tmp_path / "freeze.json"),
            "--round2-candidate-freeze-file-sha256",
            "a" * 64,
            "--round2-development-report",
            str(tmp_path / "development.json"),
            "--round2-development-report-file-sha256",
            "b" * 64,
            "--launch-output-dir",
            str(tmp_path / "launch"),
            "--execution-output-dir",
            str(tmp_path / "execution"),
            "--approved-dashscope-budget-cny",
            "1.000000000000",
            "--phase-cumulative-cap-cny",
            "1.000000000000",
        ]
    )
    assert parsed.round2_candidate_freeze == tmp_path / "freeze.json"
    assert parsed.round2_development_report == tmp_path / "development.json"


@pytest.mark.skipif(
    not _REAL_V8_CREATOR_SMOKE_AVAILABLE,
    reason="the immutable runtime-v8 and completed Creator-v5 are local smoke evidence",
)
def test_real_runtime_v8_derives_active_zero_call_s1_runtime(tmp_path: Path) -> None:
    parent = load_verified_portfolio_static_opt_runtime_evidence(
        REAL_RUNTIME_V8,
        expected_runtime_lock_file_sha256=_sha(REAL_RUNTIME_V8 / "runtime-lock.json"),
    )
    assert parent.verification_scope == "immutable_evidence"
    with pytest.raises(TypeError, match="execution requires"):
        require_verified_portfolio_static_opt_runtime(parent)

    runtime = create_portfolio_s1_experiment_runtime(
        parent_static_runtime_root=REAL_RUNTIME_V8,
        expected_parent_runtime_lock_file_sha256=_sha(
            REAL_RUNTIME_V8 / "runtime-lock.json"
        ),
        creator_output_root=REAL_CREATOR_V5,
        expected_creator_invocation_receipt_file_sha256=_sha(
            REAL_CREATOR_V5 / "invocation-receipt.json"
        ),
        expected_creator_model_invocation_receipt_file_sha256=_sha(
            REAL_CREATOR_V5 / "model-invocation-receipt.json"
        ),
        expected_candidate_bank_file_sha256=_sha(
            REAL_CREATOR_V5 / "candidate-bank.json"
        ),
        output_dir=tmp_path / "runtime",
    )
    reloaded = load_verified_portfolio_s1_experiment_runtime(
        runtime.root,
        expected_runtime_lock_file_sha256=runtime.runtime_lock_file_sha256,
    )
    assert reloaded.runtime_lock["provider_calls"] == 0
    assert reloaded.runtime_lock["model_calls_performed"] == 0
    assert reloaded.runtime_lock["parent_static_runtime_lock_file_sha256"] == _sha(
        REAL_RUNTIME_V8 / "runtime-lock.json"
    )
    for name, expected in _active_static_opt_execution_contract().items():
        assert reloaded.runtime_lock[name] == expected


@pytest.mark.skipif(
    not _REAL_S1_REPLAY_V5_AVAILABLE,
    reason="the immutable S1 runtime-v5 replay is local smoke evidence",
)
def test_real_s1_runtime_v5_is_evidence_only_and_control_is_allowlisted(
    tmp_path: Path,
) -> None:
    runtime = load_verified_portfolio_s1_experiment_runtime_evidence(
        REAL_S1_RUNTIME_V5,
        expected_runtime_lock_file_sha256=(HISTORICAL_S1_RUNTIME_V5_FILE_SHA256),
    )
    assert runtime.verification_scope == "immutable_evidence"
    assert _sha(REAL_S1_RUNTIME_V5 / "runtime-lock.json") == (
        HISTORICAL_S1_RUNTIME_V5_FILE_SHA256
    )
    with pytest.raises(TypeError, match="S1 execution requires"):
        require_verified_portfolio_s1_experiment_runtime(runtime)

    launch = load_verified_portfolio_s1_experiment_launch(
        REAL_S1_REPLAY_LAUNCH_V5,
        expected_plan_file_sha256=_sha(REAL_S1_REPLAY_LAUNCH_V5 / "launch-plan.json"),
    )
    control_path = REAL_S1_REPLAY_EXECUTION_V5 / "execution-control.json"
    control = parse_canonical_json(control_path.read_bytes(), label="replay control")
    assert isinstance(control, dict)
    assert _sha(control_path) == HISTORICAL_S1_REPLAY_V5_CONTROL_FILE_SHA256
    validate_portfolio_s1_experiment_evidence_control(control, launch, runtime)

    with pytest.raises(TypeError, match="S1 execution requires"):
        create_portfolio_s1_experiment_execution_control(
            launch,
            runtime,
            output_dir=tmp_path / "forbidden-execution",
            approved_dashscope_budget_cny=Decimal("1"),
            phase_cumulative_cap_cny=Decimal("1"),
        )
    assert not (tmp_path / "forbidden-execution").exists()

    tampered_control = dict(control)
    tampered_control["model_calls_performed"] = 1
    tampered_control.pop("control_sha256")
    tampered_control["control_sha256"] = sha256_bytes(
        canonical_json_bytes(tampered_control)
    )
    with pytest.raises(
        PortfolioS1ExperimentError,
        match="evidence control identity is not recognized",
    ):
        validate_portfolio_s1_experiment_evidence_control(
            tampered_control, launch, runtime
        )


@pytest.mark.skipif(
    not _REAL_S1_REPLAY_V5_AVAILABLE,
    reason="the immutable S1 runtime-v5 replay is local smoke evidence",
)
@pytest.mark.parametrize(
    "relative_path",
    (
        "bank-s1.json",
        "creator-invocation-receipt.json",
        "creator-output-evidence/invocation-receipt.json",
    ),
)
def test_real_s1_runtime_v5_evidence_rejects_bound_byte_tamper(
    tmp_path: Path,
    relative_path: str,
) -> None:
    copied = tmp_path / "runtime"
    shutil.copytree(REAL_S1_RUNTIME_V5, copied)
    target = copied / relative_path
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(PortfolioS1ExperimentError, match="drifted|mismatch"):
        load_verified_portfolio_s1_experiment_runtime_evidence(
            copied,
            expected_runtime_lock_file_sha256=(HISTORICAL_S1_RUNTIME_V5_FILE_SHA256),
        )


@pytest.mark.parametrize(
    ("schema_version", "policy_version", "loader_name"),
    (
        (
            1,
            "portfolio-s1-feedback-bundle-v1",
            "load_portfolio_s1_feedback_bundle",
        ),
        (
            2,
            "portfolio-s1-feedback-bundle-v2",
            "load_portfolio_s1_feedback_bundle_v2",
        ),
        (
            3,
            "portfolio-s1-feedback-bundle-v3",
            "load_portfolio_s1_feedback_bundle_v3",
        ),
        (
            4,
            "portfolio-s1-feedback-bundle-v4",
            "load_portfolio_s1_feedback_bundle_v4",
        ),
        (
            5,
            "portfolio-s1-feedback-bundle-v5",
            "load_portfolio_s1_feedback_bundle_v5",
        ),
        (
            6,
            "portfolio-s1-feedback-bundle-v6",
            "load_portfolio_s1_feedback_bundle_v6",
        ),
        (
            7,
            "portfolio-s1-feedback-bundle-v7",
            "load_portfolio_s1_feedback_bundle_v7",
        ),
        (
            8,
            "portfolio-s1-feedback-bundle-v8",
            "load_portfolio_s1_feedback_bundle_v8",
        ),
        (
            9,
            "portfolio-s1-feedback-bundle-v9",
            "load_portfolio_s1_feedback_bundle_v9",
        ),
        (
            10,
            "portfolio-s1-feedback-bundle-v10",
            "load_portfolio_s1_feedback_bundle_v10",
        ),
    ),
)
def test_creator_lineage_dispatches_only_exact_feedback_bundle_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
    policy_version: str,
    loader_name: str,
) -> None:
    import skillchain.evaluation.portfolio_s1_experiment_runtime as module

    content = canonical_json_bytes(
        {
            "schema_version": schema_version,
            "kind": "portfolio-s1-feedback-bundle",
            "policy_version": policy_version,
        }
    )
    path = tmp_path / f"feedback-v{schema_version}.json"
    path.write_bytes(content)
    expected_sha256 = sha256_bytes(content)

    class _LoadedBundle:
        def canonical_bytes(self) -> bytes:
            return content

    loaded = _LoadedBundle()

    def _load(candidate: str | Path, *, expected_file_sha256: str):
        assert Path(candidate) == path
        assert expected_file_sha256 == expected_sha256
        return loaded

    monkeypatch.setattr(module, loader_name, _load)
    assert (
        module._load_typed_s1_feedback_bundle(
            path,
            expected_file_sha256=expected_sha256,
        )
        is loaded
    )


def test_creator_lineage_rejects_arbitrary_feedback_json(tmp_path: Path) -> None:
    import skillchain.evaluation.portfolio_s1_experiment_runtime as module

    path = tmp_path / "feedback-arbitrary.json"
    path.write_bytes(canonical_json_bytes({"schema_version": 3}))
    with pytest.raises(PortfolioS1ExperimentError, match="exact Portfolio"):
        module._load_typed_s1_feedback_bundle(
            path,
            expected_file_sha256=_sha(path),
        )


def test_v7_v8_v9_and_v10_all_attempt_run_commitments_are_forward_only() -> None:
    import skillchain.evaluation.portfolio_s1_experiment_runtime as module

    run_sha256 = "a" * 64
    run_file_sha256 = "b" * 64
    v7 = SimpleNamespace(
        schema_version=7,
        run_sha256=run_sha256,
        run_file_sha256=run_file_sha256,
    )
    assert module._s1_feedback_run_lock_values(v7) == {
        "s1_feedback_bundle_run_sha256": run_sha256,
        "s1_feedback_bundle_run_file_sha256": run_file_sha256,
    }
    v8 = SimpleNamespace(
        schema_version=8,
        run_sha256="c" * 64,
        run_file_sha256="d" * 64,
        selected_count=240,
        provider_call_count=243,
    )
    assert module._s1_feedback_run_lock_values(v8) == {
        "s1_feedback_bundle_run_sha256": "c" * 64,
        "s1_feedback_bundle_run_file_sha256": "d" * 64,
        "s1_feedback_bundle_run_schema_version": 5,
        "s1_feedback_bundle_run_policy_version": "portfolio-s1-feedback-run-v5",
        "s1_feedback_bundle_all_attempt_count": 243,
        "s1_feedback_bundle_retry_count": 3,
        "s1_feedback_bundle_retry_claim_count": 3,
    }
    v9 = SimpleNamespace(
        schema_version=9,
        run_sha256="e" * 64,
        run_file_sha256="f" * 64,
        selected_count=240,
        provider_call_count=245,
        parent_provider_call_count=76,
        recovery_provider_call_count=169,
        parent_run_sha256="1" * 64,
        parent_run_file_sha256="2" * 64,
        parent_evidence_sha256="3" * 64,
        parent_evidence_file_sha256="4" * 64,
        recovery_run_sha256="e" * 64,
        recovery_run_file_sha256="f" * 64,
        recovery_authorization_sha256="5" * 64,
        recovery_control_sha256="6" * 64,
        recovery_artifact_set_sha256="7" * 64,
        recovery_claim_count=3,
        recovery_claim_sha256s=("8" * 64, "9" * 64, "a" * 64),
    )
    assert module._s1_feedback_run_lock_values(v9) == {
        "s1_feedback_bundle_run_sha256": "e" * 64,
        "s1_feedback_bundle_run_file_sha256": "f" * 64,
        "s1_feedback_bundle_run_schema_version": 1,
        "s1_feedback_bundle_run_policy_version": (
            "portfolio-s1-feedback-derived-recovery-run-v1"
        ),
        "s1_feedback_bundle_all_attempt_count": 245,
        "s1_feedback_bundle_retry_count": 5,
        "s1_feedback_bundle_retry_claim_count": 5,
        "s1_feedback_bundle_parent_provider_call_count": 76,
        "s1_feedback_bundle_parent_all_attempt_count": 76,
        "s1_feedback_bundle_recovery_provider_call_count": 169,
        "s1_feedback_bundle_recovery_all_attempt_count": 169,
        "s1_feedback_bundle_recovery_claim_count": 3,
        "s1_feedback_bundle_recovery_claim_sha256s": (
            "8" * 64,
            "9" * 64,
            "a" * 64,
        ),
        "s1_feedback_bundle_parent_run_sha256": "1" * 64,
        "s1_feedback_bundle_parent_run_file_sha256": "2" * 64,
        "s1_feedback_bundle_parent_evidence_sha256": "3" * 64,
        "s1_feedback_bundle_parent_evidence_file_sha256": "4" * 64,
        "s1_feedback_bundle_recovery_run_sha256": "e" * 64,
        "s1_feedback_bundle_recovery_run_file_sha256": "f" * 64,
        "s1_feedback_bundle_recovery_artifact_set_sha256": "7" * 64,
        "s1_feedback_bundle_recovery_authorization_sha256": "5" * 64,
        "s1_feedback_bundle_recovery_control_sha256": "6" * 64,
    }
    v10 = SimpleNamespace(
        schema_version=10,
        run_sha256="b" * 64,
        run_file_sha256="c" * 64,
        selected_count=240,
        provider_call_count=243,
        retry_claim_count=3,
        retry_claim_sha256s=("1" * 64, "2" * 64, "3" * 64),
        fresh_output_count=240,
        historical_feedback_outputs_imported=0,
        round3_run_sha256="b" * 64,
        round3_run_file_sha256="c" * 64,
        authorization_sha256="4" * 64,
        round3_authorization_sha256="4" * 64,
        control_sha256="5" * 64,
        round3_control_sha256="5" * 64,
        predecessor_receipt_sha256="6" * 64,
        predecessor_receipt_file_sha256="7" * 64,
        round3_artifact_set_sha256="a" * 64,
        bound_artifact_set_sha256="8" * 64,
        transport_policy_sha256="9" * 64,
        entry_provenance=tuple(
            SimpleNamespace(
                bound_artifact_sha256=f"{index:064x}",
                feedback_result_sha256=f"{index + 240:064x}",
                final_global_call_ordinal=index,
            )
            for index in range(1, 241)
        ),
    )
    assert module._s1_feedback_run_lock_values(v10) == {
        "s1_feedback_bundle_run_sha256": "b" * 64,
        "s1_feedback_bundle_run_file_sha256": "c" * 64,
        "s1_feedback_bundle_run_schema_version": 1,
        "s1_feedback_bundle_run_policy_version": (
            "portfolio-s1-feedback-round3-run-v1"
        ),
        "s1_feedback_bundle_all_attempt_count": 243,
        "s1_feedback_bundle_retry_count": 3,
        "s1_feedback_bundle_retry_claim_count": 3,
        "s1_feedback_bundle_retry_claim_sha256s": (
            "1" * 64,
            "2" * 64,
            "3" * 64,
        ),
        "s1_feedback_bundle_predecessor_receipt_sha256": "6" * 64,
        "s1_feedback_bundle_predecessor_receipt_file_sha256": "7" * 64,
        "s1_feedback_bundle_round3_run_sha256": "b" * 64,
        "s1_feedback_bundle_round3_run_file_sha256": "c" * 64,
        "s1_feedback_bundle_round3_authorization_sha256": "4" * 64,
        "s1_feedback_bundle_round3_control_sha256": "5" * 64,
        "s1_feedback_bundle_round3_artifact_set_sha256": "a" * 64,
        "s1_feedback_bundle_bound_artifact_set_sha256": "8" * 64,
        "s1_feedback_bundle_transport_policy_sha256": "9" * 64,
        "s1_feedback_bundle_fresh_output_count": 240,
        "s1_feedback_bundle_historical_feedback_outputs_imported": 0,
    }
    v10.round3_artifact_set_sha256 = "not-a-sha256"
    with pytest.raises(
        PortfolioS1ExperimentError,
        match="fresh attempt or provenance",
    ):
        module._s1_feedback_run_lock_values(v10)
    assert module._s1_feedback_run_lock_values(
        SimpleNamespace(schema_version=6)
    ) == {}


def test_v9_run_lock_rejects_reconstructed_claim_count_drift() -> None:
    import skillchain.evaluation.portfolio_s1_experiment_runtime as module

    with pytest.raises(PortfolioS1ExperimentError, match="attempt or claim"):
        module._s1_feedback_run_lock_values(
            SimpleNamespace(
                schema_version=9,
                run_sha256="1" * 64,
                run_file_sha256="2" * 64,
                selected_count=240,
                provider_call_count=244,
                parent_provider_call_count=76,
                recovery_provider_call_count=168,
                recovery_claim_count=3,
            )
        )


def test_v10_run_lock_rejects_fresh_retry_or_provenance_drift() -> None:
    import skillchain.evaluation.portfolio_s1_experiment_runtime as module

    with pytest.raises(PortfolioS1ExperimentError, match="fresh attempt or provenance"):
        module._s1_feedback_run_lock_values(
            SimpleNamespace(
                schema_version=10,
                run_sha256="1" * 64,
                run_file_sha256="2" * 64,
                selected_count=240,
                provider_call_count=243,
                retry_claim_count=2,
                retry_claim_sha256s=("3" * 64, "4" * 64),
                fresh_output_count=240,
                historical_feedback_outputs_imported=0,
                round3_run_sha256="1" * 64,
                round3_run_file_sha256="2" * 64,
                authorization_sha256="5" * 64,
                round3_authorization_sha256="5" * 64,
                control_sha256="6" * 64,
                round3_control_sha256="6" * 64,
                entry_provenance=tuple(range(240)),
            )
        )


@pytest.mark.parametrize(
    "relative_path",
    (
        "creator-output-evidence/invocation-receipt.json",
        "creator-output-evidence/candidate-bank.json",
        "creator-feedback-bundle.json",
    ),
)
def test_runtime_rejects_creator_lineage_tamper(
    tmp_path: Path, experiment_runtime, relative_path: str
) -> None:
    copied = tmp_path / relative_path.replace("/", "-")
    shutil.copytree(experiment_runtime.root, copied)
    (copied / relative_path).write_bytes(b"{}")
    with pytest.raises(PortfolioS1ExperimentError):
        load_verified_portfolio_s1_experiment_runtime(
            copied,
            expected_runtime_lock_file_sha256=experiment_runtime.runtime_lock_file_sha256,
        )


def test_runtime_creation_rejects_arbitrary_compatible_bank(
    tmp_path: Path, experiment_runtime
) -> None:
    creator = tmp_path / "creator"
    source_creator = experiment_runtime.root.parent / "creator"
    shutil.copytree(source_creator, creator)
    arbitrary = _different_compatible_candidate(experiment_runtime.banks["s1"])
    (creator / "candidate-bank.json").write_bytes(arbitrary.canonical_bytes())
    parent = experiment_runtime.root.parent / "parent"
    with pytest.raises(PortfolioS1ExperimentError, match="lineage"):
        create_portfolio_s1_experiment_runtime(
            parent_static_runtime_root=parent,
            expected_parent_runtime_lock_file_sha256=_sha(parent / "runtime-lock.json"),
            creator_output_root=creator,
            expected_creator_invocation_receipt_file_sha256=_sha(
                creator / "invocation-receipt.json"
            ),
            expected_creator_model_invocation_receipt_file_sha256=_sha(
                creator / "model-invocation-receipt.json"
            ),
            expected_candidate_bank_file_sha256=_sha(creator / "candidate-bank.json"),
            output_dir=tmp_path / "runtime",
        )


def test_replay_and_body_gate_have_exact_zero_call_geometry(
    tmp_path: Path,
    monkeypatch,
    capsys,
    experiment_runtime,
    synthetic_inputs,
) -> None:
    import skillchain.evaluation.portfolio_s1_experiment_runtime as module

    monkeypatch.setattr(
        module, "require_verified_portfolio_core_inputs", lambda value: value
    )
    manifest, mapping, gate = _selection_files(tmp_path, synthetic_inputs)
    replay = create_portfolio_s1_experiment_launch_package(
        synthetic_inputs,
        experiment_runtime,
        execution_scope=S1_OPT_REPLAY_SCOPE,
        matrix_run_id="s1-replay-test",
        output_dir=tmp_path / "replay",
        replay_manifest_path=manifest,
        expected_replay_manifest_file_sha256=_sha(manifest),
        replay_mapping_path=mapping,
        expected_replay_mapping_file_sha256=_sha(mapping),
    )
    body = create_portfolio_s1_experiment_launch_package(
        synthetic_inputs,
        experiment_runtime,
        execution_scope=S1_BODY_GATE_SCOPE,
        matrix_run_id="s1-body-test",
        output_dir=tmp_path / "body",
        validation_gate_path=gate,
        expected_validation_gate_file_sha256=_sha(gate),
    )
    assert (len(replay.instances), len(replay.shards)) == (200, 8)
    assert replay.plan["config_order"] == ["s1"]
    assert (len(body.instances), len(body.shards)) == (150, 6)
    assert body.plan["config_order"] == ["llm_static", "s1"]
    assert len(body.plan["selected_batch_ids"]) == 3
    assert all(item.query_count == 25 for item in (*replay.shards, *body.shards))
    assert all(
        "final" not in path.parts
        for path in (tmp_path / "body").rglob("*.json")
        if path.name not in {"launch-plan.json"}
    )

    control = create_portfolio_s1_experiment_execution_control(
        body,
        experiment_runtime,
        output_dir=tmp_path / "execution",
        approved_dashscope_budget_cny=Decimal("2.000000000000"),
        phase_cumulative_cap_cny=Decimal("1.000000000000"),
    )
    assert control["assistant_concurrency"] == 2
    assert control["local_authorized_instance_count"] == 150
    assert control["budget_authority_creation_policy"] == (
        "create_only_on_first_execute"
    )
    assert control["legacy_final_judge_enabled"] is False
    assert control["pairwise_judge_enabled"] is False

    # The standalone population entry point performs the full launch/runtime
    # preflight and honors a one-shard canary without creating a provider gate.
    from scripts import run_portfolio_s1_population as population

    # The active pricing contract deliberately fail-closes Gemini Judge calls,
    # but this GCS-only population authorizes no Final stage.  Initializing its
    # Qwen-only hard-budget ledger must therefore remain independent of the
    # unresolved Gemini pricing/ceiling contract.
    ledger_root, ledger_session = (
        population.established._load_or_initialize_budget_ledger(
            execution_root=tmp_path / "execution",
            control=control,
            matrix_run_id=body.plan["matrix_run_id"],
        )
    )
    assert ledger_root == tmp_path / "execution/budget-ledger"
    assert ledger_session.state.reservations == ()
    assert ledger_session.state.authority.matrix_run_id == "s1-body-test"

    monkeypatch.setattr(population, "_load_inputs", lambda *_args: synthetic_inputs)
    monkeypatch.setattr(
        population,
        "_build_runner",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(population, "_request_context", lambda _runtime: object())
    monkeypatch.setattr(
        population,
        "_build_requests",
        lambda _launch, _inputs, _shard, members, _context: tuple(
            object() for _member in members
        ),
    )
    assert (
        population.main(
            [
                "--execution-root",
                str(tmp_path / "execution"),
                "--dry-run",
                "--stop-after-shards",
                "1",
            ]
        )
        == 0
    )
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["status"] == "dry_run_passed"
    assert dry_run["considered_incomplete_shard_count"] == 1
    assert dry_run["model_calls_performed"] == 0
    assert not (tmp_path / "execution/provider-gate.sqlite").exists()


def test_launch_rejects_selection_byte_drift(
    tmp_path: Path,
    monkeypatch,
    experiment_runtime,
    synthetic_inputs,
) -> None:
    import skillchain.evaluation.portfolio_s1_experiment_runtime as module

    monkeypatch.setattr(
        module, "require_verified_portfolio_core_inputs", lambda value: value
    )
    manifest, mapping, _gate = _selection_files(tmp_path, synthetic_inputs)
    launch = create_portfolio_s1_experiment_launch_package(
        synthetic_inputs,
        experiment_runtime,
        execution_scope=S1_OPT_REPLAY_SCOPE,
        matrix_run_id="s1-replay-tamper-test",
        output_dir=tmp_path / "launch",
        replay_manifest_path=manifest,
        expected_replay_manifest_file_sha256=_sha(manifest),
        replay_mapping_path=mapping,
        expected_replay_mapping_file_sha256=_sha(mapping),
    )
    (launch.root / "selection/fold-manifest.json").write_bytes(b"{}")
    with pytest.raises(PortfolioS1ExperimentError, match="selection bytes"):
        load_verified_portfolio_s1_experiment_launch(
            launch.root, expected_plan_file_sha256=launch.plan_file_sha256
        )
