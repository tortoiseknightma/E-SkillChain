from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_portfolio_s1_creator_inputs import (
    CODEX_INPUT_FILE,
    FEEDBACK_BUNDLE_FILE,
    INPUT_MANIFEST_FILE,
    MODEL_ACCESS_EVIDENCE_FILE,
    PARENT_BANK_FILE,
    RUNNER_ARGV_FILE,
    SEMANTIC_INPUT_FILE,
    PortfolioS1CreatorInputError,
    _load_typed_s1_feedback_bundle,
    load_verified_portfolio_s1_creator_inputs,
    prepare_portfolio_s1_creator_inputs,
    require_current_s1_codex_authoring_input,
)
from skillchain.codex_authoring import load_codex_authoring_input
from skillchain.evaluation.evaluator_outputs import VisualFeedbackOutput
from skillchain.evaluation.portfolio_gcs import GCS_V2_POLICY_SHA256
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackBundleEntryV1,
    PortfolioS1FeedbackBundleV1,
    PortfolioS1FeedbackClusterSummaryV1,
    PortfolioS1FeedbackModelEntryV1,
    PortfolioS1FeedbackModelProjectionV1,
    PortfolioS1FeedbackRepresentativeExampleV1,
    write_portfolio_s1_feedback_bundle,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (
    load_verified_portfolio_static_opt_runtime_evidence,
)
from skillchain.schemas import ConversationTurn
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
REAL_RUNTIME_V8 = (
    ROOT.parent.parent
    / "b5cd"
    / "ECommerceSkillChain"
    / "runs"
    / "portfolio"
    / "core-static-opt"
    / "static-opt-runtime-v8"
)
MODEL_ACCESS_EVIDENCE = (
    ROOT / "specs" / "authoring" / "codex-cli-model-access-evidence-v2.json"
)
CODEX_EXECUTABLE = Path(
    r"C:\Users\torto\AppData\Local\Programs\OpenAI\Codex\bin\codex.exe"
)
LEGACY_STYLE_2_1_CODEX_INPUT = (
    ROOT / "specs" / "authoring" / "authoring-packet-codex-high-v5.json"
)
_REAL_SMOKE_AVAILABLE = all(
    path.exists()
    for path in (
        REAL_RUNTIME_V8,
        MODEL_ACCESS_EVIDENCE,
        CODEX_EXECUTABLE,
        LEGACY_STYLE_2_1_CODEX_INPUT,
    )
)


def _file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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
            capability=capabilities[(index - 1) % len(capabilities)],
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
    representative_examples = tuple(
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
        representative_examples=representative_examples,
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
        **payload,
        bundle_sha256="0" * 64,
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


@pytest.mark.skipif(
    not _REAL_SMOKE_AVAILABLE,
    reason="the checked runtime-v8/Codex executable is local smoke evidence",
)
def test_real_runtime_v8_prepare_is_create_only_and_zero_call(tmp_path: Path) -> None:
    runtime_lock_file = REAL_RUNTIME_V8 / "runtime-lock.json"
    runtime = load_verified_portfolio_static_opt_runtime_evidence(
        REAL_RUNTIME_V8,
        expected_runtime_lock_file_sha256=_file_sha256(runtime_lock_file),
    )
    assert runtime.verification_scope == "immutable_evidence"
    bundle = _typed_feedback_bundle(runtime.bank.bank_sha256)
    bundle_path = tmp_path / "feedback.json"
    write_portfolio_s1_feedback_bundle(bundle_path, bundle)
    package = tmp_path / "prepared"
    future_output = tmp_path / "creator-output"

    prepared = prepare_portfolio_s1_creator_inputs(
        static_runtime_root=REAL_RUNTIME_V8,
        expected_runtime_lock_file_sha256=_file_sha256(runtime_lock_file),
        feedback_bundle_path=bundle_path,
        expected_feedback_bundle_file_sha256=_file_sha256(bundle_path),
        model_access_evidence_path=MODEL_ACCESS_EVIDENCE,
        expected_model_access_evidence_file_sha256=_file_sha256(
            MODEL_ACCESS_EVIDENCE
        ),
        codex_executable_path=CODEX_EXECUTABLE,
        future_creator_output_dir=future_output,
        output_dir=package,
    )

    assert prepared.manifest.provider_calls == 0
    assert prepared.manifest.status == "prepared_not_invoked"
    assert not future_output.exists()
    assert tuple(item.role for item in prepared.manifest.input_files) == (
        "codex_authoring_input",
        "feedback_bundle",
        "parent_static_bank",
        "semantic_authoring_input",
    )
    assert (
        prepared.manifest.model_access_evidence_file_sha256
        == _file_sha256(MODEL_ACCESS_EVIDENCE)
    )
    assert prepared.parent_bank == runtime.bank
    assert prepared.feedback_bundle == bundle
    assert (
        prepared.codex_input.semantic_source_packet_file_sha256
        == runtime.semantic_authoring_input_file_sha256
    )
    assert prepared.runner_argv.requested_model == "gpt-5.6-sol"
    assert prepared.runner_argv.reasoning_effort == "high"
    assert prepared.runner_argv.timeout_seconds == 600
    assert prepared.runner_argv.command[0] == str(Path(prepared.runner_argv.command[0]))
    assert set(item.name for item in package.iterdir()) == {
        CODEX_INPUT_FILE,
        FEEDBACK_BUNDLE_FILE,
        INPUT_MANIFEST_FILE,
        MODEL_ACCESS_EVIDENCE_FILE,
        PARENT_BANK_FILE,
        RUNNER_ARGV_FILE,
        SEMANTIC_INPUT_FILE,
    }
    reloaded = load_verified_portfolio_s1_creator_inputs(
        package,
        expected_manifest_file_sha256=prepared.manifest_file_sha256,
    )
    assert reloaded.manifest == prepared.manifest
    assert reloaded.runtime.verification_scope == "immutable_evidence"
    with pytest.raises(FileExistsError, match="create-only"):
        prepare_portfolio_s1_creator_inputs(
            static_runtime_root=REAL_RUNTIME_V8,
            expected_runtime_lock_file_sha256=_file_sha256(runtime_lock_file),
            feedback_bundle_path=bundle_path,
            expected_feedback_bundle_file_sha256=_file_sha256(bundle_path),
            model_access_evidence_path=MODEL_ACCESS_EVIDENCE,
            expected_model_access_evidence_file_sha256=_file_sha256(
                MODEL_ACCESS_EVIDENCE
            ),
            codex_executable_path=CODEX_EXECUTABLE,
            future_creator_output_dir=future_output,
            output_dir=package,
        )


@pytest.mark.skipif(
    not _REAL_SMOKE_AVAILABLE,
    reason="the checked runtime-v8/Codex packet is local smoke evidence",
)
def test_real_runtime_v8_rejects_legacy_style_2_1_codex_packet() -> None:
    runtime = load_verified_portfolio_static_opt_runtime_evidence(
        REAL_RUNTIME_V8,
        expected_runtime_lock_file_sha256=_file_sha256(
            REAL_RUNTIME_V8 / "runtime-lock.json"
        ),
    )
    legacy = load_codex_authoring_input(
        LEGACY_STYLE_2_1_CODEX_INPUT,
        expected_file_sha256=_file_sha256(LEGACY_STYLE_2_1_CODEX_INPUT),
    )
    with pytest.raises(PortfolioS1CreatorInputError, match="Style-2.3"):
        require_current_s1_codex_authoring_input(runtime, legacy)


def test_typed_feedback_loader_dispatches_exact_v3_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = canonical_json_bytes(
        {
            "schema_version": 3,
            "kind": "portfolio-s1-feedback-bundle",
            "policy_version": "portfolio-s1-feedback-bundle-v3",
        }
    )
    path = tmp_path / "feedback-v3.json"
    path.write_bytes(content)
    expected_sha256 = sha256_bytes(content)

    class _LoadedV3:
        def canonical_bytes(self) -> bytes:
            return content

    loaded = _LoadedV3()

    def _load_v3(
        candidate: str | Path,
        *,
        expected_file_sha256: str,
    ) -> _LoadedV3:
        assert Path(candidate) == path
        assert expected_file_sha256 == expected_sha256
        return loaded

    monkeypatch.setattr(
        "scripts.prepare_portfolio_s1_creator_inputs."
        "load_portfolio_s1_feedback_bundle_v3",
        _load_v3,
    )

    assert (
        _load_typed_s1_feedback_bundle(
            path,
            expected_file_sha256=expected_sha256,
        )
        is loaded
    )


def test_typed_feedback_loader_dispatches_exact_v4_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = canonical_json_bytes(
        {
            "schema_version": 4,
            "kind": "portfolio-s1-feedback-bundle",
            "policy_version": "portfolio-s1-feedback-bundle-v4",
        }
    )
    path = tmp_path / "feedback-v4.json"
    path.write_bytes(content)
    expected_sha256 = sha256_bytes(content)

    class _LoadedV4:
        def canonical_bytes(self) -> bytes:
            return content

    loaded = _LoadedV4()

    def _load_v4(
        candidate: str | Path,
        *,
        expected_file_sha256: str,
    ) -> _LoadedV4:
        assert Path(candidate) == path
        assert expected_file_sha256 == expected_sha256
        return loaded

    monkeypatch.setattr(
        "scripts.prepare_portfolio_s1_creator_inputs."
        "load_portfolio_s1_feedback_bundle_v4",
        _load_v4,
    )

    assert (
        _load_typed_s1_feedback_bundle(
            path,
            expected_file_sha256=expected_sha256,
        )
        is loaded
    )
