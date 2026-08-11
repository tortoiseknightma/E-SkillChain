from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_portfolio_evolution_model import (
    CODEX_COMMAND_SHAPE,
    CodexProcessResult,
    EvolutionInvocationReceipt,
    LEGACY_S3_IMPLEMENTATION_VERSION,
    LEGACY_S3_INVOCATION_POLICY_VERSION,
    PERSISTENT_NEW_COMMAND_SHAPE,
    PortfolioEvolutionModelError,
    S1_AUTHOR_CONTENT_FORBIDDEN_REGEX,
    S1_AUTHOR_CONTENT_FORBIDDEN_WHOLE_WORDS,
    S3_IMPLEMENTATION_VERSION,
    S3_INVOCATION_POLICY_VERSION,
    S1_IMPLEMENTATION_VERSION,
    S1_INVOCATION_POLICY_VERSION,
    S3_TEXTOPT_COMPILER_PATH,
    S3_TEXTOPT_COMPILER_SNAPSHOT_FILE,
    _actionable_mutation_scope,
    _mutation_schema,
    _parse_mutation_changes,
    _STAGE_INPUT_KIND,
    _sha_regular_file,
    run_portfolio_evolution_model,
)
from skillchain.evaluation.portfolio_attribution import (
    PORTFOLIO_OPTIMIZATION_QUERY_IDS,
    PORTFOLIO_OPTIMIZATION_QUERY_IDS_SHA256,
    PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION,
    PortfolioBodyAttributionRecord,
    PortfolioParentAttributionPacket,
    PortfolioRouteAttributionRecord,
)
from skillchain.static_authoring import (
    AuthoringContractError,
    AuthoringDraftBundle,
    StaticBankArtifact,
)
from skillchain import static_authoring as static_authoring_module
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
    build_portfolio_stage_gate_report,
)
from skillchain.evaluation.portfolio_s3_textopt import (
    PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION,
    PortfolioS3TextPatchArtifact,
    build_empty_portfolio_s3_rejected_edit_buffer,
)
from skillchain.evaluation.evaluator_outputs import VisualFeedbackOutput
from skillchain.evaluation.portfolio_gcs import GCS_V2_POLICY_SHA256
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackBundleEntryV1,
    PortfolioS1FeedbackBundleV1,
    PortfolioS1FeedbackClusterSummaryV1,
    PortfolioS1FeedbackModelEntryV1,
    PortfolioS1FeedbackModelProjectionV1,
    PortfolioS1FeedbackRepresentativeExampleV1,
)
from skillchain.schemas import ConversationTurn
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]


def test_s1_artifact_binding_kind_is_typed_feedback_bundle() -> None:
    assert _STAGE_INPUT_KIND["s1_creator"] == "s1_feedback_bundle"


@pytest.mark.parametrize("word", S1_AUTHOR_CONTENT_FORBIDDEN_WHOLE_WORDS)
def test_s1_disclosed_forbidden_words_match_trusted_validator(word: str) -> None:
    trusted = static_authoring_module._FORBIDDEN_STRONG  # noqa: SLF001
    assert trusted.pattern == S1_AUTHOR_CONTENT_FORBIDDEN_REGEX
    with pytest.raises(AuthoringContractError, match="experiment-derived"):
        static_authoring_module._scan_untrusted_text(  # noqa: SLF001
            f"safe {word} prose",
            "S1 authored prose",
        )


SEMANTIC_INPUT = (
    ROOT / "specs" / "authoring" / "authoring-packet-primary-v5-candidate.json"
)
CODEX_INPUT = ROOT / "specs" / "authoring" / "authoring-packet-codex-high-v5.json"
AUTHOR_RAW = (
    ROOT
    / "runs"
    / "formal-authoring"
    / "llm-static-codex-primary-20260724-high-v5"
    / "author-content.raw.json"
)
CAPABILITY_BY_QUERY = {
    **{
        query_id: "product.exact_match"
        for query_id in ("dm-001", "dm-004", "dm-006", "dm-007", "dm-008")
    },
    **{
        query_id: "product.multi_search"
        for query_id in ("dm-005", "dm-009", "dm-010", "dm-011", "dm-012")
    },
    **{
        query_id: "product.style_recommendation"
        for query_id in ("dm-002", "dm-013", "dm-014", "dm-015")
    },
    **{
        query_id: "knowledge.visual_encyclopedia"
        for query_id in ("dm-003", "dm-016", "dm-017", "dm-018")
    },
    **{
        query_id: "utility.document_reading"
        for query_id in ("dm-019", "dm-020", "dm-021", "dm-022")
    },
    **{
        query_id: "utility.recipe_guidance"
        for query_id in ("dm-023", "dm-024", "dm-025")
    },
}


def _file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def test_executable_hash_limit_is_independent_from_model_input_limit(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"codex")

    with pytest.raises(PortfolioEvolutionModelError, match="too large"):
        _sha_regular_file(executable, "Codex executable", max_bytes=4)
    assert _sha_regular_file(
        executable,
        "Codex executable",
        max_bytes=5,
    ) == sha256_bytes(b"codex")


def test_s1_rejects_untyped_stage_input_before_any_model_call(
    tmp_path: Path,
) -> None:
    untyped = tmp_path / "legacy-trajectories.json"
    untyped_sha = _write_canonical(
        untyped,
        {"schema_version": 1, "trajectories": [{"query_id": "legacy-1"}]},
    )
    model = _MockCodex(AUTHOR_RAW.read_bytes())
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="verified PortfolioS1FeedbackBundleV1",
    ):
        run_portfolio_evolution_model(
            stage="s1_creator",
            output_dir=tmp_path / "must-not-exist",
            stage_input_path=untyped,
            expected_stage_input_file_sha256=untyped_sha,
            semantic_authoring_input_path=SEMANTIC_INPUT,
            expected_semantic_authoring_input_file_sha256=_file_sha(SEMANTIC_INPUT),
            codex_authoring_input_path=CODEX_INPUT,
            expected_codex_authoring_input_file_sha256=_file_sha(CODEX_INPUT),
            parent_bank_path=untyped,
            expected_parent_bank_file_sha256=untyped_sha,
            tool_registry_runtime_sha256="a" * 64,
            codex_executable=_fake_codex_executable(tmp_path),
            process_runner=model,
        )
    assert model.calls == []
    assert not (tmp_path / "must-not-exist").exists()


def _write_canonical(path: Path, value: object) -> str:
    content = canonical_json_bytes(value)
    path.write_bytes(content)
    return sha256_bytes(content)


def _write_current_s1_authoring_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Refresh the unit fixture to the current Style 2.3 registry contract."""

    from skillchain.task_spec import load_mvp_task_specification_v1
    from skillchain.tools.registry import build_mvp_registry_spec

    registry = build_mvp_registry_spec(include_multi_product=True)
    task_spec = load_mvp_task_specification_v1()
    registry_bytes = canonical_json_bytes(registry.manifest.model_dump(mode="json"))
    task_spec_bytes = canonical_json_bytes(task_spec.model_dump(mode="json"))
    registry_spec = {
        "specification_kind": "tool_registry",
        "version": f"mvp-tool-registry-v{registry.manifest.schema_version}",
        "identity_sha256": registry.manifest.registry_sha256,
        "canonical_json": registry_bytes.decode("utf-8"),
        "bytes_sha256": sha256_bytes(registry_bytes),
    }
    task_specification = {
        "specification_kind": "task_specification",
        "version": task_spec.task_spec_version,
        "identity_sha256": task_spec.task_spec_sha256,
        "canonical_json": task_spec_bytes.decode("utf-8"),
        "bytes_sha256": sha256_bytes(task_spec_bytes),
    }

    semantic_raw = json.loads(SEMANTIC_INPUT.read_text(encoding="utf-8"))
    semantic_raw["task_specification"] = task_specification
    semantic_raw["tool_registry"] = registry_spec
    semantic_unsigned = {
        key: value for key, value in semantic_raw.items() if key != "input_sha256"
    }
    semantic_raw["input_sha256"] = sha256_bytes(canonical_json_bytes(semantic_unsigned))
    semantic_path = tmp_path / "semantic-authoring-input-style-2.3.json"
    semantic_file_sha256 = _write_canonical(semantic_path, semantic_raw)

    codex_raw = json.loads(CODEX_INPUT.read_text(encoding="utf-8"))
    codex_raw["task_specification"] = task_specification
    codex_raw["tool_registry"] = registry_spec
    codex_raw["semantic_source_packet_file_sha256"] = semantic_file_sha256
    codex_unsigned = {
        key: value for key, value in codex_raw.items() if key != "input_sha256"
    }
    codex_raw["input_sha256"] = sha256_bytes(canonical_json_bytes(codex_unsigned))
    codex_path = tmp_path / "codex-authoring-input-style-2.3.json"
    _write_canonical(codex_path, codex_raw)

    source_draft_path = (
        ROOT
        / "runs"
        / "formal-authoring"
        / "llm-static-codex-primary-20260724-high-v5"
        / "pre-review-draft.json"
    )
    draft_raw = json.loads(source_draft_path.read_text(encoding="utf-8"))
    draft_raw["authoring_input_sha256"] = codex_raw["input_sha256"]
    draft_unsigned = {
        key: value for key, value in draft_raw.items() if key != "bundle_sha256"
    }
    draft_raw["bundle_sha256"] = sha256_bytes(canonical_json_bytes(draft_unsigned))
    draft_path = tmp_path / "pre-review-draft-style-2.3.json"
    _write_canonical(draft_path, draft_raw)
    return semantic_path, codex_path, draft_path


def _static_parent_bank(
    *,
    runtime_sha256: str,
    semantic_input_path: Path = SEMANTIC_INPUT,
    codex_input_path: Path = CODEX_INPUT,
    draft_path: Path | None = None,
) -> StaticBankArtifact:
    from skillchain.evaluation.portfolio_treatments import (
        compile_verified_codex_llm_static_bank,
        load_verified_codex_draft_rebind,
    )

    draft_path = draft_path or (
        ROOT
        / "runs"
        / "formal-authoring"
        / "llm-static-codex-primary-20260724-high-v5"
        / "pre-review-draft.json"
    )
    rebound = load_verified_codex_draft_rebind(
        codex_input_path=codex_input_path,
        expected_codex_input_file_sha256=_file_sha(codex_input_path),
        semantic_input_path=semantic_input_path,
        expected_semantic_input_file_sha256=_file_sha(semantic_input_path),
        draft_path=draft_path,
        expected_draft_file_sha256=_file_sha(draft_path),
    )
    return compile_verified_codex_llm_static_bank(
        rebound,
        tool_registry_runtime_sha256=runtime_sha256,
    )


def _s1_feedback_bundle(parent_bank_sha256: str) -> PortfolioS1FeedbackBundleV1:
    feedback = VisualFeedbackOutput(
        schema_version=1,
        summary="Use the verified output contract more consistently.",
        rule_violations=(),
        ideal_response_gaps=(),
        skill_suggestions=("Make evidence and fallback steps explicit.",),
    )
    entries = tuple(
        PortfolioS1FeedbackBundleEntryV1(
            selection_ordinal=index,
            query_id=f"feedback-{index:03d}",
            capability=(
                "knowledge.visual_encyclopedia",
                "product.exact_match",
                "product.multi_search",
                "product.style_recommendation",
                "utility.document_reading",
                "utility.recipe_guidance",
            )[(index - 1) % 6],
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
        "excluded_execution_lapse_query_ids": ("excluded-private-query",),
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


def _write_s1_bound_inputs(
    tmp_path: Path,
    *,
    runtime_sha256: str = "a" * 64,
) -> tuple[
    StaticBankArtifact,
    Path,
    str,
    PortfolioS1FeedbackBundleV1,
    Path,
    str,
    Path,
    Path,
]:
    semantic_path, codex_path, draft_path = _write_current_s1_authoring_inputs(tmp_path)
    parent_bank = _static_parent_bank(
        runtime_sha256=runtime_sha256,
        semantic_input_path=semantic_path,
        codex_input_path=codex_path,
        draft_path=draft_path,
    )
    parent_path = tmp_path / "parent-static-bank.json"
    parent_file_sha256 = _write_canonical(
        parent_path,
        parent_bank.model_dump(mode="json"),
    )
    feedback_bundle = _s1_feedback_bundle(parent_bank.bank_sha256)
    feedback_path = tmp_path / "s1-feedback-bundle.json"
    feedback_file_sha256 = _write_canonical(
        feedback_path,
        feedback_bundle.model_dump(mode="json"),
    )
    return (
        parent_bank,
        parent_path,
        parent_file_sha256,
        feedback_bundle,
        feedback_path,
        feedback_file_sha256,
        semantic_path,
        codex_path,
    )


def _write_parent_attribution(
    path: Path,
    *,
    stage: str,
    source_bank_sha256: str,
    route_confusion_by_query: dict[str, str] | None = None,
    low_tier_query_ids: tuple[str, ...] = ("dm-001", "dm-004"),
    hard_error_query_ids: tuple[str, ...] = (),
    judge_failure_query_ids: tuple[str, ...] = (),
    include_visible_cards: bool = True,
    include_visible_evidence: bool = True,
) -> str:
    query_ids = PORTFOLIO_OPTIMIZATION_QUERY_IDS
    route_confusions = (
        {"dm-001": "product.multi_search", "dm-004": "product.multi_search"}
        if route_confusion_by_query is None
        else dict(route_confusion_by_query)
    )
    low_tier_queries = frozenset(low_tier_query_ids)
    hard_error_queries = frozenset(hard_error_query_ids)
    judge_failure_queries = frozenset(judge_failure_query_ids)
    records = []
    for ordinal, (query_id, capability) in enumerate(
        ((query_id, CAPABILITY_BY_QUERY[query_id]) for query_id in query_ids)
    ):
        selected_capability = capability
        if stage == "s2_route_optimizer" and query_id in route_confusions:
            selected_capability = route_confusions[query_id]
        body_hard_error = stage == "s3_body_refiner" and query_id in hard_error_queries
        judge_failure = stage == "s3_body_refiner" and query_id in judge_failure_queries
        common = {
            "schema_version": 1,
            "record_ordinal": ordinal,
            "query_id": query_id,
            "canonical_capability": capability,
            "selected_capability": selected_capability,
            "route_correct": selected_capability == capability,
            "skill_slug": f"skill-{ordinal}",
            "route_trace_sha256": "1" * 64,
            "assistant_error_code": (
                "execution_contract_failed" if body_hard_error else None
            ),
            "source_result_sha256": "2" * 64,
            "source_assistant_file_sha256": "3" * 64,
            "source_assistant_row_sha256": "4" * 64,
            "source_request_sha256": "5" * 64,
            "source_response_sha256": "6" * 64,
            "source_receipt_sha256": "7" * 64,
            "source_final_file_sha256": "8" * 64,
            "source_final_result_sha256": "9" * 64,
        }
        if stage == "s2_route_optimizer":
            payload = {
                **common,
                "kind": "portfolio-current-parent-route-attribution",
                "tool_names": [],
            }
            record = PortfolioRouteAttributionRecord.model_validate(
                {
                    **payload,
                    "record_sha256": sha256_bytes(canonical_json_bytes(payload)),
                },
                strict=True,
            )
        else:
            dimensions = []
            if not body_hard_error:
                dimensions = [
                    {
                        "dimension": "TCR",
                        "score": 5 if query_id in low_tier_queries else 10,
                        "tier": ("Average" if query_id in low_tier_queries else "Good"),
                    }
                ]
            payload = {
                **common,
                "kind": "portfolio-current-parent-body-attribution",
                "response_text": "" if body_hard_error else "Supported response.",
                "visible_cards": (
                    [
                        {
                            "title": "Grounded result",
                            "body": "Visible candidate evidence.",
                            "fields": [],
                        }
                    ]
                    if include_visible_cards and not body_hard_error
                    else []
                ),
                "visible_tool_evidence": (
                    [
                        {
                            "tool_name": "text_product_search",
                            "status": "success",
                            "visible_text": "Public grounded evidence.",
                            "cards": [],
                            "citations": [],
                            "detections": [],
                            "error_code": None,
                        }
                    ]
                    if include_visible_evidence and not body_hard_error
                    else []
                ),
                "tool_trace": [],
                "backend_error_code": None,
                "hard_error": body_hard_error or judge_failure,
                "final_kind": (
                    "assistant_fixed_zero" if body_hard_error else "visual_final_judge"
                ),
                "evaluation_id": None if body_hard_error else "a" * 64,
                "judge_status": (
                    "assistant_fixed_zero"
                    if body_hard_error
                    else "provider_error"
                    if judge_failure
                    else "scored"
                ),
                "judge_error_code": "provider_error" if judge_failure else None,
                "dimensions": dimensions,
                "j_project": 0.0 if body_hard_error or judge_failure else 100.0,
            }
            record = PortfolioBodyAttributionRecord.model_validate(
                {
                    **payload,
                    "record_sha256": sha256_bytes(canonical_json_bytes(payload)),
                },
                strict=True,
            )
        records.append(record)
    packet_payload = {
        "schema_version": 1,
        "kind": "portfolio-current-parent-attribution-packet",
        "policy_version": PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION,
        "stage": stage,
        "source_config": ("s1" if stage == "s2_route_optimizer" else "s1s2"),
        "source_bank_sha256": source_bank_sha256,
        "optimization_query_ids_sha256": PORTFOLIO_OPTIMIZATION_QUERY_IDS_SHA256,
        "source_summary_file_sha256": "c" * 64,
        "source_summary_sha256": "d" * 64,
        "source_results_file_sha256": "e" * 64,
        "query_ids": list(query_ids),
        "records": [item.model_dump(mode="json") for item in records],
    }
    packet = PortfolioParentAttributionPacket.model_validate(
        {
            **packet_payload,
            "packet_sha256": sha256_bytes(canonical_json_bytes(packet_payload)),
        },
        strict=True,
    )
    path.write_bytes(packet.canonical_bytes())
    return sha256_bytes(packet.canonical_bytes())


def _read_parent_attribution(path: Path) -> PortfolioParentAttributionPacket:
    return PortfolioParentAttributionPacket.model_validate_json(
        path.read_bytes(),
        strict=True,
    )


def _first_editable_rule(body: str) -> tuple[str, str]:
    in_success = False
    for line in body.splitlines():
        if line == "## Success criteria":
            in_success = True
            continue
        if in_success and line.startswith("## "):
            break
        if in_success and line.startswith("- [") and "] " in line:
            return line[3 : line.index("] ")], line
    raise AssertionError("fixture Body lacks an editable success rule")


def test_s2_scope_prioritizes_repeated_exact_pairs(tmp_path: Path) -> None:
    repeated_path = tmp_path / "s2-repeated.json"
    _write_parent_attribution(
        repeated_path,
        stage="s2_route_optimizer",
        source_bank_sha256="0" * 64,
    )
    capability_ids, scope = _actionable_mutation_scope(
        "s2_route_optimizer",
        _read_parent_attribution(repeated_path),
    )
    assert capability_ids == ("product.exact_match", "product.multi_search")
    assert scope["evidence_strength"] == "repeated_exact_pair"
    assert scope["recurring_confusion_endpoints"] == []
    assert scope["confusion_pairs"] == [
        {
            "expected_capability": "product.exact_match",
            "observed_capability": "product.multi_search",
            "count": 2,
            "query_ids": ["dm-001", "dm-004"],
            "actionable": True,
        }
    ]


def test_s2_scope_falls_back_to_recurring_confusion_endpoint(tmp_path: Path) -> None:
    attribution_path = tmp_path / "s2-recurring-endpoint.json"
    _write_parent_attribution(
        attribution_path,
        stage="s2_route_optimizer",
        source_bank_sha256="0" * 64,
        route_confusion_by_query={
            "dm-001": "product.multi_search",
            "dm-004": "product.style_recommendation",
        },
    )
    capability_ids, scope = _actionable_mutation_scope(
        "s2_route_optimizer",
        _read_parent_attribution(attribution_path),
    )
    assert capability_ids == (
        "product.exact_match",
        "product.multi_search",
        "product.style_recommendation",
    )
    assert scope["evidence_strength"] == "recurring_confusion_endpoint"
    assert scope["recurring_confusion_endpoints"] == ["product.exact_match"]
    assert all(item["actionable"] for item in scope["confusion_pairs"])


def test_s2_scope_uses_one_deterministic_singleton_pair(tmp_path: Path) -> None:
    attribution_path = tmp_path / "s2-singleton-fallback.json"
    _write_parent_attribution(
        attribution_path,
        stage="s2_route_optimizer",
        source_bank_sha256="0" * 64,
        route_confusion_by_query={
            "dm-001": "product.multi_search",
            "dm-002": "knowledge.visual_encyclopedia",
        },
    )
    capability_ids, scope = _actionable_mutation_scope(
        "s2_route_optimizer",
        _read_parent_attribution(attribution_path),
    )
    assert capability_ids == ("product.exact_match", "product.multi_search")
    assert scope["evidence_strength"] == "singleton_fallback"
    assert scope["recurring_confusion_endpoints"] == []
    assert scope["confusion_pairs"] == [
        {
            "expected_capability": "product.exact_match",
            "observed_capability": "product.multi_search",
            "count": 1,
            "query_ids": ["dm-001"],
            "actionable": True,
        },
        {
            "expected_capability": "product.style_recommendation",
            "observed_capability": "knowledge.visual_encyclopedia",
            "count": 1,
            "query_ids": ["dm-002"],
            "actionable": False,
        },
    ]


def test_s3_scope_uses_only_repeated_scored_skill_defects(tmp_path: Path) -> None:
    singleton_path = tmp_path / "s3-singleton-low-tier.json"
    _write_parent_attribution(
        singleton_path,
        stage="s3_body_refiner",
        source_bank_sha256="0" * 64,
        low_tier_query_ids=("dm-001",),
    )
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="no recurring scored low-tier dimension",
    ):
        _actionable_mutation_scope(
            "s3_body_refiner",
            _read_parent_attribution(singleton_path),
        )

    judge_failure_path = tmp_path / "s3-judge-outage.json"
    _write_parent_attribution(
        judge_failure_path,
        stage="s3_body_refiner",
        source_bank_sha256="0" * 64,
        low_tier_query_ids=(),
        judge_failure_query_ids=("dm-001",),
    )
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="no recurring scored low-tier dimension",
    ):
        _actionable_mutation_scope(
            "s3_body_refiner",
            _read_parent_attribution(judge_failure_path),
        )

    repeated_path = tmp_path / "s3-repeated-low-tier.json"
    _write_parent_attribution(
        repeated_path,
        stage="s3_body_refiner",
        source_bank_sha256="0" * 64,
    )
    capability_ids, scope = _actionable_mutation_scope(
        "s3_body_refiner",
        _read_parent_attribution(repeated_path),
    )
    assert capability_ids == ("product.exact_match",)
    aggregates = {
        item["capability_id"]: item for item in scope["capability_aggregates"]
    }
    assert {key: value["sample_count"] for key, value in aggregates.items()} == {
        "knowledge.visual_encyclopedia": 4,
        "product.exact_match": 5,
        "product.multi_search": 5,
        "product.style_recommendation": 4,
        "utility.document_reading": 4,
        "utility.recipe_guidance": 3,
    }
    assert aggregates["product.exact_match"]["judge_path"] == {
        "tier_counts": [
            {"dimension": "TCR", "tier": "Average", "count": 2},
            {"dimension": "TCR", "tier": "Good", "count": 3},
        ],
        "recurring_low_dimensions": ["TCR"],
    }
    cluster = scope["failure_clusters"]
    assert len(cluster) == 1
    assert cluster[0]["classification"] == "SKILL_DEFECT"
    assert cluster[0]["evidence_ids"] == ["dm-001", "dm-004"]
    assert cluster[0]["addressed_dimensions"] == ["TCR"]
    assert cluster[0]["support_count"] == 2

    hard_error_path = tmp_path / "s3-one-hard-error.json"
    _write_parent_attribution(
        hard_error_path,
        stage="s3_body_refiner",
        source_bank_sha256="0" * 64,
        low_tier_query_ids=(),
        hard_error_query_ids=("dm-001",),
    )
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="execution lapses cannot mutate Skill Body",
    ):
        _actionable_mutation_scope(
            "s3_body_refiner",
            _read_parent_attribution(hard_error_path),
        )


def _clean_events(
    message: str,
    *,
    thread_id: str = "portfolio-test-thread",
) -> bytes:
    events = (
        {"type": "thread.started", "thread_id": thread_id},
        {
            "type": "turn.started",
            "thread_id": thread_id,
        },
        {
            "type": "item.completed",
            "thread_id": thread_id,
            "item": {
                "id": "message-1",
                "type": "agent_message",
                "text": message,
            },
        },
        {
            "type": "turn.completed",
            "thread_id": thread_id,
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 30,
            },
        },
    )
    return b"".join(canonical_json_bytes(item) for item in events)


class _MockCodex:
    def __init__(
        self,
        raw_output: bytes,
        *,
        events: bytes | None = None,
        session_mode: str = "ephemeral",
        thread_id: str = "portfolio-test-thread",
    ):
        self.raw_output = raw_output
        self.events = events
        self.session_mode = session_mode
        self.thread_id = thread_id
        self.calls: list[tuple[tuple[str, ...], bytes, Path]] = []

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        stdin: bytes,
        cwd: Path,
        environment,
        timeout_seconds: int,
    ) -> CodexProcessResult:
        del environment
        assert timeout_seconds == 600
        assert not tuple(cwd.iterdir())
        assert "--output-schema" in command
        assert "--ignore-user-config" in command
        assert "--ignore-rules" in command
        if self.session_mode == "ephemeral":
            assert "--ephemeral" in command
            assert command[command.index("--sandbox") + 1] == "read-only"
        elif self.session_mode == "new_persistent":
            assert "--ephemeral" not in command
            assert command[:2] == (str(command[0]), "exec")
            assert command[command.index("--sandbox") + 1] == "read-only"
        else:
            assert self.session_mode == "resume"
            assert command[1:3] == ("exec", "resume")
            assert "--ephemeral" not in command
            assert "--sandbox" not in command
            assert command[-2:] == (self.thread_id, "-")
        assert command[command.index("--model") + 1] == "gpt-5.6-sol"
        if self.session_mode != "resume":
            assert command[-3:] == (
                "--config",
                'model_reasoning_effort="high"',
                "-",
            )
        final_path = Path(command[command.index("--output-last-message") + 1])
        final_path.write_bytes(self.raw_output)
        self.calls.append((command, stdin, cwd))
        events = self.events or _clean_events(
            self.raw_output.decode("utf-8"),
            thread_id=self.thread_id,
        )
        return CodexProcessResult(
            returncode=0,
            stdout=events,
            stderr=b"",
        )


def _fake_codex_executable(tmp_path: Path) -> Path:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"mock codex binary")
    return executable


def _load_receipt(path: Path) -> EvolutionInvocationReceipt:
    content = path.read_bytes()
    raw = parse_canonical_json(content, label="test invocation receipt")
    assert isinstance(raw, dict)
    receipt = EvolutionInvocationReceipt.model_validate(raw, strict=True)
    assert receipt.canonical_bytes() == content
    return receipt


def test_s1_rejects_budget_drift_before_model_call(tmp_path: Path) -> None:
    (
        _parent,
        parent_path,
        parent_sha,
        _bundle,
        feedback_path,
        feedback_sha,
        semantic_path,
        codex_path,
    ) = _write_s1_bound_inputs(tmp_path)
    model = _MockCodex(AUTHOR_RAW.read_bytes())
    with pytest.raises(PortfolioEvolutionModelError, match="timeout must exactly"):
        run_portfolio_evolution_model(
            stage="s1_creator",
            output_dir=tmp_path / "budget-drift",
            stage_input_path=feedback_path,
            expected_stage_input_file_sha256=feedback_sha,
            semantic_authoring_input_path=semantic_path,
            expected_semantic_authoring_input_file_sha256=_file_sha(semantic_path),
            codex_authoring_input_path=codex_path,
            expected_codex_authoring_input_file_sha256=_file_sha(codex_path),
            parent_bank_path=parent_path,
            expected_parent_bank_file_sha256=parent_sha,
            tool_registry_runtime_sha256="a" * 64,
            codex_executable=_fake_codex_executable(tmp_path),
            timeout_seconds=601,
            process_runner=model,
        )
    assert model.calls == []


def test_s1_v3_discloses_exact_kimi_partial_feedback_and_full_gcs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        parent,
        parent_path,
        parent_sha,
        _old_bundle,
        feedback_path,
        _old_feedback_sha,
        semantic_path,
        codex_path,
    ) = _write_s1_bound_inputs(tmp_path)
    v3_bytes = canonical_json_bytes(
        {
            "schema_version": 3,
            "kind": "portfolio-s1-feedback-bundle",
            "policy_version": "portfolio-s1-feedback-bundle-v3",
        }
    )
    feedback_path.write_bytes(v3_bytes)
    feedback_sha = sha256_bytes(v3_bytes)
    projection = {
        "schema_version": 3,
        "kind": "portfolio-s1-feedback-model-projection",
        "status": "incomplete_diagnostic_feedback",
        "selected_count": 48,
        "attempted_count": 12,
        "parsed_count": 11,
        "parse_error_count": 1,
        "unattempted_count": 36,
        "missing_feedback_count": 37,
        "full_gcs_summary": {"query_count": 800},
    }

    class _LoadedV3:
        schema_version = 3
        selected_count = 48
        attempted_count = 12
        parsed_count = 11
        parse_error_count = 1
        unattempted_count = 36
        missing_feedback_count = 37
        parent_static_bank_sha256 = parent.bank_sha256
        bundle_sha256 = "b" * 64

        def canonical_bytes(self) -> bytes:
            return v3_bytes

        def model_projection_payload(self) -> dict[str, object]:
            return projection

    loaded = _LoadedV3()

    def _load_v3(
        candidate: str | Path,
        *,
        expected_file_sha256: str,
    ) -> _LoadedV3:
        assert Path(candidate) == feedback_path
        assert expected_file_sha256 == feedback_sha
        return loaded

    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback."
        "load_portfolio_s1_feedback_bundle_v3",
        _load_v3,
    )
    model = _MockCodex(AUTHOR_RAW.read_bytes())
    result = run_portfolio_evolution_model(
        stage="s1_creator",
        output_dir=tmp_path / "s1-kimi-v3",
        stage_input_path=feedback_path,
        expected_stage_input_file_sha256=feedback_sha,
        semantic_authoring_input_path=semantic_path,
        expected_semantic_authoring_input_file_sha256=_file_sha(semantic_path),
        codex_authoring_input_path=codex_path,
        expected_codex_authoring_input_file_sha256=_file_sha(codex_path),
        parent_bank_path=parent_path,
        expected_parent_bank_file_sha256=parent_sha,
        tool_registry_runtime_sha256="a" * 64,
        codex_executable=_fake_codex_executable(tmp_path),
        process_runner=model,
    )

    assert len(model.calls) == 1
    assert len(result.candidate_bank.skills) == 6
    prompt = model.calls[0][1]
    request_bytes = prompt.split(b"<portfolio_evolution_request>\n", 1)[1].split(
        b"\n</portfolio_evolution_request>", 1
    )[0]
    request = parse_canonical_json(request_bytes, label="Kimi S1 model request")
    instruction = request["instruction"]
    assert "Kimi partial11/48: 11 of 48 selected rows parsed" in instruction
    assert "12 were attempted, 1 ended in parse error" in instruction
    assert "36 were unattempted" in instruction
    assert "37 therefore have no structured Feedback" in instruction
    assert "full-opt GCS aggregates" in instruction
    assert "Feedback evaluator" in instruction
    assert "Kimi evaluator" not in instruction
    assert "Gemini evaluator" not in instruction
    assert request["model_inputs"]["feedback_bundle"] == projection


def test_s1_v4_uses_complete_qwen_bundle_with_provider_neutral_evidence_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        parent,
        parent_path,
        parent_sha,
        _old_bundle,
        feedback_path,
        _old_feedback_sha,
        semantic_path,
        codex_path,
    ) = _write_s1_bound_inputs(tmp_path)
    v4_bytes = canonical_json_bytes(
        {
            "schema_version": 4,
            "kind": "portfolio-s1-feedback-bundle",
            "policy_version": "portfolio-s1-feedback-bundle-v4",
        }
    )
    feedback_path.write_bytes(v4_bytes)
    feedback_sha = sha256_bytes(v4_bytes)
    projection = {
        "schema_version": 4,
        "kind": "portfolio-s1-feedback-model-projection",
        "status": "complete_diagnostic_feedback",
        "selected_count": 48,
        "attempted_count": 48,
        "parsed_count": 48,
        "parse_error_count": 0,
        "unattempted_count": 0,
        "missing_feedback_count": 0,
        "feedback_entries": [],
    }

    class _LoadedV4:
        schema_version = 4
        selected_count = 48
        parsed_count = 48
        provider_call_count = 48
        parent_static_bank_sha256 = parent.bank_sha256
        bundle_sha256 = "c" * 64

        def canonical_bytes(self) -> bytes:
            return v4_bytes

        def model_projection_payload(self) -> dict[str, object]:
            return projection

    loaded = _LoadedV4()

    def _load_v4(
        candidate: str | Path,
        *,
        expected_file_sha256: str,
    ) -> _LoadedV4:
        assert Path(candidate) == feedback_path
        assert expected_file_sha256 == feedback_sha
        return loaded

    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback."
        "load_portfolio_s1_feedback_bundle_v4",
        _load_v4,
    )
    model = _MockCodex(AUTHOR_RAW.read_bytes())
    run_portfolio_evolution_model(
        stage="s1_creator",
        output_dir=tmp_path / "s1-qwen-v4",
        stage_input_path=feedback_path,
        expected_stage_input_file_sha256=feedback_sha,
        semantic_authoring_input_path=semantic_path,
        expected_semantic_authoring_input_file_sha256=_file_sha(semantic_path),
        codex_authoring_input_path=codex_path,
        expected_codex_authoring_input_file_sha256=_file_sha(codex_path),
        parent_bank_path=parent_path,
        expected_parent_bank_file_sha256=parent_sha,
        tool_registry_runtime_sha256="a" * 64,
        codex_executable=_fake_codex_executable(tmp_path),
        process_runner=model,
    )

    assert len(model.calls) == 1
    prompt = model.calls[0][1]
    request_bytes = prompt.split(b"<portfolio_evolution_request>\n", 1)[1].split(
        b"\n</portfolio_evolution_request>", 1
    )[0]
    request = parse_canonical_json(request_bytes, label="Qwen S1 model request")
    instruction = request["instruction"]
    assert "Feedback evaluator" in instruction
    assert "Kimi evaluator" not in instruction
    assert "Gemini evaluator" not in instruction
    assert "incomplete_diagnostic_feedback" not in instruction
    assert request["model_inputs"]["feedback_bundle"] == projection
    guard = request["constraints"]["author_content_lexical_guard"]
    assert guard["forbidden_whole_words"] == list(
        S1_AUTHOR_CONTENT_FORBIDDEN_WHOLE_WORDS
    )
    assert guard["scope"] == [
        "drafts[].objective",
        "drafts[].steps[].instruction",
        "drafts[].fallback_instruction",
    ]
    assert guard["matching"] == "python_re_ignorecase_exact_pattern"
    assert guard["trusted_validator_regex"] == S1_AUTHOR_CONTENT_FORBIDDEN_REGEX
    assert guard["trusted_validator_regex_sha256"] == sha256_bytes(
        S1_AUTHOR_CONTENT_FORBIDDEN_REGEX.encode("utf-8")
    )
    assert guard["required_detector_prediction_phrase"] == "predicted class name"
    assert (
        "constraints.author_content_lexical_guard.forbidden_whole_words" in instruction
    )
    assert "predicted class name" in instruction


def test_s1_direct_runner_rejects_legacy_style_2_1_before_model_call(
    tmp_path: Path,
) -> None:
    (
        _parent,
        parent_path,
        parent_sha,
        _bundle,
        feedback_path,
        feedback_sha,
        _current_semantic_path,
        _current_codex_path,
    ) = _write_s1_bound_inputs(tmp_path)
    model = _MockCodex(AUTHOR_RAW.read_bytes())
    output_dir = tmp_path / "legacy-style-must-not-run"

    with pytest.raises(PortfolioEvolutionModelError, match="Style-2.3"):
        run_portfolio_evolution_model(
            stage="s1_creator",
            output_dir=output_dir,
            stage_input_path=feedback_path,
            expected_stage_input_file_sha256=feedback_sha,
            semantic_authoring_input_path=SEMANTIC_INPUT,
            expected_semantic_authoring_input_file_sha256=_file_sha(SEMANTIC_INPUT),
            codex_authoring_input_path=CODEX_INPUT,
            expected_codex_authoring_input_file_sha256=_file_sha(CODEX_INPUT),
            parent_bank_path=parent_path,
            expected_parent_bank_file_sha256=parent_sha,
            tool_registry_runtime_sha256="a" * 64,
            codex_executable=_fake_codex_executable(tmp_path),
            process_runner=model,
        )

    assert model.calls == []
    assert not output_dir.exists()


def test_s1_final_larger_than_64k_consumes_one_rejected_attempt(
    tmp_path: Path,
) -> None:
    (
        parent,
        parent_path,
        parent_sha,
        bundle,
        feedback_path,
        feedback_sha,
        semantic_path,
        codex_path,
    ) = _write_s1_bound_inputs(tmp_path)
    model = _MockCodex(b"x" * 65537)
    output_dir = tmp_path / "oversized-final"
    with pytest.raises(PortfolioEvolutionModelError, match="consumed and rejected"):
        run_portfolio_evolution_model(
            stage="s1_creator",
            output_dir=output_dir,
            stage_input_path=feedback_path,
            expected_stage_input_file_sha256=feedback_sha,
            semantic_authoring_input_path=semantic_path,
            expected_semantic_authoring_input_file_sha256=_file_sha(semantic_path),
            codex_authoring_input_path=codex_path,
            expected_codex_authoring_input_file_sha256=_file_sha(codex_path),
            parent_bank_path=parent_path,
            expected_parent_bank_file_sha256=parent_sha,
            tool_registry_runtime_sha256="a" * 64,
            codex_executable=_fake_codex_executable(tmp_path),
            process_runner=model,
        )
    assert len(model.calls) == 1
    receipt = _load_receipt(output_dir / "invocation-receipt.json")
    assert receipt.status == "rejected"
    assert receipt.parent_bank_sha256 == parent.bank_sha256
    assert receipt.s1_feedback_bundle_sha256 == bundle.bundle_sha256
    assert receipt.max_final_output_bytes == 65536


def test_one_mock_session_compiles_s1_s2_and_s3_with_exact_field_scopes(
    tmp_path: Path,
) -> None:
    executable = _fake_codex_executable(tmp_path)
    runtime_sha256 = "a" * 64
    (
        parent_bank,
        parent_path,
        parent_sha,
        feedback_bundle,
        feedback_path,
        feedback_sha,
        semantic_path,
        codex_path,
    ) = _write_s1_bound_inputs(tmp_path, runtime_sha256=runtime_sha256)
    author = _MockCodex(AUTHOR_RAW.read_bytes())
    s1 = run_portfolio_evolution_model(
        stage="s1_creator",
        output_dir=tmp_path / "s1",
        stage_input_path=feedback_path,
        expected_stage_input_file_sha256=feedback_sha,
        semantic_authoring_input_path=semantic_path,
        expected_semantic_authoring_input_file_sha256=_file_sha(semantic_path),
        codex_authoring_input_path=codex_path,
        expected_codex_authoring_input_file_sha256=_file_sha(codex_path),
        parent_bank_path=parent_path,
        expected_parent_bank_file_sha256=parent_sha,
        tool_registry_runtime_sha256=runtime_sha256,
        codex_executable=executable,
        process_runner=author,
    )
    assert len(author.calls) == 1
    assert len(s1.candidate_bank.skills) == 6
    assert s1.normalized_draft_path is not None
    normalized_raw = parse_canonical_json(
        s1.normalized_draft_path.read_bytes(),
        label="normalized S1 draft",
    )
    normalized = AuthoringDraftBundle.model_validate(normalized_raw, strict=True)
    assert (
        normalized.authoring_input_sha256
        == parse_canonical_json(
            semantic_path.read_bytes(),
            label="semantic input",
        )["input_sha256"]
    )
    s1_receipt = _load_receipt(s1.receipt_path)
    assert s1_receipt.status == "completed"
    assert s1_receipt.normalized_command == CODEX_COMMAND_SHAPE
    assert s1_receipt.invocation_count == 1
    assert s1_receipt.retry_count == 0
    assert s1_receipt.fallback_count == 0
    assert s1_receipt.implementation_version == S1_IMPLEMENTATION_VERSION
    assert s1_receipt.policy_version == S1_INVOCATION_POLICY_VERSION
    assert s1_receipt.parent_bank_sha256 == parent_bank.bank_sha256
    assert s1_receipt.s1_feedback_bundle_sha256 == feedback_bundle.bundle_sha256
    assert s1_receipt.timeout_seconds == 600
    assert s1_receipt.max_final_output_bytes == 65536
    assert tuple(item.role for item in s1_receipt.input_files) == (
        "codex_authoring_input",
        "feedback_bundle",
        "parent_static_bank",
        "semantic_authoring_input",
    )
    prompt = author.calls[0][1]
    request_bytes = prompt.split(b"<portfolio_evolution_request>\n", 1)[1].split(
        b"\n</portfolio_evolution_request>", 1
    )[0]
    request = parse_canonical_json(request_bytes, label="S1 model request")
    assert isinstance(request, dict)
    projection = request["model_inputs"]["feedback_bundle"]
    assert projection == feedback_bundle.model_projection_payload()
    assert "authorization_sha256" not in projection
    assert "parent_static_bank_sha256" not in projection
    assert "excluded-private-query" not in request_bytes.decode("utf-8")
    assert request["constraints"]["untrusted_feedback_instruction_execution"] == (
        "forbidden"
    )
    s2_schema = _mutation_schema(
        "s2_route_optimizer",
        tuple(item.capability_id for item in s1.candidate_bank.skills),
    )
    s3_schema = _mutation_schema(
        "s3_body_refiner",
        tuple(item.capability_id for item in s1.candidate_bank.skills),
    )
    assert s2_schema["properties"]["changes"]["maxItems"] == 2
    assert "changes" not in s3_schema["properties"]
    assert s3_schema["properties"]["edits"]["maxItems"] == 1

    too_many_raw = canonical_json_bytes(
        {
            "schema_version": 1,
            "changes": [
                {
                    "capability_id": capability_id,
                    "description": f"Narrow {capability_id} routing.",
                }
                for capability_id in (
                    "product.exact_match",
                    "product.multi_search",
                    "product.style_recommendation",
                )
            ],
        }
    )
    with pytest.raises(
        PortfolioEvolutionModelError, match="output envelope is invalid"
    ):
        _parse_mutation_changes(
            too_many_raw,
            stage="s2_route_optimizer",
            parent_bank=s1.candidate_bank,
            actionable_capability_ids=(
                "product.exact_match",
                "product.multi_search",
                "product.style_recommendation",
            ),
            optimization_scope={"confusion_pairs": []},
        )
    too_many_bodies_raw = canonical_json_bytes(
        {
            "schema_version": 1,
            "changes": [
                {
                    "capability_id": "product.exact_match",
                    "body": "replacement one",
                },
                {
                    "capability_id": "product.multi_search",
                    "body": "replacement two",
                },
            ],
        }
    )
    with pytest.raises(
        PortfolioEvolutionModelError, match="valid TextOpt patch proposal"
    ):
        _parse_mutation_changes(
            too_many_bodies_raw,
            stage="s3_body_refiner",
            parent_bank=s1.candidate_bank,
            actionable_capability_ids=(
                "product.exact_match",
                "product.multi_search",
            ),
            optimization_scope={"failure_clusters": []},
        )
    unrelated_raw = canonical_json_bytes(
        {
            "schema_version": 1,
            "changes": [
                {
                    "capability_id": "product.exact_match",
                    "description": "Narrow exact-match routing.",
                },
                {
                    "capability_id": "product.style_recommendation",
                    "description": "Narrow style routing.",
                },
            ],
        }
    )
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="one actionable confusion pair",
    ):
        _parse_mutation_changes(
            unrelated_raw,
            stage="s2_route_optimizer",
            parent_bank=s1.candidate_bank,
            actionable_capability_ids=(
                "knowledge.visual_encyclopedia",
                "product.exact_match",
                "product.multi_search",
                "product.style_recommendation",
            ),
            optimization_scope={
                "confusion_pairs": [
                    {
                        "expected_capability": "product.exact_match",
                        "observed_capability": "product.multi_search",
                        "actionable": True,
                    },
                    {
                        "expected_capability": "product.style_recommendation",
                        "observed_capability": "knowledge.visual_encyclopedia",
                        "actionable": True,
                    },
                ]
            },
        )

    route_attribution = tmp_path / "route-attribution.json"
    route_sha = _write_parent_attribution(
        route_attribution,
        stage="s2_route_optimizer",
        source_bank_sha256=s1.candidate_bank.bank_sha256,
    )
    route_source = next(
        item
        for item in s1.candidate_bank.skills
        if item.capability_id == "product.exact_match"
    )
    replacement_description = (
        f"Route to {route_source.capability_id} only when its exact "
        "evidence requirements are present."
    )
    route_raw = canonical_json_bytes(
        {
            "schema_version": 1,
            "changes": [
                {
                    "capability_id": route_source.capability_id,
                    "description": replacement_description,
                }
            ],
        }
    )
    session_scratch = tmp_path / "s2-s3-session-scratch"
    session_scratch.mkdir()
    route_model = _MockCodex(
        route_raw,
        session_mode="new_persistent",
        thread_id="portfolio-s2-s3-session",
    )
    s2 = run_portfolio_evolution_model(
        stage="s2_route_optimizer",
        output_dir=tmp_path / "s2",
        stage_input_path=route_attribution,
        expected_stage_input_file_sha256=route_sha,
        parent_bank_path=s1.candidate_bank_path,
        expected_parent_bank_file_sha256=_file_sha(s1.candidate_bank_path),
        codex_executable=executable,
        session_mode="new_persistent",
        session_scratch=session_scratch,
        process_runner=route_model,
    )
    assert len(route_model.calls) == 1
    route_prompt = route_model.calls[0][1]
    assert b"portfolio-s2-confusion-pair-scope-v2" in route_prompt
    assert b'"count":2' in route_prompt
    assert s2.mutation_path is not None
    s1_by_capability = {item.capability_id: item for item in s1.candidate_bank.skills}
    s2_by_capability = {item.capability_id: item for item in s2.candidate_bank.skills}
    s2_changed = s2_by_capability[route_source.capability_id]
    assert s2_changed.description == replacement_description
    assert s2_changed.body == s1_by_capability[route_source.capability_id].body
    assert s2_changed.parent_skill_sha256 == route_source.skill_sha256
    s2_receipt = _load_receipt(s2.receipt_path)
    assert s2_receipt.normalized_command == PERSISTENT_NEW_COMMAND_SHAPE
    assert s2_receipt.session_mode == "new_persistent"
    assert s2_receipt.thread_id == "portfolio-s2-s3-session"
    gate = build_portfolio_stage_gate_report(
        config="s1s2",
        parent_bank_sha256=s1.candidate_bank.bank_sha256,
        candidate_bank_sha256=s2.candidate_bank.bank_sha256,
        adherence_contract_bank_sha256=s1.candidate_bank.bank_sha256,
        evaluation_query_ids=PORTFOLIO_OPTIMIZATION_QUERY_IDS,
        rubric_file_sha256=PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
        paired_shared_route_artifact_sha256s=None,
        parent_route_accuracy=0.0,
        candidate_route_accuracy=1.0,
        parent_mean_j=50.0,
        candidate_mean_j=50.0,
        parent_mean_skill_adherence=1.0,
        candidate_mean_skill_adherence=1.0,
        parent_hard_error_count=0,
        candidate_hard_error_count=0,
        decision="accepted",
        parent_result_file="parent.json",
        parent_result_file_sha256="1" * 64,
        candidate_result_file="candidate.json",
        candidate_result_file_sha256="2" * 64,
    )
    gate_path = tmp_path / "s2-gate.json"
    gate_path.write_bytes(gate.canonical_bytes())

    body_attribution = tmp_path / "body-attribution.json"
    body_sha = _write_parent_attribution(
        body_attribution,
        stage="s3_body_refiner",
        source_bank_sha256=s2.candidate_bank.bank_sha256,
    )
    body_packet = _read_parent_attribution(body_attribution)
    body_capabilities, body_scope = _actionable_mutation_scope(
        "s3_body_refiner",
        body_packet,
        parent_bank=s2.candidate_bank,
    )
    assert body_capabilities == (s2_changed.capability_id,)
    failure_cluster = body_scope["failure_clusters"][0]
    rule_id, original_rule_line = _first_editable_rule(s2_changed.body)
    replacement_statement = (
        "State the strongest supported match first, then the relevant limitation."
    )
    provenance = original_rule_line[original_rule_line.rindex(" (source:") :]
    replacement_rule_line = f"- [{rule_id}] {replacement_statement}{provenance}"
    expected_replacement_body = s2_changed.body.replace(
        original_rule_line,
        replacement_rule_line,
        1,
    )
    drifted_contract_body = s2_changed.body.replace(
        "- card_requirement: required",
        "- card_requirement: forbidden",
        1,
    )
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="valid TextOpt patch proposal",
    ):
        _parse_mutation_changes(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "changes": [
                        {
                            "capability_id": s2_changed.capability_id,
                            "body": drifted_contract_body,
                        }
                    ],
                }
            ),
            stage="s3_body_refiner",
            parent_bank=s2.candidate_bank,
            actionable_capability_ids=(s2_changed.capability_id,),
            optimization_scope=body_scope,
        )
    body_raw = canonical_json_bytes(
        {
            "schema_version": 1,
            "artifact_kind": "portfolio-s3-text-patch-proposal",
            "policy_version": PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION,
            "capability_id": s2_changed.capability_id,
            "failure_cluster_id": failure_cluster["failure_cluster_id"],
            "failure_cluster_sha256": failure_cluster["cluster_sha256"],
            "edits": [
                {
                    "op": "replace_rule",
                    "rule_id": rule_id,
                    "expected_text_sha256": sha256_bytes(
                        original_rule_line.encode("utf-8")
                    ),
                    "replacement": replacement_statement,
                    "evidence_ids": failure_cluster["evidence_ids"],
                    "support_count": failure_cluster["support_count"],
                    "addressed_dimensions": failure_cluster["addressed_dimensions"],
                    "rationale": "Repeated TCR misses need a supported-first rule.",
                }
            ],
            "reconsideration_rationale": "",
        }
    )
    rejected_buffer = build_empty_portfolio_s3_rejected_edit_buffer()
    rejected_buffer_path = tmp_path / "rejected-edit-buffer.json"
    rejected_buffer_path.write_bytes(rejected_buffer.canonical_bytes())
    body_model = _MockCodex(
        body_raw,
        session_mode="ephemeral",
        thread_id="portfolio-s3-clean-session",
    )
    s3 = run_portfolio_evolution_model(
        stage="s3_body_refiner",
        output_dir=tmp_path / "s3",
        stage_input_path=body_attribution,
        expected_stage_input_file_sha256=body_sha,
        parent_bank_path=s2.candidate_bank_path,
        expected_parent_bank_file_sha256=_file_sha(s2.candidate_bank_path),
        codex_executable=executable,
        prior_stage_gate=gate_path,
        expected_prior_stage_gate_file_sha256=_file_sha(gate_path),
        rejected_edit_buffer_path=rejected_buffer_path,
        expected_rejected_edit_buffer_file_sha256=_file_sha(rejected_buffer_path),
        process_runner=body_model,
    )
    assert len(body_model.calls) == 1
    body_prompt = body_model.calls[0][1]
    assert b"portfolio-s3-textopt-skill-defect-scope-v2" in body_prompt
    assert b'"optimization_mode":"diagnostic_l1"' in body_prompt
    assert b'"recurring_low_dimensions":["TCR"]' in body_prompt
    assert b'"editable_rule_catalog"' in body_prompt
    assert original_rule_line.encode("utf-8") in body_prompt
    assert (
        sha256_bytes(original_rule_line.encode("utf-8")).encode("ascii") in body_prompt
    )
    s3_by_capability = {item.capability_id: item for item in s3.candidate_bank.skills}
    s3_changed = s3_by_capability[s2_changed.capability_id]
    assert s3_changed.description == s2_changed.description
    assert s3_changed.body == expected_replacement_body
    assert s3_changed.parent_skill_sha256 == s2_changed.skill_sha256
    persisted = StaticBankArtifact.model_validate(
        json.loads(s3.candidate_bank_path.read_text(encoding="utf-8")),
        strict=True,
    )
    assert persisted == s3.candidate_bank
    assert s3.patch_path is not None
    patch_raw = parse_canonical_json(
        s3.patch_path.read_bytes(),
        label="normalized S3 TextOpt patch",
    )
    patch = PortfolioS3TextPatchArtifact.model_validate(patch_raw, strict=True)
    assert patch.canonical_bytes() == s3.patch_path.read_bytes()
    assert patch.capability_id == s2_changed.capability_id
    assert patch.edits[0].rule_id == rule_id
    assert s3.mutation_path is not None
    mutation_raw = parse_canonical_json(
        s3.mutation_path.read_bytes(),
        label="S3 stage mutation",
    )
    assert {item["artifact_kind"] for item in mutation_raw["input_artifacts"]} == {
        "body_attribution",
        "rejected_edit_buffer",
    }
    s3_receipt = _load_receipt(s3.receipt_path)
    assert s3_receipt.status == "completed"
    assert s3_receipt.implementation_version == S3_IMPLEMENTATION_VERSION
    assert s3_receipt.policy_version == S3_INVOCATION_POLICY_VERSION
    assert s3_receipt.s3_textopt_compiler_file_sha256 == _file_sha(
        S3_TEXTOPT_COMPILER_PATH
    )
    assert (
        tmp_path / "s3" / S3_TEXTOPT_COMPILER_SNAPSHOT_FILE
    ).read_bytes() == S3_TEXTOPT_COMPILER_PATH.read_bytes()
    assert s3_receipt.normalized_command == CODEX_COMMAND_SHAPE
    assert s3_receipt.session_mode == "ephemeral"
    assert s3_receipt.session_turn_index == 1
    assert s3_receipt.followup_count == 0
    assert s3_receipt.prior_invocation_receipt_file_sha256 is None
    assert s3_receipt.prior_gate_report_file_sha256 == _file_sha(gate_path)
    assert s3_receipt.thread_id == "portfolio-s3-clean-session"
    assert s3_receipt.thread_id != s2_receipt.thread_id
    assert tuple(item.role for item in s3_receipt.input_files) == (
        "parent_bank",
        "prior_stage_gate",
        "rejected_edit_buffer",
        "stage_input",
    )

    no_buffer_model = _MockCodex(
        body_raw,
        session_mode="ephemeral",
        thread_id="portfolio-s3-empty-buffer-session",
    )
    no_buffer_s3 = run_portfolio_evolution_model(
        stage="s3_body_refiner",
        output_dir=tmp_path / "s3-empty-buffer",
        stage_input_path=body_attribution,
        expected_stage_input_file_sha256=body_sha,
        parent_bank_path=s2.candidate_bank_path,
        expected_parent_bank_file_sha256=_file_sha(s2.candidate_bank_path),
        codex_executable=executable,
        prior_stage_gate=gate_path,
        expected_prior_stage_gate_file_sha256=_file_sha(gate_path),
        process_runner=no_buffer_model,
    )
    assert no_buffer_s3.patch_path is not None
    no_buffer_receipt = _load_receipt(no_buffer_s3.receipt_path)
    assert tuple(item.role for item in no_buffer_receipt.input_files) == (
        "parent_bank",
        "prior_stage_gate",
        "stage_input",
    )
    assert no_buffer_s3.mutation_path is not None
    no_buffer_mutation = parse_canonical_json(
        no_buffer_s3.mutation_path.read_bytes(),
        label="S3 empty-buffer stage mutation",
    )
    assert [
        item["artifact_kind"] for item in no_buffer_mutation["input_artifacts"]
    ] == ["body_attribution"]

    legacy_receipt_raw = json.loads(s3.receipt_path.read_text(encoding="utf-8"))
    legacy_receipt_raw["implementation_version"] = LEGACY_S3_IMPLEMENTATION_VERSION
    legacy_receipt_raw["policy_version"] = LEGACY_S3_INVOCATION_POLICY_VERSION
    legacy_receipt_raw.pop("s3_textopt_compiler_file_sha256")
    legacy_receipt_raw["input_files"] = [
        item
        for item in legacy_receipt_raw["input_files"]
        if item["role"] != "rejected_edit_buffer"
    ]
    legacy_receipt_raw["output_files"] = [
        item
        for item in legacy_receipt_raw["output_files"]
        if item["file"] not in {"s3-text-patch.json", S3_TEXTOPT_COMPILER_SNAPSHOT_FILE}
    ]
    legacy_receipt_raw["receipt_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in legacy_receipt_raw.items()
                if key != "receipt_sha256"
            }
        )
    )
    legacy_receipt = EvolutionInvocationReceipt.model_validate(
        legacy_receipt_raw,
        strict=True,
    )
    assert legacy_receipt.implementation_version == "1.4.0"

    missing_patch_receipt = json.loads(s3.receipt_path.read_text(encoding="utf-8"))
    missing_patch_receipt["output_files"] = [
        item
        for item in missing_patch_receipt["output_files"]
        if item["file"] != "s3-text-patch.json"
    ]
    missing_patch_receipt["receipt_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in missing_patch_receipt.items()
                if key != "receipt_sha256"
            }
        )
    )
    with pytest.raises(ValueError, match="lacks its normalized patch"):
        EvolutionInvocationReceipt.model_validate(missing_patch_receipt, strict=True)

    missing_compiler_receipt = json.loads(s3.receipt_path.read_text(encoding="utf-8"))
    missing_compiler_receipt["output_files"] = [
        item
        for item in missing_compiler_receipt["output_files"]
        if item["file"] != S3_TEXTOPT_COMPILER_SNAPSHOT_FILE
    ]
    missing_compiler_receipt["receipt_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in missing_compiler_receipt.items()
                if key != "receipt_sha256"
            }
        )
    )
    with pytest.raises(ValueError, match="compiler snapshot"):
        EvolutionInvocationReceipt.model_validate(
            missing_compiler_receipt,
            strict=True,
        )

    missing_compiler_binding = json.loads(s3.receipt_path.read_text(encoding="utf-8"))
    missing_compiler_binding.pop("s3_textopt_compiler_file_sha256")
    missing_compiler_binding["receipt_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in missing_compiler_binding.items()
                if key != "receipt_sha256"
            }
        )
    )
    with pytest.raises(ValueError, match="compiler binding"):
        EvolutionInvocationReceipt.model_validate(
            missing_compiler_binding,
            strict=True,
        )

    receipt_raw = json.loads(s3.receipt_path.read_text(encoding="utf-8"))
    receipt_raw["input_files"].append(
        {
            "role": "prior_invocation_receipt",
            "path": str(s2.receipt_path),
            "file_sha256": _file_sha(s2.receipt_path),
            "content_sha256": _file_sha(s2.receipt_path),
        }
    )
    receipt_raw["input_files"].sort(key=lambda item: item["role"])
    receipt_raw["receipt_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in receipt_raw.items()
                if key != "receipt_sha256"
            }
        )
    )
    with pytest.raises(ValueError, match="S3 v1.5 input lineage roles drifted"):
        EvolutionInvocationReceipt.model_validate(receipt_raw, strict=True)

    gate_tamper = json.loads(s3.receipt_path.read_text(encoding="utf-8"))
    gate_binding = next(
        item
        for item in gate_tamper["input_files"]
        if item["role"] == "prior_stage_gate"
    )
    gate_binding["file_sha256"] = "e" * 64
    gate_binding["content_sha256"] = "e" * 64
    gate_tamper["receipt_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in gate_tamper.items()
                if key != "receipt_sha256"
            }
        )
    )
    with pytest.raises(ValueError, match="S3 v1.5 prior gate binding drifted"):
        EvolutionInvocationReceipt.model_validate(gate_tamper, strict=True)

    rejected_resume_model = _MockCodex(
        body_raw,
        session_mode="resume",
        thread_id="portfolio-s2-s3-session",
    )
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="must use a new ephemeral clean turn",
    ):
        run_portfolio_evolution_model(
            stage="s3_body_refiner",
            output_dir=tmp_path / "s3-resume-rejected",
            stage_input_path=body_attribution,
            expected_stage_input_file_sha256=body_sha,
            parent_bank_path=s2.candidate_bank_path,
            expected_parent_bank_file_sha256=_file_sha(s2.candidate_bank_path),
            codex_executable=executable,
            session_mode="resume",
            session_scratch=session_scratch,
            resume_thread_id="portfolio-s2-s3-session",
            session_turn_index=2,
            resume_from_invocation_receipt=s2.receipt_path,
            expected_resume_invocation_receipt_file_sha256=_file_sha(s2.receipt_path),
            prior_stage_gate=gate_path,
            expected_prior_stage_gate_file_sha256=_file_sha(gate_path),
            process_runner=rejected_resume_model,
        )
    assert rejected_resume_model.calls == []

    lapse_attribution = tmp_path / "body-execution-lapse.json"
    lapse_sha = _write_parent_attribution(
        lapse_attribution,
        stage="s3_body_refiner",
        source_bank_sha256=s2.candidate_bank.bank_sha256,
        low_tier_query_ids=(),
        hard_error_query_ids=("dm-001",),
    )
    lapse_model = _MockCodex(body_raw)
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="execution lapses cannot mutate Skill Body",
    ):
        run_portfolio_evolution_model(
            stage="s3_body_refiner",
            output_dir=tmp_path / "s3-execution-lapse-rejected",
            stage_input_path=lapse_attribution,
            expected_stage_input_file_sha256=lapse_sha,
            parent_bank_path=s2.candidate_bank_path,
            expected_parent_bank_file_sha256=_file_sha(s2.candidate_bank_path),
            codex_executable=executable,
            prior_stage_gate=gate_path,
            expected_prior_stage_gate_file_sha256=_file_sha(gate_path),
            process_runner=lapse_model,
        )
    assert lapse_model.calls == []

    missing_candidate_attribution = tmp_path / "body-missing-candidate.json"
    missing_candidate_sha = _write_parent_attribution(
        missing_candidate_attribution,
        stage="s3_body_refiner",
        source_bank_sha256=s2.candidate_bank.bank_sha256,
        include_visible_cards=False,
    )
    missing_candidate_model = _MockCodex(body_raw)
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="execution lapses cannot mutate Skill Body",
    ):
        run_portfolio_evolution_model(
            stage="s3_body_refiner",
            output_dir=tmp_path / "s3-missing-candidate-rejected",
            stage_input_path=missing_candidate_attribution,
            expected_stage_input_file_sha256=missing_candidate_sha,
            parent_bank_path=s2.candidate_bank_path,
            expected_parent_bank_file_sha256=_file_sha(s2.candidate_bank_path),
            codex_executable=executable,
            prior_stage_gate=gate_path,
            expected_prior_stage_gate_file_sha256=_file_sha(gate_path),
            process_runner=missing_candidate_model,
        )
    assert missing_candidate_model.calls == []

    missing_evidence_attribution = tmp_path / "body-missing-evidence.json"
    missing_evidence_sha = _write_parent_attribution(
        missing_evidence_attribution,
        stage="s3_body_refiner",
        source_bank_sha256=s2.candidate_bank.bank_sha256,
        include_visible_evidence=False,
    )
    missing_evidence_model = _MockCodex(body_raw)
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="execution lapses cannot mutate Skill Body",
    ):
        run_portfolio_evolution_model(
            stage="s3_body_refiner",
            output_dir=tmp_path / "s3-missing-evidence-rejected",
            stage_input_path=missing_evidence_attribution,
            expected_stage_input_file_sha256=missing_evidence_sha,
            parent_bank_path=s2.candidate_bank_path,
            expected_parent_bank_file_sha256=_file_sha(s2.candidate_bank_path),
            codex_executable=executable,
            prior_stage_gate=gate_path,
            expected_prior_stage_gate_file_sha256=_file_sha(gate_path),
            process_runner=missing_evidence_model,
        )
    assert missing_evidence_model.calls == []

    wrong_gate = build_portfolio_stage_gate_report(
        config="s1s2",
        parent_bank_sha256=s1.candidate_bank.bank_sha256,
        candidate_bank_sha256="f" * 64,
        adherence_contract_bank_sha256=s1.candidate_bank.bank_sha256,
        evaluation_query_ids=PORTFOLIO_OPTIMIZATION_QUERY_IDS,
        rubric_file_sha256=PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
        paired_shared_route_artifact_sha256s=None,
        parent_route_accuracy=0.0,
        candidate_route_accuracy=1.0,
        parent_mean_j=50.0,
        candidate_mean_j=50.0,
        parent_mean_skill_adherence=1.0,
        candidate_mean_skill_adherence=1.0,
        parent_hard_error_count=0,
        candidate_hard_error_count=0,
        decision="accepted",
        parent_result_file="parent.json",
        parent_result_file_sha256="1" * 64,
        candidate_result_file="candidate.json",
        candidate_result_file_sha256="2" * 64,
    )
    wrong_gate_path = tmp_path / "wrong-s2-gate.json"
    wrong_gate_path.write_bytes(wrong_gate.canonical_bytes())
    wrong_parent_model = _MockCodex(body_raw)
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="parent is not the Bank selected",
    ):
        run_portfolio_evolution_model(
            stage="s3_body_refiner",
            output_dir=tmp_path / "s3-wrong-parent-rejected",
            stage_input_path=body_attribution,
            expected_stage_input_file_sha256=body_sha,
            parent_bank_path=s2.candidate_bank_path,
            expected_parent_bank_file_sha256=_file_sha(s2.candidate_bank_path),
            codex_executable=executable,
            prior_stage_gate=wrong_gate_path,
            expected_prior_stage_gate_file_sha256=_file_sha(wrong_gate_path),
            process_runner=wrong_parent_model,
        )
    assert wrong_parent_model.calls == []


def test_tool_activity_consumes_the_create_only_attempt_without_retry(
    tmp_path: Path,
) -> None:
    executable = _fake_codex_executable(tmp_path)
    parent_source = (
        ROOT
        / "runs"
        / "formal-authoring"
        / "llm-static-codex-primary-20260724-high-v5"
        / "pre-review-draft.json"
    )
    # Use the first test's compiler-independent real Bank fixture.
    from skillchain.evaluation.portfolio_treatments import (
        compile_verified_codex_llm_static_bank,
        load_verified_codex_draft_rebind,
    )

    rebind = load_verified_codex_draft_rebind(
        codex_input_path=CODEX_INPUT,
        expected_codex_input_file_sha256=_file_sha(CODEX_INPUT),
        semantic_input_path=SEMANTIC_INPUT,
        expected_semantic_input_file_sha256=_file_sha(SEMANTIC_INPUT),
        draft_path=parent_source,
        expected_draft_file_sha256=_file_sha(parent_source),
    )
    bank = compile_verified_codex_llm_static_bank(
        rebind,
        tool_registry_runtime_sha256="a" * 64,
    )
    bank_path = tmp_path / "parent-bank.json"
    bank_sha = _write_canonical(
        bank_path,
        bank.model_dump(mode="json"),
    )
    attribution_path = tmp_path / "route-attribution.json"
    attribution_sha = _write_parent_attribution(
        attribution_path,
        stage="s2_route_optimizer",
        source_bank_sha256=bank.bank_sha256,
    )
    raw_output = canonical_json_bytes(
        {
            "schema_version": 1,
            "changes": [
                {
                    "capability_id": bank.skills[0].capability_id,
                    "description": "A real replacement description.",
                }
            ],
        }
    )
    tool_events = b"".join(
        canonical_json_bytes(item)
        for item in (
            {"type": "thread.started", "thread_id": "tool-thread"},
            {"type": "turn.started", "thread_id": "tool-thread"},
            {
                "type": "item.completed",
                "thread_id": "tool-thread",
                "item": {
                    "id": "tool-1",
                    "type": "command_execution",
                    "command": "dir",
                },
            },
            {
                "type": "turn.completed",
                "thread_id": "tool-thread",
                "usage": {
                    "input_tokens": 1,
                    "cached_input_tokens": 0,
                    "output_tokens": 1,
                },
            },
        )
    )
    model = _MockCodex(raw_output, events=tool_events)
    output_dir = tmp_path / "rejected"
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="consumed and rejected",
    ):
        run_portfolio_evolution_model(
            stage="s2_route_optimizer",
            output_dir=output_dir,
            stage_input_path=attribution_path,
            expected_stage_input_file_sha256=attribution_sha,
            parent_bank_path=bank_path,
            expected_parent_bank_file_sha256=bank_sha,
            codex_executable=executable,
            process_runner=model,
        )
    assert len(model.calls) == 1
    receipt = _load_receipt(output_dir / "invocation-receipt.json")
    assert receipt.status == "rejected"
    assert receipt.invocation_count == 1
    assert receipt.retry_count == 0
    assert receipt.fallback_count == 0
    assert receipt.visible_tool_activity is True
    with pytest.raises(FileExistsError, match="create-only"):
        run_portfolio_evolution_model(
            stage="s2_route_optimizer",
            output_dir=output_dir,
            stage_input_path=attribution_path,
            expected_stage_input_file_sha256=attribution_sha,
            parent_bank_path=bank_path,
            expected_parent_bank_file_sha256=bank_sha,
            codex_executable=executable,
            process_runner=model,
        )
    assert len(model.calls) == 1


def test_external_sha_mismatch_fails_before_claim_or_model_call(
    tmp_path: Path,
) -> None:
    stage_input = tmp_path / "attribution.json"
    _write_canonical(stage_input, {"schema_version": 1, "records": []})
    model = _MockCodex(b"{}")
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="external SHA-256 mismatch",
    ):
        run_portfolio_evolution_model(
            stage="s2_route_optimizer",
            output_dir=output_dir,
            stage_input_path=stage_input,
            expected_stage_input_file_sha256="0" * 64,
            parent_bank_path=stage_input,
            expected_parent_bank_file_sha256=_file_sha(stage_input),
            codex_executable=_fake_codex_executable(tmp_path),
            process_runner=model,
        )
    assert model.calls == []
    assert not output_dir.exists()


def test_s3_rejected_buffer_path_and_sha_are_strictly_paired(
    tmp_path: Path,
) -> None:
    model = _MockCodex(b"{}")
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(
        PortfolioEvolutionModelError,
        match="path and expected SHA-256 must be supplied together",
    ):
        run_portfolio_evolution_model(
            stage="s3_body_refiner",
            output_dir=output_dir,
            stage_input_path=tmp_path / "unused-attribution.json",
            expected_stage_input_file_sha256="0" * 64,
            rejected_edit_buffer_path=tmp_path / "buffer.json",
            process_runner=model,
        )
    assert model.calls == []
    assert not output_dir.exists()
