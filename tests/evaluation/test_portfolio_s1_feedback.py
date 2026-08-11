from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillchain import config
from skillchain.evaluation.evaluator_outputs import (
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2,
    VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2,
    VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
    VisualFeedbackOutput,
)
from skillchain.evaluation.feedback_runtime import (
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V2,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2,
    VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
    _make_result,
)
from skillchain.evaluation.packets import (
    EvaluationImage,
    FeedbackPacket,
    RubricSnapshot,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4,
    VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
)
from skillchain.evaluation.portfolio_gcs import GCS_CAPABILITY_ORDER
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackAuthorizationV1,
    PortfolioS1FeedbackAuthorizationV2,
    PortfolioS1FeedbackControlV1,
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackFullGCSBindingV2,
    PortfolioS1FeedbackFullGCSSummaryV2,
    PortfolioS1FeedbackGCSCapabilitySummaryV2,
    PortfolioS1FeedbackGCSStrataSummaryV2,
    build_bound_feedback_artifact,
    build_feedback_call_reservation,
    build_portfolio_s1_feedback_bundle,
    build_portfolio_s1_feedback_bundle_v3,
    build_portfolio_s1_feedback_control,
    build_portfolio_s1_feedback_run,
    build_portfolio_s1_feedback_selection,
    build_selected_feedback_authorization,
    build_verified_static_feedback_sources,
    load_portfolio_s1_feedback_bundle,
    load_portfolio_s1_feedback_bundle_v3,
    load_portfolio_s1_feedback_control,
    resume_bound_feedback_artifact,
    validate_selected_feedback_authorization,
    validate_portfolio_s1_feedback_control,
    write_bound_feedback_artifact,
    write_portfolio_s1_feedback_bundle,
    write_portfolio_s1_feedback_bundle_v3,
    SelectedFeedbackAssetV1,
)
from skillchain.evaluation.portfolio_s1_feedback_remote import (
    PortfolioS1FeedbackRemoteRuntimeReceiptV1,
    SelectedFeedbackRemoteBindingV1,
    prepare_selected_kimi_feedback_remote_runtime,
)
from skillchain.evaluation.visual_runtime import (
    EvaluatorImageLoadError,
    load_verified_evaluator_image,
)
from skillchain.llm import LLMUsage
from skillchain.schemas import ConversationTurn
from skillchain.synthesis.portfolio_core_selection import CreatorSelectionEntry
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes
from scripts.run_portfolio_s1_feedback import (
    PortfolioS1FeedbackRunError,
    PreparedPortfolioS1FeedbackRun,
    _budget_snapshot,
    _exclusive_execute_writer_lock,
    _require_frozen_qwen_endpoint,
    _require_qwen_execute_environment,
    _validate_input_reservation_bounds,
    execute_run,
    prepare_run,
)


def _hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _rubric() -> RubricSnapshot:
    content = "Diagnose the public answer and evidence for Skill improvement."
    return RubricSnapshot(
        rubric_id="portfolio-s1-feedback-v1",
        rubric_version="1",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def _fake_corpus():
    rows = []
    creator = []
    rank = 0
    for capability_index, capability in enumerate(GCS_CAPABILITY_ORDER):
        for local_index in range(40):
            rank += 1
            query_id = f"q-{capability_index}-{local_index:02d}"
            leakage_group_id = f"leak-{rank:03d}"
            image_sha256 = hashlib.sha256(
                f"image-{rank:03d}".encode("utf-8")
            ).hexdigest()
            if capability == "utility.document_reading":
                success = False
            elif capability == "knowledge.visual_encyclopedia":
                success = local_index == 0
            else:
                success = local_index < 3
            hard_error = int(local_index == 39 and capability_index == 0)
            reasons = (
                ()
                if success
                else (
                    ("style_evidence_invalid",)
                    if local_index % 6 == 0
                    else ("material_claim_uncited",)
                    if local_index % 6 == 1
                    else ("card_contract_failed",)
                    if local_index % 6 == 2
                    else ("multi_mapping_invalid",)
                    if local_index % 6 == 3
                    else ("output_section_invalid",)
                )
            )
            passed = 5 if success else max(1, 4 - (local_index % 3))
            score = SimpleNamespace(
                gcs=int(success),
                hard_error=hard_error,
                route_acceptable=int(passed >= 1),
                no_hard_error=1 - hard_error,
                tool_contract_pass=int(passed >= 2),
                evidence_grounded=int(passed >= 3),
                output_contract_pass=int(passed >= 4),
                reason_codes=tuple(sorted(reasons)),
            )
            query = SimpleNamespace(
                query_id=query_id,
                canonical_capability=capability,
                asset_id=f"asset-{rank:03d}",
                leakage_group_id=leakage_group_id,
                image_path=f"selected/{rank:03d}.jpg",
                image_sha256=image_sha256,
            )
            rows.append(
                SimpleNamespace(
                    query=query,
                    query_ordinal=rank,
                    result=SimpleNamespace(),
                    score=score,
                    leakage_group_id=leakage_group_id,
                    atomic_component_id=f"atom-{capability_index}",
                    image_sha256=image_sha256,
                    checkpoint_file_sha256=f"{rank + 300:064x}",
                    checkpoint_row_sha256=f"{rank + 600:064x}",
                    sidecar=SimpleNamespace(evidence_sha256=f"{rank + 900:064x}"),
                    strata=SimpleNamespace(
                        source_dataset=f"source-{local_index % 3}",
                        repair_status=(
                            "r3_language_repaired"
                            if local_index % 4 == 0
                            else "r3_carry_forward"
                        ),
                        boundary_status=(
                            "boundary" if local_index % 5 == 0 else "non_boundary"
                        ),
                        style_submode=(
                            "same_category_alternative"
                            if capability == "product.style_recommendation"
                            else None
                        ),
                    ),
                )
            )
            creator.append(
                CreatorSelectionEntry(
                    selection_rank=rank,
                    plan_id=query_id,
                    component_id=leakage_group_id,
                    canonical_capability=capability,
                    is_boundary=local_index % 5 == 0,
                    interaction_pattern="single_turn",
                )
            )
    row_map = {row.query.query_id: row for row in rows}
    corpus = SimpleNamespace(
        corpus_sha256="a" * 64,
        runtime=SimpleNamespace(bank=SimpleNamespace(bank_sha256="b" * 64)),
        core_inputs=SimpleNamespace(runtime_asset_catalog=lambda: object()),
        row_by_query_id=lambda: row_map,
    )
    return corpus, tuple(creator)


@pytest.fixture
def selection(monkeypatch):
    corpus, creator = _fake_corpus()
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback."
        "require_verified_static_gcs_corpus",
        lambda value: value,
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback."
        "accept_loaded_verified_static_gcs_corpus",
        lambda value: value,
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback.build_feedback_packet",
        lambda query, _result, *, asset_catalog, rubric: _packet(
            query.query_id,
            query.canonical_capability,
            query.image_sha256,
            rubric,
        ),
    )
    return build_portfolio_s1_feedback_selection(
        corpus,
        creator,
        creator_selection_file_sha256="c" * 64,
        creator_selection_sha256="d" * 64,
        seed=17,
    )


def _packet(
    query_id: str,
    capability: str,
    image_sha256: str,
    rubric: RubricSnapshot,
) -> FeedbackPacket:
    unsigned = {
        "schema_version": 2,
        "packet_kind": "feedback",
        "cache_namespace": "feedback-evaluator-v2",
        "query_id": query_id,
        "turns": (ConversationTurn(role="user", content="请检查公开回答。"),),
        "image": EvaluationImage(mime_type="image/jpeg", sha256=image_sha256),
        "canonical_capability": capability,
        "acceptable_capabilities": (capability,),
        "response_text": "这是只包含用户可见内容的代表回答。",
        "cards": (),
        "tool_evidence": (),
        "tool_trace": (),
        "rubric": rubric,
    }
    draft = FeedbackPacket.model_construct(**unsigned, packet_sha256="0" * 64)
    payload = draft.model_dump(mode="json", exclude={"packet_sha256"})
    return FeedbackPacket.model_validate(
        {**unsigned, "packet_sha256": _hash(payload)}, strict=True
    )


def _parsed_result(
    packet: FeedbackPacket,
    control,
    remote_runtime=None,
    *,
    suggestion: str | None = None,
):
    raw = (
        canonical_json_bytes(
            {
                "schema_version": 1,
                "summary": "回答需要按冻结契约改进。",
                "rule_violations": [],
                "ideal_response_gaps": [],
                "skill_suggestions": [suggestion or "先核对证据，再生成结构化回答。"],
            }
        )
        .decode("utf-8")
        .strip()
    )
    parsed = VisualFeedbackOutput.model_validate_json(raw, strict=True)
    return _make_result(
        cache_namespace="feedback-evaluator-v8",
        parser_policy_version=VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        parser_policy_sha256=VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        prompt_policy_version=VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
        prompt_policy_sha256=VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
        transport_policy_version=VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
        transport_policy_sha256=VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
        requested_response_format=None,
        requested_thinking=False,
        requested_temperature=0.6,
        requested_top_p=0.95,
        query_id=packet.query_id,
        packet_sha256=packet.packet_sha256,
        prompt_sha256="1" * 64,
        image_sha256=packet.image.sha256,
        wire_sha256="2" * 64,
        asset_catalog_sha256=(
            "3" * 64
            if remote_runtime is None
            else remote_runtime.catalog.catalog_sha256
        ),
        remote_authorization_id=control.authorization_id,
        remote_authorization_file_sha256=control.authorization_file_sha256,
        remote_receipt_file_sha256=(
            "5" * 64 if remote_runtime is None else remote_runtime.receipt_file_sha256
        ),
        remote_receipt_sha256=(
            "6" * 64
            if remote_runtime is None
            else remote_runtime.receipt.receipt_sha256
        ),
        provider="kimi",
        model="kimi-k2.6",
        endpoint=config.PROVIDER_ENDPOINTS["kimi"],
        max_tokens=2048,
        status="parsed",
        request_id=f"request-{packet.query_id}",
        raw_response_text=raw,
        raw_response_sha256=sha256_bytes(raw.encode()),
        raw_response_bytes=len(raw.encode()),
        tool_calls=(),
        tool_call_count=0,
        parsed_feedback=parsed,
        usage=LLMUsage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
        latency_ms=1,
    )


def _parse_error_result(packet: FeedbackPacket, control):
    raw = "not-json"
    return _make_result(
        cache_namespace="feedback-evaluator-v8",
        parser_policy_version=VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
        parser_policy_sha256=VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
        prompt_policy_version=VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
        prompt_policy_sha256=VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
        transport_policy_version=VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3,
        transport_policy_sha256=VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
        requested_response_format=None,
        requested_thinking=False,
        requested_temperature=0.6,
        requested_top_p=0.95,
        query_id=packet.query_id,
        packet_sha256=packet.packet_sha256,
        prompt_sha256="1" * 64,
        image_sha256=packet.image.sha256,
        wire_sha256="2" * 64,
        asset_catalog_sha256="3" * 64,
        remote_authorization_id=control.authorization_id,
        remote_authorization_file_sha256=control.authorization_file_sha256,
        remote_receipt_file_sha256="5" * 64,
        remote_receipt_sha256="6" * 64,
        provider="kimi",
        model="kimi-k2.6",
        endpoint=config.PROVIDER_ENDPOINTS["kimi"],
        max_tokens=2048,
        status="parse_error",
        request_id=f"request-{packet.query_id}",
        raw_response_text=raw,
        raw_response_sha256=sha256_bytes(raw.encode()),
        raw_response_bytes=len(raw.encode()),
        tool_calls=(),
        tool_call_count=0,
        parsed_feedback=None,
        usage=LLMUsage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
        latency_ms=1,
        error_code="invalid_feedback_json",
    )


def _active_authorization(selection):
    selected_assets = tuple(
        SelectedFeedbackAssetV1(
            query_id=item.query_id,
            asset_id=item.asset_id,
            image_sha256=item.image_sha256,
        )
        for item in sorted(selection.entries, key=lambda entry: entry.query_id)
    )
    draft = PortfolioS1FeedbackAuthorizationV2.model_construct(
        authorization_id="owner-selected-feedback-v2",
        reviewer_id="project-owner",
        reviewed_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
        owner_statement=(
            "Authorize exactly the selected 48 images for private Kimi Feedback."
        ),
        selection_sha256=selection.selection_sha256,
        parent_remote_authorization_id="core-remote-v2",
        parent_remote_authorization_file_sha256="1" * 64,
        parent_remote_receipt_file_sha256="2" * 64,
        parent_remote_receipt_sha256="3" * 64,
        parent_remote_catalog_sha256="7" * 64,
        selected_assets=selected_assets,
        selected_asset_set_sha256=_hash(
            [item.model_dump(mode="json") for item in selected_assets]
        ),
        authorization_sha256="0" * 64,
    )
    payload = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    return PortfolioS1FeedbackAuthorizationV2.model_validate_json(
        canonical_json_bytes({**payload, "authorization_sha256": _hash(payload)}),
        strict=True,
    )


def _legacy_gemini_authorization(selection) -> PortfolioS1FeedbackAuthorizationV1:
    return build_selected_feedback_authorization(
        selection,
        authorization_id="owner-selected-gemini-feedback-v1",
        reviewer_id="project-owner",
        reviewed_at=datetime(2026, 8, 8, tzinfo=timezone.utc),
        owner_statement=(
            "Authorize exactly the selected 48 images for private Gemini Feedback."
        ),
    )


def _authorized_control_sources_and_artifacts(
    selection, *, first_suggestion: str | None = None
):
    authorization = _active_authorization(selection)
    control = build_portfolio_s1_feedback_control(
        selection, authorization, rubric=_rubric()
    )
    artifacts = []
    corpus, _creator = _fake_corpus()
    sources = build_verified_static_feedback_sources(corpus, selection, control)
    for entry, source in zip(selection.entries, sources, strict=True):
        packet = source.packet
        reservation = build_feedback_call_reservation(
            selection, control, entry, verified_source=source
        )
        artifacts.append(
            build_bound_feedback_artifact(
                selection,
                control,
                entry,
                packet,
                _parsed_result(
                    packet,
                    control,
                    suggestion=(
                        first_suggestion if entry.selection_ordinal == 1 else None
                    ),
                ),
                verified_source=source,
                reservation=reservation,
            )
        )
    return authorization, control, sources, tuple(artifacts)


def _authorized_control_and_artifacts(selection):
    authorization, control, _sources, artifacts = (
        _authorized_control_sources_and_artifacts(selection)
    )
    return authorization, control, artifacts


class _SelectedCatalog:
    def __init__(self, root: Path, selection) -> None:
        self.asset_root = root
        self.catalog_sha256 = "7" * 64
        assets = []
        for entry in selection.entries:
            rank = int(entry.asset_id.rsplit("-", 1)[1])
            local_path = f"selected/{rank:03d}.jpg"
            path = root / local_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"image-{rank:03d}".encode("utf-8"))
            assets.append(
                SimpleNamespace(
                    asset_id=entry.asset_id,
                    sha256=entry.image_sha256,
                    cloud_upload_allowed=True,
                    local_path=local_path,
                )
            )
        self.assets = tuple(assets)

    def require_verified_files(self) -> None:
        return None

    def verify_reference(self, asset_id, image_path, _leakage_group_id):
        asset = next(item for item in self.assets if item.asset_id == asset_id)
        if asset.local_path != image_path:
            raise ValueError("image path mismatch")
        return SimpleNamespace(asset=asset)

    def verify_asset_ids(self, asset_ids) -> None:
        known = {item.asset_id for item in self.assets}
        if not set(asset_ids).issubset(known):
            raise ValueError("asset identity mismatch")


def _remote_prepared(selection, tmp_path: Path, monkeypatch):
    authorization, control, sources, _artifacts = (
        _authorized_control_sources_and_artifacts(selection)
    )
    catalog = _SelectedCatalog(tmp_path / "assets", selection)
    verified_corpus = sources[0].corpus
    verified_corpus.core_inputs.expected_output_catalog_sha256 = catalog.catalog_sha256
    verified_corpus.core_inputs.runtime_asset_catalog = lambda: pytest.fail(
        "Kimi receipt must reuse the already-verified parent catalog"
    )
    parent_runtime = SimpleNamespace(
        authorization=SimpleNamespace(
            authorization_id="core-remote-v2",
            processor_scope=("dashscope-kimi-feedback",),
        ),
        authorization_file_sha256="1" * 64,
        receipt_file_sha256="2" * 64,
        receipt=SimpleNamespace(receipt_sha256="3" * 64),
        catalog=catalog,
        processor="dashscope-kimi-feedback",
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback_remote."
        "require_verified_portfolio_remote_processing_runtime",
        lambda value, **_kwargs: value,
    )
    runtime = prepare_selected_kimi_feedback_remote_runtime(
        verified_corpus,
        selection,
        authorization,
        control,
        parent_runtime,
        receipt_path=tmp_path / "remote-runtime-receipt.json",
        verified_sources=sources,
    )
    output_dir = tmp_path / "feedback-run"
    output_dir.mkdir()
    prepared = PreparedPortfolioS1FeedbackRun(
        output_dir=output_dir,
        corpus=verified_corpus,
        selection=selection,
        authorization=authorization,
        control=control,
        sources=sources,
        remote_runtime=runtime,
    )
    return prepared


def test_selection_is_deterministic_balanced_unique_and_marks_partial_anchors(
    monkeypatch,
) -> None:
    corpus, creator = _fake_corpus()
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback."
        "require_verified_static_gcs_corpus",
        lambda value: value,
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback."
        "accept_loaded_verified_static_gcs_corpus",
        lambda value: value,
    )
    first = build_portfolio_s1_feedback_selection(
        corpus,
        creator,
        creator_selection_file_sha256="c" * 64,
        creator_selection_sha256="d" * 64,
        seed=17,
    )
    second = build_portfolio_s1_feedback_selection(
        corpus,
        creator,
        creator_selection_file_sha256="c" * 64,
        creator_selection_sha256="d" * 64,
        seed=17,
    )

    assert first == second
    assert len(first.entries) == 48
    assert len({item.leakage_group_id for item in first.entries}) == 48
    assert len({item.asset_id for item in first.entries}) == 48
    for capability in GCS_CAPABILITY_ORDER:
        members = [item for item in first.entries if item.capability == capability]
        assert len(members) == 8
        assert sum(item.role == "failure" for item in members) == 6
    document = [
        item for item in first.entries if item.capability == "utility.document_reading"
    ]
    assert sum(item.role == "partial_anchor" for item in document) == 2


def test_selected_authorization_is_exact_and_control_is_self_bound(
    selection,
    tmp_path: Path,
) -> None:
    authorization, control, _artifacts = _authorized_control_and_artifacts(selection)

    validate_selected_feedback_authorization(authorization, selection)
    assert len(authorization.selected_assets) == 48
    assert authorization.processor == "dashscope-kimi-feedback"
    assert control.provider_call_ceiling == 48
    assert control.policy_version == "portfolio-s1-feedback-control-v7"
    assert control.parser_policy_version == VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3
    assert control.parser_policy_sha256 == VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3
    assert control.prompt_policy_version == VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5
    assert control.prompt_policy_sha256 == VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5
    assert (
        control.transport_policy_version == VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3
    )
    assert control.transport_policy_sha256 == VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3
    assert control.requested_response_format is None
    assert control.requested_thinking is False
    assert control.requested_temperature == 0.6
    assert control.requested_top_p == 0.95
    assert len(control.canary_entry_sha256s) == 6
    assert (
        tuple(
            next(
                item.capability
                for item in selection.entries
                if item.entry_sha256 == entry_sha256
            )
            for entry_sha256 in control.canary_entry_sha256s
        )
        == GCS_CAPABILITY_ORDER
    )
    assert len(control.remaining_entry_sha256s) == 42
    drifted = authorization.model_copy(update={"selection_sha256": "f" * 64})
    with pytest.raises(PortfolioS1FeedbackError, match="differs"):
        validate_selected_feedback_authorization(drifted, selection)

    control_path = tmp_path / "feedback-control-v3.json"
    control_path.write_bytes(control.canonical_bytes())
    assert (
        load_portfolio_s1_feedback_control(
            control_path,
            expected_file_sha256=sha256_bytes(control.canonical_bytes()),
        )
        == control
    )

    parser_drift = control.model_dump(mode="json")
    parser_drift["parser_policy_sha256"] = "f" * 64
    parser_drift["control_sha256"] = _hash(
        {key: value for key, value in parser_drift.items() if key != "control_sha256"}
    )
    with pytest.raises(ValueError, match="parser policy identity mismatch"):
        PortfolioS1FeedbackControlV1.model_validate_json(
            canonical_json_bytes(parser_drift), strict=True
        )

    prompt_drift = control.model_dump(mode="json")
    prompt_drift["prompt_policy_sha256"] = "f" * 64
    prompt_drift["control_sha256"] = _hash(
        {key: value for key, value in prompt_drift.items() if key != "control_sha256"}
    )
    with pytest.raises(ValueError, match="prompt policy identity mismatch"):
        PortfolioS1FeedbackControlV1.model_validate_json(
            canonical_json_bytes(prompt_drift), strict=True
        )

    transport_drift = control.model_dump(mode="json")
    transport_drift["transport_policy_sha256"] = "f" * 64
    transport_drift["control_sha256"] = _hash(
        {
            key: value
            for key, value in transport_drift.items()
            if key != "control_sha256"
        }
    )
    with pytest.raises(ValueError, match="transport policy identity mismatch"):
        PortfolioS1FeedbackControlV1.model_validate_json(
            canonical_json_bytes(transport_drift), strict=True
        )

    legacy = control.model_dump(mode="json")
    legacy["policy_version"] = "portfolio-s1-feedback-control-v1"
    legacy["provider"] = "gemini"
    legacy["model"] = "gemini-3.6-flash"
    legacy["processor"] = "aifast-gemini-feedback"
    legacy.pop("parser_policy_version")
    legacy.pop("parser_policy_sha256")
    legacy.pop("prompt_policy_version")
    legacy.pop("prompt_policy_sha256")
    legacy.pop("transport_policy_version")
    legacy.pop("transport_policy_sha256")
    legacy.pop("requested_response_format")
    legacy.pop("requested_thinking")
    legacy.pop("requested_temperature")
    legacy.pop("requested_top_p")
    legacy["control_sha256"] = _hash(
        {key: value for key, value in legacy.items() if key != "control_sha256"}
    )
    legacy_bytes = canonical_json_bytes(legacy)
    legacy_path = tmp_path / "feedback-control-v1.json"
    legacy_path.write_bytes(legacy_bytes)
    loaded_legacy = load_portfolio_s1_feedback_control(
        legacy_path,
        expected_file_sha256=sha256_bytes(legacy_bytes),
    )
    assert loaded_legacy.policy_version == "portfolio-s1-feedback-control-v1"
    with pytest.raises(PortfolioS1FeedbackError, match="binding drifted"):
        validate_portfolio_s1_feedback_control(loaded_legacy, selection, authorization)

    legacy_v2 = control.model_dump(mode="json")
    legacy_v2["policy_version"] = "portfolio-s1-feedback-control-v2"
    legacy_v2["provider"] = "gemini"
    legacy_v2["model"] = "gemini-3.6-flash"
    legacy_v2["processor"] = "aifast-gemini-feedback"
    legacy_v2["parser_policy_version"] = VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V2
    legacy_v2["parser_policy_sha256"] = VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V2
    legacy_v2.pop("prompt_policy_version")
    legacy_v2.pop("prompt_policy_sha256")
    legacy_v2.pop("transport_policy_version")
    legacy_v2.pop("transport_policy_sha256")
    legacy_v2.pop("requested_response_format")
    legacy_v2.pop("requested_thinking")
    legacy_v2.pop("requested_temperature")
    legacy_v2.pop("requested_top_p")
    legacy_v2["control_sha256"] = _hash(
        {key: value for key, value in legacy_v2.items() if key != "control_sha256"}
    )
    legacy_v2_bytes = canonical_json_bytes(legacy_v2)
    legacy_v2_path = tmp_path / "feedback-control-v2.json"
    legacy_v2_path.write_bytes(legacy_v2_bytes)
    loaded_legacy_v2 = load_portfolio_s1_feedback_control(
        legacy_v2_path,
        expected_file_sha256=sha256_bytes(legacy_v2_bytes),
    )
    assert loaded_legacy_v2.policy_version == "portfolio-s1-feedback-control-v2"
    assert loaded_legacy_v2.canonical_bytes() == legacy_v2_bytes
    with pytest.raises(PortfolioS1FeedbackError, match="binding drifted"):
        validate_portfolio_s1_feedback_control(
            loaded_legacy_v2, selection, authorization
        )

    legacy_v3 = control.model_dump(mode="json")
    legacy_v3["policy_version"] = "portfolio-s1-feedback-control-v3"
    legacy_v3["provider"] = "gemini"
    legacy_v3["model"] = "gemini-3.6-flash"
    legacy_v3["processor"] = "aifast-gemini-feedback"
    legacy_v3.pop("prompt_policy_version")
    legacy_v3.pop("prompt_policy_sha256")
    legacy_v3.pop("transport_policy_version")
    legacy_v3.pop("transport_policy_sha256")
    legacy_v3.pop("requested_response_format")
    legacy_v3.pop("requested_thinking")
    legacy_v3.pop("requested_temperature")
    legacy_v3.pop("requested_top_p")
    legacy_v3["control_sha256"] = _hash(
        {key: value for key, value in legacy_v3.items() if key != "control_sha256"}
    )
    legacy_v3_bytes = canonical_json_bytes(legacy_v3)
    legacy_v3_path = tmp_path / "feedback-control-v3.json"
    legacy_v3_path.write_bytes(legacy_v3_bytes)
    loaded_legacy_v3 = load_portfolio_s1_feedback_control(
        legacy_v3_path,
        expected_file_sha256=sha256_bytes(legacy_v3_bytes),
    )
    assert loaded_legacy_v3.policy_version == "portfolio-s1-feedback-control-v3"
    assert loaded_legacy_v3.canonical_bytes() == legacy_v3_bytes
    with pytest.raises(PortfolioS1FeedbackError, match="binding drifted"):
        validate_portfolio_s1_feedback_control(
            loaded_legacy_v3, selection, authorization
        )

    legacy_v4 = control.model_dump(mode="json")
    legacy_v4["policy_version"] = "portfolio-s1-feedback-control-v4"
    legacy_v4["provider"] = "gemini"
    legacy_v4["model"] = "gemini-3.6-flash"
    legacy_v4["processor"] = "aifast-gemini-feedback"
    legacy_v4["prompt_policy_version"] = VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4
    legacy_v4["prompt_policy_sha256"] = VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4
    legacy_v4.pop("transport_policy_version")
    legacy_v4.pop("transport_policy_sha256")
    legacy_v4.pop("requested_response_format")
    legacy_v4.pop("requested_thinking")
    legacy_v4.pop("requested_temperature")
    legacy_v4.pop("requested_top_p")
    legacy_v4["control_sha256"] = _hash(
        {key: value for key, value in legacy_v4.items() if key != "control_sha256"}
    )
    legacy_v4_bytes = canonical_json_bytes(legacy_v4)
    legacy_v4_path = tmp_path / "feedback-control-v4.json"
    legacy_v4_path.write_bytes(legacy_v4_bytes)
    loaded_legacy_v4 = load_portfolio_s1_feedback_control(
        legacy_v4_path,
        expected_file_sha256=sha256_bytes(legacy_v4_bytes),
    )
    assert loaded_legacy_v4.policy_version == "portfolio-s1-feedback-control-v4"
    assert loaded_legacy_v4.canonical_bytes() == legacy_v4_bytes
    assert b"transport_policy" not in legacy_v4_bytes
    with pytest.raises(PortfolioS1FeedbackError, match="binding drifted"):
        validate_portfolio_s1_feedback_control(
            loaded_legacy_v4, selection, authorization
        )

    legacy_gemini_authorization = _legacy_gemini_authorization(selection)
    legacy_v5 = control.model_dump(mode="json")
    legacy_v5["policy_version"] = "portfolio-s1-feedback-control-v5"
    legacy_v5["provider"] = "gemini"
    legacy_v5["model"] = "gemini-3.6-flash"
    legacy_v5["processor"] = "aifast-gemini-feedback"
    legacy_v5["authorization_sha256"] = legacy_gemini_authorization.authorization_sha256
    legacy_v5["authorization_id"] = legacy_gemini_authorization.authorization_id
    legacy_v5["authorization_file_sha256"] = sha256_bytes(
        legacy_gemini_authorization.canonical_bytes()
    )
    legacy_v5["prompt_policy_version"] = VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4
    legacy_v5["prompt_policy_sha256"] = VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4
    legacy_v5["transport_policy_version"] = VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V1
    legacy_v5["transport_policy_sha256"] = VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V1
    legacy_v5["requested_response_format"] = "json_object"
    legacy_v5.pop("requested_thinking")
    legacy_v5.pop("requested_temperature")
    legacy_v5.pop("requested_top_p")
    legacy_v5["control_sha256"] = _hash(
        {key: value for key, value in legacy_v5.items() if key != "control_sha256"}
    )
    legacy_v5_bytes = canonical_json_bytes(legacy_v5)
    legacy_v5_path = tmp_path / "feedback-control-v5.json"
    legacy_v5_path.write_bytes(legacy_v5_bytes)
    loaded_legacy_v5 = load_portfolio_s1_feedback_control(
        legacy_v5_path,
        expected_file_sha256=sha256_bytes(legacy_v5_bytes),
    )
    assert loaded_legacy_v5.policy_version == "portfolio-s1-feedback-control-v5"
    assert loaded_legacy_v5.canonical_bytes() == legacy_v5_bytes
    # BundleV2 deliberately consumes the frozen terminal Gemini control-v5.
    # The active prompt-v5 switch must not make that diagnostic handoff invalid.
    validate_portfolio_s1_feedback_control(
        loaded_legacy_v5,
        selection,
        legacy_gemini_authorization,
    )

    legacy_v6 = control.model_dump(mode="json")
    legacy_v6["policy_version"] = "portfolio-s1-feedback-control-v6"
    legacy_v6["prompt_policy_version"] = VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V4
    legacy_v6["prompt_policy_sha256"] = VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V4
    legacy_v6["transport_policy_version"] = VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V2
    legacy_v6["transport_policy_sha256"] = VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V2
    legacy_v6["control_sha256"] = _hash(
        {key: value for key, value in legacy_v6.items() if key != "control_sha256"}
    )
    legacy_v6_bytes = canonical_json_bytes(legacy_v6)
    legacy_v6_path = tmp_path / "feedback-control-v6.json"
    legacy_v6_path.write_bytes(legacy_v6_bytes)
    loaded_legacy_v6 = load_portfolio_s1_feedback_control(
        legacy_v6_path,
        expected_file_sha256=sha256_bytes(legacy_v6_bytes),
    )
    assert loaded_legacy_v6.policy_version == "portfolio-s1-feedback-control-v6"
    assert loaded_legacy_v6.canonical_bytes() == legacy_v6_bytes
    # It remains readable byte-for-byte, but cannot resume under prompt-v5.
    with pytest.raises(PortfolioS1FeedbackError, match="binding drifted"):
        validate_portfolio_s1_feedback_control(
            loaded_legacy_v6,
            selection,
            authorization,
        )


def test_bound_artifact_create_only_resume_and_conflict(
    selection, tmp_path: Path
) -> None:
    _authorization, control, sources, artifacts = (
        _authorized_control_sources_and_artifacts(selection)
    )
    path = tmp_path / "feedback" / "row.json"
    write_bound_feedback_artifact(path, artifacts[0])

    assert (
        resume_bound_feedback_artifact(
            path,
            selection=selection,
            control=control,
            entry=selection.entries[0],
            verified_source=sources[0],
            reservation=build_feedback_call_reservation(
                selection,
                control,
                selection.entries[0],
                verified_source=sources[0],
            ),
        )
        == artifacts[0]
    )
    with pytest.raises(PortfolioS1FeedbackError, match="conflict"):
        resume_bound_feedback_artifact(
            path,
            selection=selection,
            control=control,
            entry=selection.entries[1],
            verified_source=sources[1],
            reservation=build_feedback_call_reservation(
                selection,
                control,
                selection.entries[1],
                verified_source=sources[1],
            ),
        )
    with pytest.raises(FileExistsError):
        write_bound_feedback_artifact(path, artifacts[0])


def test_bundle_requires_parsed48_and_projection_excludes_private_identity(
    selection,
    tmp_path: Path,
) -> None:
    authorization, control, artifacts = _authorized_control_and_artifacts(selection)
    bundle = build_portfolio_s1_feedback_bundle(
        selection,
        control,
        authorization,
        artifacts,
        build_portfolio_s1_feedback_run(selection, control, artifacts),
    )
    projection = canonical_json_bytes(bundle.model_projection_payload())

    assert len(bundle.entries) == bundle.parsed_count == 48
    assert len(bundle.model_projection.feedback_entries) == 48
    examples = bundle.model_projection.representative_examples
    assert len(examples) == 18
    assert sum(item.role != "failure" for item in examples) == 12
    for forbidden in (
        b"query_id",
        b"asset_id",
        b"leakage_group_id",
        b"checkpoint",
        b"sidecar",
        b"bound_artifact_sha256",
        b"feedback_result_sha256",
        b"source_dataset",
        b"repair_status",
        b"authorization_sha256",
    ):
        assert forbidden not in projection
    assert b"response_text" in projection
    assert b"skill_suggestions" in projection

    path = tmp_path / "s1-feedback-bundle.json"
    write_portfolio_s1_feedback_bundle(path, bundle)
    assert (
        load_portfolio_s1_feedback_bundle(
            path, expected_file_sha256=sha256_bytes(path.read_bytes())
        )
        == bundle
    )

    incomplete = artifacts[:-1]
    with pytest.raises(PortfolioS1FeedbackError, match="48 artifacts"):
        build_portfolio_s1_feedback_bundle(
            selection,
            control,
            authorization,
            incomplete,
            build_portfolio_s1_feedback_run(selection, control, artifacts),
        )


@pytest.mark.parametrize(
    "forbidden_value",
    (
        "Gemini echoed private query q-0-00 in its advice.",
        "Do not expose asset.v2.private-item-17.",
        r"The artifact came from C:\Users\operator\private\row.json.",
        "Observed hash " + "a" * 64,
        "Screenshot data:image/png;base64,QUFBQUFBQUFBQUFB",
        "Reuse the scorer_payload private handle.",
    ),
)
def test_bundle_rejects_gemini_private_value_echoes(
    selection,
    forbidden_value: str,
) -> None:
    with pytest.raises(ValueError, match="Base64|private"):
        authorization, control, _sources, artifacts = (
            _authorized_control_sources_and_artifacts(
                selection, first_suggestion=forbidden_value
            )
        )
        build_portfolio_s1_feedback_bundle(
            selection,
            control,
            authorization,
            artifacts,
            build_portfolio_s1_feedback_run(selection, control, artifacts),
        )


def test_bundle_loader_rechecks_projection_value_privacy(
    selection,
    tmp_path: Path,
) -> None:
    authorization, control, artifacts = _authorized_control_and_artifacts(selection)
    bundle = build_portfolio_s1_feedback_bundle(
        selection,
        control,
        authorization,
        artifacts,
        build_portfolio_s1_feedback_run(selection, control, artifacts),
    )
    payload = bundle.model_dump(mode="json")
    malicious = "Private checkpoint_row handle was echoed."
    payload["entries"][0]["feedback"]["skill_suggestions"][0] = malicious
    payload["model_projection"]["feedback_entries"][0]["feedback"]["skill_suggestions"][
        0
    ] = malicious
    unsigned = {key: value for key, value in payload.items() if key != "bundle_sha256"}
    payload["bundle_sha256"] = _hash(unsigned)
    path = tmp_path / "malicious-feedback-bundle.json"
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="private value"):
        load_portfolio_s1_feedback_bundle(
            path, expected_file_sha256=sha256_bytes(path.read_bytes())
        )


def test_selected48_remote_runtime_enforces_query_asset_membership(
    selection, tmp_path: Path, monkeypatch
) -> None:
    prepared = _remote_prepared(selection, tmp_path, monkeypatch)
    first, second = prepared.sources[:2]

    receipt_path = tmp_path / "remote-runtime-receipt.json"
    receipt_content = receipt_path.read_bytes()
    assert receipt_content == prepared.remote_runtime.receipt.canonical_bytes()
    assert sha256_bytes(receipt_content) == prepared.remote_runtime.receipt_file_sha256
    assert (
        prepared.remote_runtime.receipt.catalog_sha256
        == prepared.remote_runtime.catalog.catalog_sha256
    )

    _runtime, image_bytes = load_verified_evaluator_image(
        prepared.remote_runtime,
        processor="dashscope-kimi-feedback",
        image=first.packet.image,
        query_id=first.packet.query_id,
    )
    rank = int(first.row.query.asset_id.rsplit("-", 1)[1])
    assert image_bytes == f"image-{rank:03d}".encode("utf-8")
    with pytest.raises(EvaluatorImageLoadError, match="selected48"):
        load_verified_evaluator_image(
            prepared.remote_runtime,
            processor="dashscope-kimi-feedback",
            image=second.packet.image,
            query_id=first.packet.query_id,
        )
    with pytest.raises(EvaluatorImageLoadError, match="selected48"):
        load_verified_evaluator_image(
            prepared.remote_runtime,
            processor="dashscope-kimi-feedback",
            image=first.packet.image,
            query_id="not-selected",
        )


def test_prepare_reuses_one_loaded_corpus_parent_runtime(
    selection, tmp_path: Path, monkeypatch
) -> None:
    loads = 0
    runtime_lookups = 0
    parent_runtime = object()

    def load_corpus(*_args, **_kwargs):
        nonlocal loads
        loads += 1
        return SimpleNamespace(
            core_inputs=SimpleNamespace(runtime_for=runtime_for),
        )

    def runtime_for(processor: str):
        nonlocal runtime_lookups
        runtime_lookups += 1
        assert processor == "dashscope-qwen-assistant"
        return parent_runtime

    authorization = _active_authorization(selection)
    control = build_portfolio_s1_feedback_control(
        selection, authorization, rubric=_rubric()
    )
    sources = (object(),)
    remote_runtime = object()
    observed_parents: list[object] = []
    validated_sources: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.load_verified_static_gcs_corpus",
        load_corpus,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._load_creator_entries",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.build_portfolio_s1_feedback_selection",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.build_selected_qwen_feedback_authorization",
        lambda _selection, parent, **_kwargs: (
            observed_parents.append(parent) or authorization
        ),
    )
    model_source_lock = SimpleNamespace(endpoint=config.PROVIDER_ENDPOINTS["qwen"])
    pricing_lock = object()
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.load_qwen37_feedback_model_source_lock",
        lambda *_args, **_kwargs: model_source_lock,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.load_qwen37_feedback_pricing_lock",
        lambda *_args, **_kwargs: pricing_lock,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._load_qwen_role_selection",
        lambda *_args, **_kwargs: ("4" * 64, "5" * 64),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._rubric", lambda *_args, **_kwargs: _rubric()
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.build_portfolio_s1_feedback_control",
        lambda *_args, **_kwargs: control,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.build_verified_static_feedback_sources",
        lambda *_args: sources,
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._validate_input_reservation_bounds",
        lambda observed: validated_sources.append(observed),
    )
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.prepare_selected_qwen_feedback_remote_runtime",
        lambda _corpus, _selection, _authorization, _control, parent, **_kwargs: (
            observed_parents.append(parent) or remote_runtime
        ),
    )
    launch_lock = SimpleNamespace(canonical_bytes=lambda: b"{}")
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.build_qwen37_feedback_launch_lock",
        lambda **_kwargs: launch_lock,
    )
    arguments = SimpleNamespace(
        execution_root=tmp_path / "execution",
        expected_execution_control_sha256="1" * 64,
        artifact_repository_root=tmp_path,
        creator_selection_index=tmp_path / "creator.jsonl",
        expected_creator_selection_sha256="2" * 64,
        authorization_id="selected48-v3",
        reviewer_id="project-owner",
        reviewed_at="2026-08-09T20:54:19+08:00",
        owner_statement="Authorize the frozen selected48 Qwen Feedback run.",
        run_id="qwen-feedback-run-v1",
        model_source_lock=tmp_path / "source-lock.json",
        expected_model_source_lock_sha256="6" * 64,
        pricing_lock=tmp_path / "pricing-lock.json",
        expected_pricing_lock_sha256="7" * 64,
        role_selection_file=tmp_path / "role-v8.json",
        expected_role_selection_file_sha256="4" * 64,
        expected_role_selection_sha256="5" * 64,
        rubric_file=tmp_path / "rubric.txt",
        expected_rubric_sha256="3" * 64,
        rubric_id="portfolio-s1-feedback-v1",
        rubric_version="1",
        output_dir=tmp_path / "prepared",
    )

    prepared = prepare_run(arguments)

    assert loads == 1
    assert runtime_lookups == 1
    assert observed_parents == [parent_runtime, parent_runtime]
    assert validated_sources == [sources]
    assert prepared.remote_runtime is remote_runtime


def test_historical_aifast_selected48_receipt_remains_parseable(selection) -> None:
    bindings = tuple(
        SelectedFeedbackRemoteBindingV1(
            selection_ordinal=entry.selection_ordinal,
            selection_entry_sha256=entry.entry_sha256,
            query_id=entry.query_id,
            asset_id=entry.asset_id,
            image_sha256=entry.image_sha256,
        )
        for entry in selection.entries
    )
    binding_hash = _hash([item.model_dump(mode="json") for item in bindings])
    draft = PortfolioS1FeedbackRemoteRuntimeReceiptV1.model_construct(
        selection_sha256=selection.selection_sha256,
        authorization_sha256="1" * 64,
        authorization_file_sha256="2" * 64,
        control_sha256="3" * 64,
        corpus_sha256="4" * 64,
        catalog_sha256="5" * 64,
        selected_bindings=bindings,
        selected_binding_set_sha256=binding_hash,
        receipt_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"receipt_sha256"})
    receipt = PortfolioS1FeedbackRemoteRuntimeReceiptV1.model_validate_json(
        canonical_json_bytes({**unsigned, "receipt_sha256": _hash(unsigned)}),
        strict=True,
    )

    assert receipt.schema_version == 1
    assert receipt.processor == "aifast-gemini-feedback"
    assert receipt.canonical_bytes() == canonical_json_bytes(
        receipt.model_dump(mode="json")
    )


def test_fake_provider_executes_canary_then_remaining_and_resumes_zero_call(
    selection, tmp_path: Path, monkeypatch
) -> None:
    prepared = _remote_prepared(selection, tmp_path, monkeypatch)
    calls: list[str] = []

    def fake_feedback(packet, _isolation, **_kwargs):
        calls.append(packet.query_id)
        return _parsed_result(packet, prepared.control, prepared.remote_runtime)

    assert (
        execute_run(
            prepared,
            feedback_runner=fake_feedback,
            stop_after_canary=True,
        )
        is None
    )
    assert len(calls) == 6
    assert set(calls[:6]) == set(selection.canary_query_ids)
    assert not (prepared.output_dir / "run.json").exists()
    assert not (prepared.output_dir / "portfolio-s1-feedback-bundle.json").exists()

    bundle = execute_run(prepared, feedback_runner=fake_feedback)
    assert len(calls) == 48
    assert set(calls[6:]) == set(selection.remaining_query_ids)
    assert bundle.run_sha256
    assert (prepared.output_dir / "run.json").is_file()
    assert (prepared.output_dir / "portfolio-s1-feedback-bundle.json").is_file()

    def forbidden_feedback(*_args, **_kwargs):
        raise AssertionError("resume attempted a provider call")

    assert execute_run(prepared, feedback_runner=forbidden_feedback) == bundle


def test_fake_provider_nonparsed_stops_without_replacement(
    selection, tmp_path: Path, monkeypatch
) -> None:
    prepared = _remote_prepared(selection, tmp_path, monkeypatch)
    calls: list[str] = []

    def fake_feedback(packet, _isolation, **_kwargs):
        calls.append(packet.query_id)
        if len(calls) == 1:
            return _make_result(
                cache_namespace="feedback-evaluator-v8",
                parser_policy_version=VISUAL_FEEDBACK_PARSER_POLICY_VERSION_V3,
                parser_policy_sha256=VISUAL_FEEDBACK_PARSER_POLICY_SHA256_V3,
                prompt_policy_version=VISUAL_FEEDBACK_PROMPT_POLICY_VERSION_V5,
                prompt_policy_sha256=VISUAL_FEEDBACK_PROMPT_POLICY_SHA256_V5,
                transport_policy_version=(VISUAL_FEEDBACK_TRANSPORT_POLICY_VERSION_V3),
                transport_policy_sha256=VISUAL_FEEDBACK_TRANSPORT_POLICY_SHA256_V3,
                requested_response_format=None,
                requested_thinking=False,
                requested_temperature=0.6,
                requested_top_p=0.95,
                query_id=packet.query_id,
                packet_sha256=packet.packet_sha256,
                prompt_sha256="1" * 64,
                image_sha256=packet.image.sha256,
                wire_sha256="2" * 64,
                asset_catalog_sha256=(prepared.remote_runtime.catalog.catalog_sha256),
                remote_authorization_id=prepared.authorization.authorization_id,
                remote_authorization_file_sha256=(
                    prepared.control.authorization_file_sha256
                ),
                remote_receipt_file_sha256=(
                    prepared.remote_runtime.receipt_file_sha256
                ),
                remote_receipt_sha256=(prepared.remote_runtime.receipt.receipt_sha256),
                provider="kimi",
                model="kimi-k2.6",
                endpoint=config.PROVIDER_ENDPOINTS["kimi"],
                max_tokens=2048,
                status="provider_error",
                error_code="provider_error",
            )
        return _parsed_result(packet, prepared.control, prepared.remote_runtime)

    with pytest.raises(PortfolioS1FeedbackRunError, match="nonparsed"):
        execute_run(prepared, feedback_runner=fake_feedback)
    assert len(calls) == 2
    run = (prepared.output_dir / "run.json").read_bytes()
    assert b'"status":"stopped_nonparsed"' in run

    with pytest.raises(PortfolioS1FeedbackRunError, match="previously stopped"):
        execute_run(
            prepared,
            feedback_runner=lambda *_args, **_kwargs: pytest.fail(
                "stopped run attempted a replacement call"
            ),
        )


def test_provider_reservation_crash_window_never_recalls(
    selection, tmp_path: Path, monkeypatch
) -> None:
    prepared = _remote_prepared(selection, tmp_path, monkeypatch)
    calls: list[str] = []

    def crashing_feedback(packet, _isolation, **_kwargs):
        calls.append(packet.query_id)
        raise RuntimeError("simulated crash after provider-call reservation")

    with pytest.raises(RuntimeError, match="simulated crash"):
        execute_run(prepared, feedback_runner=crashing_feedback)
    reserved_calls = len(calls)
    assert reserved_calls == 2
    reservations = tuple((prepared.output_dir / "provider-attempts").glob("*.json"))
    assert len(reservations) == reserved_calls
    assert not tuple((prepared.output_dir / "bound-feedback").glob("*.json"))
    budget = _budget_snapshot(prepared)
    assert budget.provider_calls_reserved == 2
    assert budget.usage_unknown_or_unresolved == 2
    assert budget.accountable_cost_cny > 0

    with pytest.raises(PortfolioS1FeedbackRunError, match="orphan"):
        execute_run(
            prepared,
            feedback_runner=lambda *_args, **_kwargs: pytest.fail(
                "orphan reservation attempted a provider recall"
            ),
        )
    assert len(calls) == reserved_calls


def test_pre_call_budget_guard_blocks_before_provider_and_reservation(
    selection, tmp_path: Path, monkeypatch
) -> None:
    prepared = _remote_prepared(selection, tmp_path, monkeypatch)
    provider_calls = 0

    def deny_budget(**_kwargs):
        raise ValueError("frozen Qwen Feedback budget denied")

    def forbidden_provider(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("budget rejection reached the provider")

    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback.require_qwen37_feedback_pre_call_budget",
        deny_budget,
    )

    with pytest.raises(ValueError, match="budget denied"):
        execute_run(prepared, feedback_runner=forbidden_provider)

    assert provider_calls == 0
    assert not (prepared.output_dir / "provider-attempts").exists()
    assert not (prepared.output_dir / "execute-writer.lock").exists()


def test_execute_writer_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    output = tmp_path / "feedback-run"
    output.mkdir()

    with _exclusive_execute_writer_lock(output):
        with pytest.raises(PortfolioS1FeedbackRunError, match="already exists"):
            with _exclusive_execute_writer_lock(output):
                pytest.fail("a second execute writer acquired the same run")

    assert not (output / "execute-writer.lock").exists()


def test_input_reservation_overflow_fails_before_remote_or_provider(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.run_portfolio_s1_feedback._estimated_input_tokens",
        lambda *_args: 20_001,
    )

    with pytest.raises(PortfolioS1FeedbackRunError, match="input-token reservation"):
        _validate_input_reservation_bounds((object(),))


def test_qwen_endpoint_drift_fails_before_provider(monkeypatch) -> None:
    monkeypatch.setitem(
        config.PROVIDER_ENDPOINTS,
        "qwen",
        "https://untrusted.example.invalid/v1",
    )

    with pytest.raises(PortfolioS1FeedbackRunError, match="frozen model source lock"):
        _require_frozen_qwen_endpoint(
            SimpleNamespace(
                endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        )


def test_qwen_missing_api_key_fails_before_reservation(monkeypatch) -> None:
    monkeypatch.delenv(config.PROVIDER_API_KEY_ENV["qwen"], raising=False)

    with pytest.raises(PortfolioS1FeedbackRunError, match="credential is absent"):
        _require_qwen_execute_environment()


def _full_gcs_diagnostics_for_bundle_v3():
    query_counts = (134, 134, 133, 133, 133, 133)
    capabilities = tuple(
        PortfolioS1FeedbackGCSCapabilitySummaryV2(
            capability=capability,
            query_count=count,
            gcs_pass_count=count,
            gcs_failure_count=0,
            hard_error_count=0,
        )
        for capability, count in zip(GCS_CAPABILITY_ORDER, query_counts, strict=True)
    )
    strata = tuple(
        PortfolioS1FeedbackGCSStrataSummaryV2(
            capability=capability,
            source_dataset="public-source",
            repair_status="r3_carry_forward",
            boundary_status="non_boundary",
            query_count=count,
            gcs_pass_count=count,
            gcs_failure_count=0,
        )
        for capability, count in zip(GCS_CAPABILITY_ORDER, query_counts, strict=True)
    )
    return (
        PortfolioS1FeedbackFullGCSBindingV2(
            corpus_sha256="a" * 64,
            execution_control_file_sha256="1" * 64,
            execution_control_sha256="2" * 64,
            launch_plan_sha256="3" * 64,
            runtime_lock_sha256="4" * 64,
            population_mapping_sha256="5" * 64,
            checkpoint_set_sha256="6" * 64,
            sidecar_set_sha256="7" * 64,
            parent_static_bank_sha256="b" * 64,
            gcs_reason_strata_sha256="8" * 64,
        ),
        PortfolioS1FeedbackFullGCSSummaryV2(
            gcs_pass_count=800,
            gcs_failure_count=0,
            hard_error_count=0,
            capability_summaries=capabilities,
            reason_summaries=(),
            strata_summaries=strata,
        ),
    )


def test_bundle_v3_freezes_exact_kimi_12_11_1_and_scans_projection(
    selection,
    tmp_path: Path,
    monkeypatch,
) -> None:
    authorization, control, sources, parsed48 = (
        _authorized_control_sources_and_artifacts(selection)
    )
    source = sources[6]
    entry = selection.entries[6]
    parse_error = build_bound_feedback_artifact(
        selection,
        control,
        entry,
        source.packet,
        _parse_error_result(source.packet, control),
        verified_source=source,
        reservation=build_feedback_call_reservation(
            selection, control, entry, verified_source=source
        ),
    )
    attempted_ordinals = (1, 2, 3, 4, 5, 6, 7, 9, 17, 25, 33, 41)
    by_entry = {artifact.selection_entry_sha256: artifact for artifact in parsed48}
    by_entry[entry.entry_sha256] = parse_error
    attempted = tuple(
        by_entry[selection.entries[ordinal - 1].entry_sha256]
        for ordinal in attempted_ordinals
    )
    run = build_portfolio_s1_feedback_run(selection, control, attempted)
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback._full_gcs_v2_diagnostics",
        lambda _corpus: _full_gcs_diagnostics_for_bundle_v3(),
    )

    bundle = build_portfolio_s1_feedback_bundle_v3(
        selection,
        control,
        authorization,
        attempted,
        run,
        corpus=sources[0].corpus,
    )

    assert bundle.schema_version == 3
    assert bundle.policy_version == "portfolio-s1-feedback-bundle-v3"
    assert (
        bundle.selected_count,
        bundle.attempted_count,
        bundle.parsed_count,
        bundle.parse_error_count,
        bundle.unattempted_count,
        bundle.missing_feedback_count,
    ) == (48, 12, 11, 1, 36, 37)
    assert bundle.parse_error_ordinals == (7,)
    assert tuple(item.selection_ordinal for item in bundle.entries) == (
        1,
        2,
        3,
        4,
        5,
        6,
        9,
        17,
        25,
        33,
        41,
    )
    projection = canonical_json_bytes(bundle.model_projection_payload())
    for forbidden in (
        b"query_id",
        b"asset_id",
        b"leakage_group_id",
        b"checkpoint",
        b"sidecar",
        b"bound_artifact_sha256",
        b"feedback_result_sha256",
        b"authorization_sha256",
    ):
        assert forbidden not in projection

    path = tmp_path / "portfolio-s1-feedback-bundle-v3.json"
    write_portfolio_s1_feedback_bundle_v3(path, bundle)
    assert (
        load_portfolio_s1_feedback_bundle_v3(
            path, expected_file_sha256=sha256_bytes(path.read_bytes())
        )
        == bundle
    )
    with pytest.raises(PortfolioS1FeedbackError, match="attempted12"):
        build_portfolio_s1_feedback_bundle_v3(
            selection,
            control,
            authorization,
            attempted[:-1],
            run,
            corpus=sources[0].corpus,
        )


def test_bundle_v4_requires_complete_qwen_parsed48_and_hides_reasoning(
    selection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillchain.evaluation.portfolio_s1_feedback import (
        build_portfolio_s1_feedback_bundle_v4,
        load_portfolio_s1_feedback_bundle_v4,
        write_portfolio_s1_feedback_bundle_v4,
    )
    from skillchain.evaluation.portfolio_s1_qwen_governance import (
        PortfolioS1QwenFeedbackAuthorizationV3,
        QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256,
        QWEN37_FEEDBACK_PRICING_LOCK_SHA256,
        QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
        QWEN37_FEEDBACK_ROLE_SELECTION_SHA256,
        QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
        QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
        SelectedQwenFeedbackAssetV1,
    )

    selected_assets = tuple(
        SelectedQwenFeedbackAssetV1(
            query_id=item.query_id,
            asset_id=item.asset_id,
            image_sha256=item.image_sha256,
        )
        for item in sorted(selection.entries, key=lambda entry: entry.query_id)
    )
    authorization_draft = PortfolioS1QwenFeedbackAuthorizationV3.model_construct(
        authorization_id="owner-selected-qwen-feedback-v3",
        reviewer_id="project-owner",
        reviewed_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        owner_statement=(
            "Authorize exactly the selected 48 images for private Qwen Feedback."
        ),
        selection_sha256=selection.selection_sha256,
        parent_remote_authorization_id="core-remote-v2",
        parent_remote_authorization_file_sha256="1" * 64,
        parent_remote_receipt_file_sha256="2" * 64,
        parent_remote_receipt_sha256="3" * 64,
        parent_remote_catalog_sha256="4" * 64,
        model_source_lock_file_sha256=QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
        model_source_lock_sha256=QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
        pricing_lock_file_sha256=QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256,
        pricing_lock_sha256=QWEN37_FEEDBACK_PRICING_LOCK_SHA256,
        role_selection_file_sha256=QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
        role_selection_sha256=QWEN37_FEEDBACK_ROLE_SELECTION_SHA256,
        selected_assets=selected_assets,
        selected_asset_set_sha256=_hash(
            [item.model_dump(mode="json") for item in selected_assets]
        ),
        authorization_sha256="0" * 64,
    )
    authorization_payload = authorization_draft.model_dump(
        mode="json", exclude={"authorization_sha256"}
    )
    authorization = PortfolioS1QwenFeedbackAuthorizationV3.model_validate_json(
        canonical_json_bytes(
            {
                **authorization_payload,
                "authorization_sha256": _hash(authorization_payload),
            }
        ),
        strict=True,
    )
    control = build_portfolio_s1_feedback_control(
        selection,
        authorization,
        rubric=_rubric(),
        feedback_concurrency=2,
    )
    corpus, _creator = _fake_corpus()
    sources = build_verified_static_feedback_sources(corpus, selection, control)

    artifacts = []
    for entry, source in zip(selection.entries, sources, strict=True):
        raw = (
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "summary": "The public answer needs grounded improvement.",
                    "rule_violations": [],
                    "ideal_response_gaps": [],
                    "skill_suggestions": [
                        "Verify visible evidence before producing the answer."
                    ],
                }
            )
            .decode("utf-8")
            .strip()
        )
        result = _make_result(
            schema_version=3,
            cache_namespace="feedback-evaluator-v9",
            parser_policy_version=control.parser_policy_version,
            parser_policy_sha256=control.parser_policy_sha256,
            prompt_policy_version=control.prompt_policy_version,
            prompt_policy_sha256=control.prompt_policy_sha256,
            transport_policy_version=control.transport_policy_version,
            transport_policy_sha256=control.transport_policy_sha256,
            requested_response_format=control.requested_response_format,
            requested_json_schema_sha256=control.requested_json_schema_sha256,
            requested_thinking=control.requested_thinking,
            requested_thinking_budget=control.requested_thinking_budget,
            requested_timeout_seconds=control.requested_timeout_seconds,
            requested_temperature=control.requested_temperature,
            requested_top_p=control.requested_top_p,
            query_id=source.packet.query_id,
            packet_sha256=source.packet.packet_sha256,
            prompt_sha256="b" * 64,
            image_sha256=source.packet.image.sha256,
            wire_sha256="c" * 64,
            asset_catalog_sha256="d" * 64,
            remote_authorization_id=control.authorization_id,
            remote_authorization_file_sha256=control.authorization_file_sha256,
            remote_receipt_file_sha256="e" * 64,
            remote_receipt_sha256="f" * 64,
            provider="qwen",
            model="qwen3.7-plus-2026-05-26",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            max_tokens=control.max_tokens,
            max_completion_tokens=control.max_completion_tokens,
            status="parsed",
            request_id=f"request-{source.packet.query_id}",
            raw_response_text=raw,
            raw_response_sha256=sha256_bytes(raw.encode("utf-8")),
            raw_response_bytes=len(raw.encode("utf-8")),
            tool_calls=(),
            tool_call_count=0,
            parsed_feedback=VisualFeedbackOutput.model_validate_json(raw, strict=True),
            usage=LLMUsage(input_tokens=10, output_tokens=7),
            finish_reason="stop",
            latency_ms=1,
            reasoning_present=True,
            reasoning_tokens=2,
            reasoning_bytes=9,
            reasoning_sha256="1" * 64,
        )
        reservation = build_feedback_call_reservation(
            selection,
            control,
            entry,
            verified_source=source,
        )
        artifacts.append(
            build_bound_feedback_artifact(
                selection,
                control,
                entry,
                source.packet,
                result,
                verified_source=source,
                reservation=reservation,
            )
        )
    artifacts = tuple(artifacts)
    run = build_portfolio_s1_feedback_run(selection, control, artifacts)
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_s1_feedback._full_gcs_v2_diagnostics",
        lambda _corpus: _full_gcs_diagnostics_for_bundle_v3(),
    )

    bundle = build_portfolio_s1_feedback_bundle_v4(
        selection,
        control,
        authorization,
        artifacts,
        run,
        corpus=sources[0].corpus,
    )

    assert bundle.schema_version == 4
    assert bundle.policy_version == "portfolio-s1-feedback-bundle-v4"
    assert (
        bundle.selected_count,
        bundle.attempted_count,
        bundle.parsed_count,
        bundle.provider_call_count,
        bundle.missing_feedback_count,
    ) == (48, 48, 48, 48, 0)
    projection = canonical_json_bytes(bundle.model_projection_payload())
    for forbidden in (
        b"query_id",
        b"asset_id",
        b"leakage_group_id",
        b"raw_response",
        b"reasoning",
        b"checkpoint",
        b"sidecar",
        b"bound_artifact_sha256",
        b"feedback_result_sha256",
        b"authorization_sha256",
    ):
        assert forbidden not in projection

    path = tmp_path / "portfolio-s1-feedback-bundle-v4.json"
    write_portfolio_s1_feedback_bundle_v4(path, bundle)
    assert (
        load_portfolio_s1_feedback_bundle_v4(
            path,
            expected_file_sha256=sha256_bytes(path.read_bytes()),
        )
        == bundle
    )
    with pytest.raises(PortfolioS1FeedbackError, match="exact 48"):
        build_portfolio_s1_feedback_bundle_v4(
            selection,
            control,
            authorization,
            artifacts[:-1],
            run,
            corpus=sources[0].corpus,
        )
