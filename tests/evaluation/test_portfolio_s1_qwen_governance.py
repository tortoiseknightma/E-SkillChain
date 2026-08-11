from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import skillchain.evaluation.portfolio_s1_qwen_governance as governance
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    PortfolioS1QwenFeedbackGovernanceError,
    build_qwen37_feedback_launch_lock,
    build_selected_qwen_feedback_authorization,
    load_qwen37_feedback_launch_lock,
    load_qwen37_feedback_model_source_lock,
    load_qwen37_feedback_pricing_lock,
    load_selected_qwen_feedback_authorization,
    require_qwen37_feedback_pre_call_budget,
    write_qwen37_feedback_launch_lock,
    write_selected_qwen_feedback_authorization,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
SOURCE_LOCK = (
    ROOT / "specs/authoring/qwen3.7-plus-2026-05-26-feedback-source-lock-v1.json"
)
PRICING_LOCK = ROOT / "specs/authoring/price-qwen3.7-plus-2026-05-26-feedback-v1.json"
ROLE_V7 = ROOT / "specs/authoring/model-role-selection-v7.json"
ROLE_V8 = ROOT / "specs/authoring/model-role-selection-v8.json"
CORE_AUTH = (
    ROOT / "specs/data_sources/c2/portfolio-core-remote-processing-v1/"
    "owner-authorization-v1.json"
)


def _hash(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


class _FakeCatalog:
    def __init__(self, assets: tuple[SimpleNamespace, ...]) -> None:
        self.assets = assets
        self.catalog_sha256 = _hash("core-catalog")
        self.verified = False

    def require_verified_files(self) -> None:
        self.verified = True


def _selection(count: int = 48) -> SimpleNamespace:
    return SimpleNamespace(
        selection_sha256=_hash(f"selection-{count}"),
        entries=tuple(
            SimpleNamespace(
                query_id=f"q-{index:02d}",
                asset_id=f"asset-{index:02d}",
                image_sha256=_hash(f"image-{index:02d}"),
            )
            for index in range(count)
        ),
    )


def _parent_runtime(selection: SimpleNamespace) -> SimpleNamespace:
    catalog = _FakeCatalog(
        tuple(
            SimpleNamespace(
                asset_id=entry.asset_id,
                sha256=entry.image_sha256,
                cloud_upload_allowed=True,
            )
            for entry in selection.entries
        )
    )
    return SimpleNamespace(
        authorization=SimpleNamespace(
            authorization_id="historical-core-owner-v1",
            processor_scope=(
                "dashscope-kimi-feedback",
                "dashscope-kimi-judge",
                "dashscope-qwen-assistant",
            ),
        ),
        receipt=SimpleNamespace(receipt_sha256=_hash("parent-receipt-self")),
        catalog=catalog,
        authorization_file_sha256=_hash("parent-auth-file"),
        receipt_file_sha256=_hash("parent-receipt-file"),
    )


def _locks() -> tuple[object, str, object, str]:
    source_file_sha256 = sha256_bytes(SOURCE_LOCK.read_bytes())
    pricing_file_sha256 = sha256_bytes(PRICING_LOCK.read_bytes())
    source = load_qwen37_feedback_model_source_lock(
        SOURCE_LOCK, expected_file_sha256=source_file_sha256
    )
    pricing = load_qwen37_feedback_pricing_lock(
        PRICING_LOCK, expected_file_sha256=pricing_file_sha256
    )
    return source, source_file_sha256, pricing, pricing_file_sha256


def _role_identity() -> tuple[str, str, dict[str, object]]:
    content = ROLE_V8.read_bytes()
    payload = json.loads(content)
    unsigned = {
        key: value for key, value in payload.items() if key != "selection_sha256"
    }
    assert content == canonical_json_bytes(payload)
    assert payload["selection_sha256"] == sha256_bytes(canonical_json_bytes(unsigned))
    return sha256_bytes(content), payload["selection_sha256"], payload


def _patch_parent_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def _accept(runtime, *, processor: str, catalog_sha256: str):
        assert processor == "dashscope-qwen-assistant"
        assert catalog_sha256 == runtime.catalog.catalog_sha256
        return runtime

    monkeypatch.setattr(
        governance,
        "require_verified_portfolio_remote_processing_runtime",
        _accept,
    )


def _authorization(monkeypatch: pytest.MonkeyPatch):
    _patch_parent_verifier(monkeypatch)
    selection = _selection()
    parent = _parent_runtime(selection)
    source, source_file_sha256, pricing, pricing_file_sha256 = _locks()
    role_file_sha256, role_sha256, _ = _role_identity()
    authorization = build_selected_qwen_feedback_authorization(
        selection,
        parent,
        authorization_id="qwen37-feedback-selected48-20260810",
        reviewer_id="portfolio-owner",
        reviewed_at=datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
        owner_statement=(
            "Approve qwen3.7-plus-2026-05-26 Feedback for only the frozen "
            "selected48 assets."
        ),
        model_source_lock=source,
        model_source_lock_file_sha256=source_file_sha256,
        pricing_lock=pricing,
        pricing_lock_file_sha256=pricing_file_sha256,
        role_selection_file_sha256=role_file_sha256,
        role_selection_sha256=role_sha256,
    )
    return (
        authorization,
        selection,
        parent,
        source,
        source_file_sha256,
        pricing,
        pricing_file_sha256,
        role_file_sha256,
        role_sha256,
    )


def test_qwen_source_pricing_and_role_locks_are_exact() -> None:
    source, source_file_sha256, pricing, pricing_file_sha256 = _locks()

    assert source_file_sha256 == governance.QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256
    assert source.source_lock_sha256 == governance.QWEN37_FEEDBACK_SOURCE_LOCK_SHA256
    assert source.provider_max_output_tokens == 131_072
    assert source.structured_output_response_format == "json_schema"
    assert source.structured_output_strict is True
    assert source.max_tokens_excludes_reasoning is True
    assert source.max_completion_tokens_includes_reasoning_and_answer is True
    assert source.max_completion_tokens_documented_upper_tolerance_tokens == 10
    assert pricing_file_sha256 == governance.QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256
    assert pricing.pricing_lock_sha256 == governance.QWEN37_FEEDBACK_PRICING_LOCK_SHA256
    assert pricing.input_cny_per_million_tokens == 2
    assert pricing.output_cny_per_million_tokens == 8
    assert pricing.reasoning_tokens_billed_as_output_tokens is True
    assert pricing.wire_max_completion_tokens == 4096
    assert pricing.max_completion_tokens_documented_upper_tolerance_tokens == 10
    assert pricing.input_token_reservation_ceiling_per_call == 20_000
    assert pricing.output_token_reservation_ceiling_per_call == 4106
    assert pricing.per_call_reservation_cny == "0.072848000000"
    assert pricing.maximum_reservation_cny == "3.496704000000"

    role_file_sha256, _, role = _role_identity()
    feedback = role["feedback_evaluator"]
    isolation = role["evaluator_isolation"]
    assert role_file_sha256 == governance.QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256
    assert role["selection_sha256"] == governance.QWEN37_FEEDBACK_ROLE_SELECTION_SHA256
    assert feedback["provider"] == "qwen"
    assert feedback["model"] == "qwen3.7-plus-2026-05-26"
    assert feedback["processor"] == "dashscope-qwen37-feedback"
    assert feedback["requested_response_format"] == "json_schema"
    assert feedback["requested_json_schema_strict"] is True
    assert feedback["requested_json_schema_sha256"] == (
        governance.QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
    )
    assert feedback["enable_thinking"] is True
    assert feedback["thinking_budget"] == 2048
    assert feedback["max_tokens"] is None
    assert feedback["max_completion_tokens"] == 4096
    assert feedback["max_completion_tokens_documented_upper_tolerance_tokens"] == 10
    assert feedback["input_token_reservation_ceiling_per_call"] == 20_000
    assert feedback["output_token_reservation_ceiling_per_call"] == 4106
    assert feedback["per_call_reservation_cny"] == "0.072848000000"
    assert feedback["worst_case_reservation_cny"] == "3.496704000000"
    assert feedback["phase_hard_cap_cny"] == "4.000000000000"
    assert isolation["assistant_feedback_provider_independent"] is False
    assert isolation["feedback_judge_provider_independent"] is True
    assert (
        "share the DashScope Qwen provider/gateway"
        in isolation["independence_limitation"]
    )
    assert sha256_bytes(ROLE_V7.read_bytes()) == (
        "fbfbe9437731052743b3025962a22e4d3cd6432c2212b7cc418b2118e9fd2ba4"
    )


def test_qwen_selected48_authorization_is_independent_and_create_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authorization, selection, parent, *_ = _authorization(monkeypatch)

    assert authorization.provider == "qwen"
    assert authorization.processor == "dashscope-qwen37-feedback"
    assert authorization.requested_response_format == "json_schema"
    assert authorization.requested_json_schema_strict is True
    assert authorization.requested_thinking is True
    assert authorization.requested_thinking_budget == 2048
    assert authorization.requested_max_tokens is None
    assert authorization.max_completion_tokens == 4096
    assert authorization.max_completion_tokens_documented_upper_tolerance_tokens == 10
    assert authorization.input_token_reservation_ceiling_per_call == 20_000
    assert authorization.output_token_reservation_ceiling_per_call == 4106
    assert authorization.parent_membership_processor == "dashscope-qwen-assistant"
    assert authorization.parent_authority_used_for_role_permission is False
    assert authorization.per_call_reservation_cny == "0.072848000000"
    assert authorization.maximum_reservation_cny == "3.496704000000"
    assert authorization.phase_hard_cap_cny == "4.000000000000"
    assert len(authorization.selected_assets) == 48
    assert parent.catalog.verified is True

    output = tmp_path / "selected48-qwen-authorization.json"
    write_selected_qwen_feedback_authorization(output, authorization)
    loaded = load_selected_qwen_feedback_authorization(
        output, expected_file_sha256=sha256_bytes(output.read_bytes())
    )
    assert loaded == authorization
    with pytest.raises(FileExistsError):
        write_selected_qwen_feedback_authorization(output, authorization)

    oversized = _selection(49)
    with pytest.raises(PortfolioS1QwenFeedbackGovernanceError, match="exactly 48"):
        governance.validate_selected_qwen_feedback_authorization(
            authorization, oversized
        )
    assert selection.selection_sha256 == authorization.selection_sha256


def test_qwen_authorization_rejects_catalog_and_lock_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_parent_verifier(monkeypatch)
    selection = _selection()
    parent = _parent_runtime(selection)
    source, source_file_sha256, pricing, pricing_file_sha256 = _locks()
    role_file_sha256, role_sha256, _ = _role_identity()
    parent.catalog.assets[0].sha256 = _hash("wrong-image")

    arguments = dict(
        authorization_id="qwen37-feedback-selected48-20260810",
        reviewer_id="portfolio-owner",
        reviewed_at=datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
        owner_statement="Approve only the frozen selected48 assets.",
        model_source_lock=source,
        model_source_lock_file_sha256=source_file_sha256,
        pricing_lock=pricing,
        pricing_lock_file_sha256=pricing_file_sha256,
        role_selection_file_sha256=role_file_sha256,
        role_selection_sha256=role_sha256,
    )
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError, match="catalog membership"
    ):
        build_selected_qwen_feedback_authorization(selection, parent, **arguments)

    parent = _parent_runtime(selection)
    arguments["model_source_lock_file_sha256"] = "0" * 64
    with pytest.raises(PortfolioS1QwenFeedbackGovernanceError, match="source lock"):
        build_selected_qwen_feedback_authorization(selection, parent, **arguments)


def test_qwen_launch_and_budget_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (
        authorization,
        selection,
        _,
        source,
        source_file_sha256,
        pricing,
        pricing_file_sha256,
        role_file_sha256,
        role_sha256,
    ) = _authorization(monkeypatch)
    launch = build_qwen37_feedback_launch_lock(
        run_id="qwen37-feedback-run-20260810",
        selection_sha256=selection.selection_sha256,
        corpus_sha256=_hash("verified-corpus"),
        authorization=authorization,
        model_source_lock=source,
        model_source_lock_file_sha256=source_file_sha256,
        pricing_lock=pricing,
        pricing_lock_file_sha256=pricing_file_sha256,
        role_selection_file_sha256=role_file_sha256,
        role_selection_sha256=role_sha256,
        control_file_sha256=_hash("control-file"),
        control_sha256=_hash("control-self"),
    )
    assert launch.provider_calls_performed == 0
    assert launch.requested_max_tokens is None
    assert launch.max_completion_tokens == 4096
    assert launch.max_completion_tokens_documented_upper_tolerance_tokens == 10
    assert launch.input_token_reservation_ceiling_per_call == 20_000
    assert launch.output_token_reservation_ceiling_per_call == 4106
    assert launch.output_reservation_includes_reasoning_and_answer_tokens is True
    assert launch.per_call_reservation_cny == "0.072848000000"
    assert launch.maximum_reservation_cny == "3.496704000000"
    assert launch.phase_hard_cap_cny == "4.000000000000"

    output = tmp_path / "launch-lock.json"
    write_qwen37_feedback_launch_lock(output, launch)
    assert (
        load_qwen37_feedback_launch_lock(
            output, expected_file_sha256=sha256_bytes(output.read_bytes())
        )
        == launch
    )

    assert (
        require_qwen37_feedback_pre_call_budget(
            estimated_input_tokens_including_images=20_000,
            provider_calls_already_reserved=47,
            committed_cost_cny="3.927152000000",
        )
        == "0.072848000000"
    )
    with pytest.raises(PortfolioS1QwenFeedbackGovernanceError, match="input estimate"):
        require_qwen37_feedback_pre_call_budget(
            estimated_input_tokens_including_images=20_001,
            provider_calls_already_reserved=0,
            committed_cost_cny="0",
        )
    with pytest.raises(PortfolioS1QwenFeedbackGovernanceError, match="call ceiling"):
        require_qwen37_feedback_pre_call_budget(
            estimated_input_tokens_including_images=1,
            provider_calls_already_reserved=48,
            committed_cost_cny="0",
        )
    with pytest.raises(PortfolioS1QwenFeedbackGovernanceError, match="hard cap"):
        require_qwen37_feedback_pre_call_budget(
            estimated_input_tokens_including_images=1,
            provider_calls_already_reserved=47,
            committed_cost_cny="3.927153000000",
        )


def test_historical_core_owner_scope_is_not_retyped_for_qwen_feedback() -> None:
    assert sha256_bytes(CORE_AUTH.read_bytes()) == (
        "2cbcb856fc2ae1a578bbbbd73dbe86df538d8053ab5f988c11a0ba7b046afd55"
    )
    payload = json.loads(CORE_AUTH.read_text(encoding="utf-8"))
    assert payload["processor_scope"] == [
        "dashscope-kimi-feedback",
        "dashscope-kimi-judge",
        "dashscope-qwen-assistant",
    ]
    assert "dashscope-qwen37-feedback" not in payload["processor_scope"]
