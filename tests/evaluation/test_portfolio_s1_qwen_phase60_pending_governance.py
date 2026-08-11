from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import pytest

import skillchain.evaluation.portfolio_s1_qwen_governance as governance
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    PortfolioS1QwenFeedbackGovernanceError,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
SOURCE_V5 = ROOT / governance.QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V5
PRICING_V8 = ROOT / governance.QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V8
ROLE_V15 = ROOT / governance.QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V15


def _load_triad():
    source = governance.load_qwen38_feedback_model_source_lock_v5(
        SOURCE_V5,
        expected_file_sha256=governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5,
    )
    pricing = governance.load_qwen38_feedback_pricing_lock_v8(
        PRICING_V8,
        expected_file_sha256=governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8,
    )
    role = governance.load_qwen38_feedback_role_selection_v15(
        ROLE_V15,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
        ),
    )
    return source, pricing, role


def test_phase60_retry_policy_is_new_pending_authority_not_canary_retry_reuse() -> None:
    policy = governance.round3_phase60_retry_policy_v1()

    assert policy["policy_version"] == (
        "portfolio-s1-feedback-round3-phase60-global-retry-v1"
    )
    assert policy["prefix_selected_count"] == 12
    assert policy["prefix_provider_call_count"] == 15
    assert policy["prefix_retry_claims_consumed"] == 3
    assert policy["prefix_retry_claims_reusable"] is False
    assert policy["new_first_calls"] == 48
    assert policy["new_global_retry_ceiling"] == 12
    assert policy["new_provider_call_ceiling"] == 60
    assert policy["cumulative_provider_call_ceiling"] == 75
    assert policy["live_provider_calls_authorized"] is False
    assert policy["owner_phase60_budget_and_retry_approval_status"] == "pending"
    assert sha256_bytes(canonical_json_bytes(policy)) == (
        governance.ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    )
    assert governance.ROUND3_PHASE60_RETRY_POLICY_SHA256_V1 != (
        governance.QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2
    )


def test_phase60_pending_triad_is_canonical_exact_and_zero_call() -> None:
    source, pricing, role = _load_triad()

    assert sha256_bytes(SOURCE_V5.read_bytes()) == (
        governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    )
    assert sha256_bytes(PRICING_V8.read_bytes()) == (
        governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8
    )
    assert sha256_bytes(ROLE_V15.read_bytes()) == (
        governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    )
    assert source.source_lock_sha256 == (
        governance.QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5
    )
    assert pricing.pricing_lock_sha256 == (
        governance.QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8
    )
    assert role.selection_sha256 == (
        governance.QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15
    )

    assert source.requested_response_format == "json_schema"
    assert source.requested_json_schema_strict is True
    assert source.requested_json_schema_sha256 == (
        governance.QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    assert source.transport_policy_sha256 == (
        governance.ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    )
    assert source.outer_orchestration_policy_sha256 == (
        governance.ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    )
    assert source.phase60_prefix_selected_count == 12
    assert source.phase60_new_first_call_count == 48
    assert source.phase60_new_retry_token_count == 12
    assert source.provider_call_ceiling == 60
    assert source.cumulative_provider_call_ceiling == 75
    assert source.canary_prefix_retry_count == 3
    assert source.canary_prefix_retry_tokens_reusable is False
    assert source.historical_feedback_outputs_imported == 12
    assert source.terminated_json_object_canary_outputs_imported == 0
    assert source.live_call_authority is False
    assert source.live_provider_calls_authorized is False
    assert source.owner_phase60_budget_authorization_status == "pending"
    assert source.owner_phase60_retry_authorization_status == "pending"

    assert pricing.live_authorized_selected_query_count == 0
    assert pricing.live_authorized_phase_counts == ()
    assert pricing.phase60_prefix_selected_count == 12
    assert pricing.phase60_new_first_call_count == 48
    assert pricing.global_retry_token_count == 12
    assert pricing.prefix_global_retry_token_count == 3
    assert pricing.prefix_global_retry_tokens_reusable is False
    assert pricing.provider_call_ceiling == 60
    assert pricing.cumulative_provider_call_ceiling == 75
    assert pricing.live_provider_calls_authorized is False
    assert pricing.fresh_run_and_retry_scope_owner_approved is False
    assert pricing.owner_budget_authorized_cap_cny == "0.000000000000"
    assert pricing.owner_budget_authorized_on is None
    assert pricing.owner_budget_authorization_status == (
        "pending_phase60_budget_and_retry_approval"
    )
    assert pricing.owner_phase60_retry_authorization_status == "pending"

    feedback = role.feedback_evaluator
    assert feedback["model_source_lock_file_sha256"] == sha256_bytes(
        SOURCE_V5.read_bytes()
    )
    assert feedback["pricing_lock_file_sha256"] == sha256_bytes(PRICING_V8.read_bytes())
    assert feedback["live_call_authority"] is False
    assert feedback["live_provider_calls_authorized"] is False
    assert feedback["owner_phase60_budget_authorization_status"] == "pending"
    assert feedback["owner_phase60_retry_authorization_status"] == "pending"
    assert feedback["creator_authorized"] is False
    assert feedback["bundle_v11_publishable"] is False


def test_phase60_pending_budget_arithmetic_is_exact() -> None:
    _source, pricing, _role = _load_triad()

    per_call = Decimal("20000") * Decimal("12") / Decimal("1000000") + (
        Decimal("6154") * Decimal("36") / Decimal("1000000")
    )
    assert per_call == Decimal("0.461544")
    assert Decimal(pricing.per_call_reservation_cny) == per_call
    assert Decimal("48") * per_call == Decimal("22.154112")
    assert Decimal("12") * per_call == Decimal("5.538528")
    assert Decimal("60") * per_call == Decimal("27.692640")
    assert pricing.maximum_reservation_cny == "27.692640000000"
    assert Decimal(
        governance.QWEN38_FEEDBACK_ROUND3_SCHEMA_PRIOR_ACTUAL_COST_CNY
    ) + Decimal(pricing.canary_prefix_actual_cost_cny) == Decimal(
        pricing.prior_cumulative_actual_cost_cny
    )
    assert pricing.prior_cumulative_actual_cost_cny == "26.264100000000"
    assert Decimal(pricing.prior_cumulative_actual_cost_cny) + Decimal(
        pricing.maximum_reservation_cny
    ) == Decimal(pricing.cumulative_maximum_reservation_cny)
    assert pricing.cumulative_maximum_reservation_cny == "53.956740000000"
    assert pricing.technical_phase_hard_cap_cny == "28.000000000000"
    assert pricing.live_cumulative_hard_cap_cny == "54.264100000000"


def test_phase60_pending_budget_helper_fails_before_provider_authority() -> None:
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="owner budget and retry approval is pending",
    ):
        governance.require_qwen38_feedback_pre_call_budget_v5(
            estimated_input_tokens_including_images=20_000,
            provider_calls_already_reserved=0,
            committed_cumulative_cost_cny="26.264100000000",
        )

    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="60-call provider ceiling",
    ):
        governance.require_qwen38_feedback_pre_call_budget_v5(
            estimated_input_tokens_including_images=1,
            provider_calls_already_reserved=60,
            committed_cumulative_cost_cny="26.264100000000",
        )


def test_phase60_pending_loaders_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="source lock v5 file SHA-256 mismatch",
    ):
        governance.load_qwen38_feedback_model_source_lock_v5(
            SOURCE_V5, expected_file_sha256="0" * 64
        )

    pricing_payload = json.loads(PRICING_V8.read_text(encoding="utf-8"))
    pricing_payload["live_provider_calls_authorized"] = True
    invalid_pricing = tmp_path / "pricing-v8-invalid.json"
    invalid_pricing.write_bytes(canonical_json_bytes(pricing_payload))
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="pricing lock v8 is invalid",
    ):
        governance.load_qwen38_feedback_pricing_lock_v8(
            invalid_pricing,
            expected_file_sha256=sha256_bytes(invalid_pricing.read_bytes()),
        )

    role_payload = json.loads(ROLE_V15.read_text(encoding="utf-8"))
    noncanonical_role = tmp_path / "role-v15-noncanonical.json"
    noncanonical_role.write_text(
        json.dumps(role_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="role selection v15 is not canonical JSON",
    ):
        governance.load_qwen38_feedback_role_selection_v15(
            noncanonical_role,
            expected_file_sha256=sha256_bytes(noncanonical_role.read_bytes()),
        )
