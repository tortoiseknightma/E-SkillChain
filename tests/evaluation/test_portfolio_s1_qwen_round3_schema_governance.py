from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import pytest

import skillchain.evaluation.portfolio_s1_qwen_governance as governance
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1 as RUNTIME_TRANSPORT_SHA256,
)
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1 as RUNTIME_TRANSPORT_VERSION,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_schema_v1 import (
    ROUND3_SCHEMA_RETRY_POLICY_SHA256_V2 as RUNTIME_RETRY_SHA256,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_schema_v1 import (
    ROUND3_SCHEMA_RETRY_POLICY_VERSION_V2 as RUNTIME_RETRY_VERSION,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    PortfolioS1QwenFeedbackGovernanceError,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
SOURCE_V3 = ROOT / governance.QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3
SOURCE_V4 = ROOT / governance.QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V4
PRICING_V6 = ROOT / governance.QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6
PRICING_V7 = ROOT / governance.QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V7
ROLE_V13 = ROOT / governance.QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V13
ROLE_V14 = ROOT / governance.QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V14


def test_round3_schema_governance_triad_is_canonical_and_exact() -> None:
    assert governance.ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1 == (
        RUNTIME_TRANSPORT_VERSION
    )
    assert governance.ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1 == (
        RUNTIME_TRANSPORT_SHA256
    )
    assert governance.QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V2 == (
        RUNTIME_RETRY_VERSION
    )
    assert governance.QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2 == (
        RUNTIME_RETRY_SHA256
    )
    source = governance.load_qwen38_feedback_model_source_lock_v4(
        SOURCE_V4,
        expected_file_sha256=governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4,
    )
    pricing = governance.load_qwen38_feedback_pricing_lock_v7(
        PRICING_V7,
        expected_file_sha256=governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7,
    )
    role = governance.load_qwen38_feedback_role_selection_v14(
        ROLE_V14,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
        ),
    )

    assert sha256_bytes(SOURCE_V4.read_bytes()) == (
        governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
    )
    assert sha256_bytes(PRICING_V7.read_bytes()) == (
        governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7
    )
    assert sha256_bytes(ROLE_V14.read_bytes()) == (
        governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14
    )
    assert source.source_lock_sha256 == governance.QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4
    assert pricing.pricing_lock_sha256 == (
        governance.QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7
    )
    assert role.selection_sha256 == governance.QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14

    assert source.wire_kind == "round3_primary_json_schema_v1"
    assert source.requested_response_format == "json_schema"
    assert source.requested_json_schema is True
    assert source.requested_json_schema_strict is True
    assert source.requested_json_schema_sha256 == (
        governance.QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    assert source.outer_orchestration_policy_version == RUNTIME_RETRY_VERSION
    assert source.outer_orchestration_policy_sha256 == RUNTIME_RETRY_SHA256
    assert source.structured_output_json_schema_used is True
    assert source.structured_output_strict is True
    assert source.cache_namespace == "feedback-evaluator-v14"
    assert source.result_schema_version == 8
    assert source.bound_artifact_schema_version == 2
    assert source.bound_artifact_policy_version == (
        "portfolio-s1-bound-feedback-round3-schema-v1"
    )
    assert source.fixed_selected_query_count == 240
    assert source.live_canary_selected_query_count == 12
    assert source.provider_call_ceiling == 15
    assert source.future_full_provider_call_ceiling == 243
    assert source.concurrency == 2
    assert source.full_run_live_authorized is False
    assert source.future_full_owner_budget_authorization_status == "not_granted"
    assert source.phase60_requires_new_owner_approval is True
    assert source.historical_feedback_outputs_imported == 0
    assert source.terminated_json_object_canary_outputs_imported == 0

    assert pricing.phase_counts == (12, 60, 120, 240)
    assert pricing.concurrency == 2
    assert pricing.selected_query_count == 240
    assert pricing.live_authorized_selected_query_count == 12
    assert pricing.live_authorized_phase_counts == (12,)
    assert pricing.provider_call_ceiling == 15
    assert pricing.future_full_provider_call_ceiling == 243
    assert pricing.global_retry_token_count == 3
    assert pricing.outer_orchestration_policy_version == RUNTIME_RETRY_VERSION
    assert pricing.outer_orchestration_policy_sha256 == RUNTIME_RETRY_SHA256
    assert pricing.provider_internal_max_attempts == 1
    assert pricing.max_lifetime_attempts_per_retried_query == 2
    assert pricing.maximum_reservation_cny == "6.923160000000"
    assert pricing.prior_cumulative_actual_cost_cny == "24.319500000000"
    assert pricing.cumulative_maximum_reservation_cny == "31.242660000000"
    assert pricing.technical_phase_hard_cap_cny == "10.000000000000"
    assert pricing.live_cumulative_hard_cap_cny == "34.319500000000"
    assert pricing.owner_budget_authorized_cap_cny == "10.000000000000"
    assert pricing.future_full_fresh_maximum_reservation_cny == (
        "112.155192000000"
    )
    assert pricing.future_full_cumulative_maximum_reservation_cny == (
        "136.474692000000"
    )
    assert pricing.future_full_technical_cumulative_hard_cap_cny == (
        "137.000000000000"
    )
    assert pricing.future_full_envelope_live_authorized is False
    assert pricing.future_full_owner_budget_authorization_status == "not_granted"
    assert pricing.full_run_and_retry_scope_owner_approved is False
    assert pricing.phase60_requires_new_owner_approval is True
    assert pricing.prior_actual_includes_terminated_json_object_canary is True
    assert pricing.terminated_json_object_canary_outputs_imported == 0
    assert Decimal(pricing.prior_cumulative_actual_cost_cny) + Decimal(
        pricing.maximum_reservation_cny
    ) == Decimal(pricing.cumulative_maximum_reservation_cny)
    assert Decimal(pricing.maximum_reservation_cny) < Decimal(
        pricing.technical_phase_hard_cap_cny
    ) == Decimal(pricing.owner_budget_authorized_cap_cny)
    assert Decimal(pricing.cumulative_maximum_reservation_cny) < Decimal(
        pricing.live_cumulative_hard_cap_cny
    )
    assert Decimal(pricing.prior_cumulative_actual_cost_cny) + Decimal(
        pricing.future_full_fresh_maximum_reservation_cny
    ) == Decimal(pricing.future_full_cumulative_maximum_reservation_cny)

    feedback = role.feedback_evaluator
    assert feedback["model_source_lock_file_sha256"] == sha256_bytes(
        SOURCE_V4.read_bytes()
    )
    assert feedback["pricing_lock_file_sha256"] == sha256_bytes(
        PRICING_V7.read_bytes()
    )
    assert feedback["transport_policy_version"] == (
        "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
    )
    assert feedback["transport_policy_sha256"] == (
        governance.ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    )
    assert feedback["outer_orchestration_policy_sha256"] == (
        governance.QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2
    )
    assert feedback["cache_namespace"] == "feedback-evaluator-v14"
    assert feedback["result_schema_version"] == 8
    assert feedback["bound_artifact_policy_version"] == (
        "portfolio-s1-bound-feedback-round3-schema-v1"
    )
    assert feedback["provider_call_ceiling"] == 15
    assert feedback["future_full_provider_call_ceiling"] == 243
    assert feedback["live_authorized_selected_query_count"] == 12
    assert feedback["prior_cumulative_actual_cost_cny"] == "24.319500000000"
    assert feedback["fresh_maximum_reservation_cny"] == "6.923160000000"
    assert feedback["cumulative_maximum_reservation_cny"] == "31.242660000000"
    assert feedback["fresh_stage_hard_cap_cny"] == "10.000000000000"
    assert feedback["future_full_cumulative_maximum_reservation_cny"] == (
        "136.474692000000"
    )
    assert feedback["future_full_technical_cumulative_hard_cap_cny"] == (
        "137.000000000000"
    )
    assert feedback["future_full_envelope_live_authorized"] is False
    assert feedback["future_full_owner_budget_authorization_status"] == (
        "not_granted"
    )
    assert feedback["phase60_requires_new_owner_approval"] is True
    assert feedback["historical_feedback_outputs_imported"] == 0
    assert feedback["terminated_json_object_canary_outputs_imported"] == 0


def test_round3_schema_budget_helper_enforces_cumulative_stop() -> None:
    assert (
        governance.require_qwen38_feedback_pre_call_budget_v4(
            estimated_input_tokens_including_images=20_000,
            provider_calls_already_reserved=14,
            committed_cumulative_cost_cny="31.242660000000",
        )
        == "0.461544000000"
    )
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="15-call provider ceiling",
    ):
        governance.require_qwen38_feedback_pre_call_budget_v4(
            estimated_input_tokens_including_images=1,
            provider_calls_already_reserved=15,
            committed_cumulative_cost_cny="24.319500000000",
        )
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="CNY10 fresh stage hard cap",
    ):
        governance.require_qwen38_feedback_pre_call_budget_v4(
            estimated_input_tokens_including_images=1,
            provider_calls_already_reserved=14,
            committed_cumulative_cost_cny="33.857956000001",
        )
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="cumulative committed cost is invalid",
    ):
        governance.require_qwen38_feedback_pre_call_budget_v4(
            estimated_input_tokens_including_images=1,
            provider_calls_already_reserved=0,
            committed_cumulative_cost_cny="24.319499999999",
        )


def test_round3_schema_loaders_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="source lock v4 file SHA-256 mismatch",
    ):
        governance.load_qwen38_feedback_model_source_lock_v4(
            SOURCE_V4,
            expected_file_sha256="0" * 64,
        )

    source_payload = json.loads(SOURCE_V4.read_text(encoding="utf-8"))
    source_payload["requested_response_format"] = "json_object"
    invalid_source = tmp_path / "source-v4-invalid.json"
    invalid_source.write_bytes(canonical_json_bytes(source_payload))
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="source lock v4 is invalid",
    ):
        governance.load_qwen38_feedback_model_source_lock_v4(
            invalid_source,
            expected_file_sha256=sha256_bytes(invalid_source.read_bytes()),
        )

    noncanonical_role = tmp_path / "role-v14-noncanonical.json"
    role_payload = json.loads(ROLE_V14.read_text(encoding="utf-8"))
    noncanonical_role.write_text(
        json.dumps(role_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="role selection v14 is not canonical JSON",
    ):
        governance.load_qwen38_feedback_role_selection_v14(
            noncanonical_role,
            expected_file_sha256=sha256_bytes(noncanonical_role.read_bytes()),
        )


def test_round3_schema_addition_preserves_v3_v6_v13_bytes_and_loaders() -> None:
    assert sha256_bytes(SOURCE_V3.read_bytes()) == (
        "eb075d4d09741a06e77050f79539bc85610323af21b160e36ee502be96c7c1a3"
    )
    assert sha256_bytes(PRICING_V6.read_bytes()) == (
        "e23cc28625cde23accc3f88d456acfa20a5ba4b2f351301a04524ba7e418487c"
    )
    assert sha256_bytes(ROLE_V13.read_bytes()) == (
        "f8444da36fc65dbd63784b4668fda3c39ce79e414dbb9a4c5b053504d52fca40"
    )
    assert governance.load_qwen38_feedback_model_source_lock_v3(
        SOURCE_V3,
        expected_file_sha256=governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3,
    ).source_lock_sha256 == governance.QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3
    assert governance.load_qwen38_feedback_pricing_lock_v6(
        PRICING_V6,
        expected_file_sha256=governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6,
    ).pricing_lock_sha256 == governance.QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6
    assert governance.load_qwen38_feedback_role_selection_v13(
        ROLE_V13,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
        ),
    ).selection_sha256 == governance.QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13
