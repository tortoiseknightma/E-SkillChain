from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import pytest

import skillchain.evaluation.portfolio_s1_qwen_governance as governance
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1 as RUNTIME_TRANSPORT_SHA256,
)
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1 as RUNTIME_TRANSPORT_VERSION,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_v1 import (
    ROUND3_RETRY_POLICY_SHA256_V1 as RUNTIME_RETRY_SHA256,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    PortfolioS1QwenFeedbackGovernanceError,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
SOURCE_V2 = ROOT / "specs/authoring/qwen3.8-max-feedback-source-lock-v2.json"
SOURCE_V3 = ROOT / governance.QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3
PRICING_V5 = ROOT / "specs/authoring/price-qwen3.8-max-feedback-v5.json"
PRICING_V6 = ROOT / governance.QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6
ROLE_V12 = ROOT / "specs/authoring/model-role-selection-v12.json"
ROLE_V13 = ROOT / governance.QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V13


def test_round3_governance_triad_is_canonical_and_forward_only() -> None:
    assert governance.ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1 == (
        RUNTIME_TRANSPORT_VERSION
    )
    assert governance.ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1 == (
        RUNTIME_TRANSPORT_SHA256
    )
    assert governance.QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V1 == (
        RUNTIME_RETRY_SHA256
    )
    source = governance.load_qwen38_feedback_model_source_lock_v3(
        SOURCE_V3,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
        ),
    )
    pricing = governance.load_qwen38_feedback_pricing_lock_v6(
        PRICING_V6,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
        ),
    )
    role = governance.load_qwen38_feedback_role_selection_v13(
        ROLE_V13,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
        ),
    )

    assert sha256_bytes(SOURCE_V3.read_bytes()) == (
        governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
    )
    assert sha256_bytes(PRICING_V6.read_bytes()) == (
        governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6
    )
    assert sha256_bytes(ROLE_V13.read_bytes()) == (
        governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13
    )
    assert source.source_lock_sha256 == (
        governance.QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3
    )
    assert pricing.pricing_lock_sha256 == (
        governance.QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6
    )
    assert role.selection_sha256 == (
        governance.QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13
    )

    assert source.model == "qwen3.8-max"
    assert source.model_identity_kind == "moving_alias"
    assert source.frozen_snapshot is False
    assert source.requested_response_format == "json_object"
    assert source.structured_output_response_format == "json_object"
    assert source.requested_json_schema is False
    assert source.structured_output_json_schema_used is False
    assert source.wire_kind == "round3_primary_json_object_v1"
    assert source.transport_policy_sha256 == (
        "7f9e91479e024ac1ade536d57f30ab1f7bd8f7efdd9865e21cc9fd2776769313"
    )
    assert source.thinking_budget == 2048
    assert source.max_completion_tokens == 6144
    assert source.provider_internal_max_attempts == 1
    assert source.cache_namespace == "feedback-evaluator-v13"
    assert source.result_schema_version == 7
    assert source.bound_artifact_schema_version == 1
    assert source.bound_artifact_policy_version == (
        "portfolio-s1-bound-feedback-round3-v1"
    )
    assert source.image_source_membership_ancestry_only is True
    assert source.old_remote_auth_does_not_authorize_round3_transport is True
    assert source.live_call_authority is True

    assert pricing.phase_counts == (12, 60, 120, 240)
    assert pricing.selected_query_count == 240
    assert pricing.provider_call_ceiling == 243
    assert pricing.global_retry_token_count == 3
    assert pricing.provider_internal_max_attempts == 1
    assert pricing.maximum_reservation_cny == "112.155192000000"
    assert pricing.prior_cumulative_actual_cost_cny == "22.764432000000"
    assert pricing.cumulative_maximum_reservation_cny == "134.919624000000"
    assert pricing.technical_phase_hard_cap_cny == "135.000000000000"
    assert pricing.owner_budget_authorized_cap_cny == "150.000000000000"
    assert Decimal(pricing.prior_cumulative_actual_cost_cny) + Decimal(
        pricing.maximum_reservation_cny
    ) == Decimal(pricing.cumulative_maximum_reservation_cny)
    assert Decimal(pricing.cumulative_maximum_reservation_cny) < Decimal(
        pricing.technical_phase_hard_cap_cny
    ) <= Decimal(pricing.owner_budget_authorized_cap_cny)
    assert pricing.live_provider_calls_authorized is True

    feedback = role.feedback_evaluator
    assert feedback["model_source_lock_file_sha256"] == sha256_bytes(
        SOURCE_V3.read_bytes()
    )
    assert feedback["pricing_lock_file_sha256"] == sha256_bytes(
        PRICING_V6.read_bytes()
    )
    assert feedback["requested_response_format"] == "json_object"
    assert feedback["requested_json_schema"] is False
    assert "requested_json_schema_sha256" not in feedback
    assert feedback["provider_call_ceiling"] == 243
    assert feedback["prior_cumulative_actual_cost_cny"] == "22.764432000000"
    assert feedback["cumulative_maximum_reservation_cny"] == "134.919624000000"
    assert feedback["technical_cumulative_hard_cap_cny"] == "135.000000000000"
    assert feedback["owner_authorized_budget_ceiling_cny"] == "150.000000000000"
    assert feedback["old_remote_auth_does_not_authorize_round3_transport"] is True
    assert feedback["live_call_authority"] is True


def test_round3_loaders_fail_closed_on_digest_schema_and_canonical_drift(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="source lock v3 file SHA-256 mismatch",
    ):
        governance.load_qwen38_feedback_model_source_lock_v3(
            SOURCE_V3, expected_file_sha256="0" * 64
        )

    source_payload = json.loads(SOURCE_V3.read_text(encoding="utf-8"))
    source_payload["requested_response_format"] = "json_schema"
    invalid_source = tmp_path / "source-v3-invalid.json"
    invalid_source.write_bytes(canonical_json_bytes(source_payload))
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="source lock v3 is invalid",
    ):
        governance.load_qwen38_feedback_model_source_lock_v3(
            invalid_source,
            expected_file_sha256=sha256_bytes(invalid_source.read_bytes()),
        )

    noncanonical_role = tmp_path / "role-v13-noncanonical.json"
    role_payload = json.loads(ROLE_V13.read_text(encoding="utf-8"))
    noncanonical_role.write_text(
        json.dumps(role_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        PortfolioS1QwenFeedbackGovernanceError,
        match="role selection v13 is not canonical JSON",
    ):
        governance.load_qwen38_feedback_role_selection_v13(
            noncanonical_role,
            expected_file_sha256=sha256_bytes(noncanonical_role.read_bytes()),
        )


def test_round3_addition_preserves_historical_lock_bytes_and_loaders() -> None:
    assert sha256_bytes(SOURCE_V2.read_bytes()) == (
        governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2
    )
    assert sha256_bytes(PRICING_V5.read_bytes()) == (
        governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5
    )
    assert sha256_bytes(ROLE_V12.read_bytes()) == (
        governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12
    )
    assert governance.load_qwen38_feedback_model_source_lock_v2(
        SOURCE_V2,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2
        ),
    ).source_lock_sha256 == governance.QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2
    assert governance.load_qwen38_feedback_pricing_lock_v5(
        PRICING_V5,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5
        ),
    ).pricing_lock_sha256 == governance.QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5
    assert governance.load_qwen38_feedback_role_selection_v12(
        ROLE_V12,
        expected_file_sha256=(
            governance.QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12
        ),
    ).selection_sha256 == governance.QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12
