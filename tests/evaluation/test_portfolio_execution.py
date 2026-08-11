from __future__ import annotations

from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from skillchain.evaluation import portfolio_execution as portfolio_execution_module
from skillchain.evaluation.portfolio_execution import (
    PORTFOLIO_ACTIVE_FINAL_JUDGE_MODEL,
    PORTFOLIO_ACTIVE_FINAL_JUDGE_PROVIDER,
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_BUDGET_POLICY_VERSION_V1,
    PORTFOLIO_KIMI_FINAL_JUDGE_RESERVE_OUTPUT_TOKENS,
    PORTFOLIO_KIMI_MODEL,
    PORTFOLIO_QWEN_LEGACY_ROUTE_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_MODEL,
    PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS,
    PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS,
    PortfolioAttemptLimitError,
    PortfolioAttemptReceipt,
    PortfolioBudgetCallIdentity,
    PortfolioBudgetDuplicateCallError,
    PortfolioBudgetError,
    PortfolioBudgetExceededError,
    PortfolioBudgetForfeitError,
    PortfolioBudgetSettlementError,
    PortfolioProviderPricingProfile,
    ProviderPreResponseCircuitBreaker,
    ValidatedRouteIdentity,
    calculate_portfolio_settled_usage_cost_cny,
    create_retryable_attempt_receipt,
    forfeit_portfolio_provider_call,
    initialize_portfolio_budget_ledger,
    load_portfolio_budget_ledger,
    load_query_attempt_receipts,
    make_portfolio_budget_call_identity,
    make_portfolio_provider_pricing_profile,
    open_portfolio_budget_ledger_session,
    portfolio_budget_settled_cost_cny,
    require_retryable_attempt_available,
    reserve_portfolio_provider_call,
    settle_portfolio_provider_call_success,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


@pytest.mark.parametrize(
    ("failure_subtype", "captured_count", "expected_calls"),
    (
        ("provider_pre_response", 0, 1),
        ("provider_pre_response_route", 0, 1),
        ("provider_pre_response_action", 1, 2),
        ("provider_pre_response_provider_error", 1, 2),
        ("provider_pre_response_timeout", 1, 2),
        ("route_contract_invalid_json", 1, 1),
        ("orphaned_provider_call", 1, 1),
    ),
)
def test_attempt_provider_call_count_adds_only_pre_response_terminal_call(
    failure_subtype: str,
    captured_count: int,
    expected_calls: int,
) -> None:
    receipt = SimpleNamespace(
        failure_subtype=failure_subtype,
        captured_provider_response_count=captured_count,
    )

    assert (
        portfolio_execution_module.portfolio_attempt_provider_call_count(receipt)
        == expected_calls
    )


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64


def _budget_identity(
    *,
    stage: str = "assistant_action",
    call_index: int = 1,
    wire: str | None = _SHA_E,
    shard_id: str = "00-batch-s1",
):
    return make_portfolio_budget_call_identity(
        matrix_run_id="portfolio-mini",
        shard_id=shard_id,
        config="s1",
        query_id="query-001",
        instance_sha256=_SHA_A,
        request_sha256=_SHA_B,
        wire_request_sha256=wire,
        stage=stage,
        attempt_index=1,
        call_index=call_index,
    )


def _legacy_kimi_budget_identity(*, shard_id: str = "01-batch-s1"):
    active = _budget_identity(
        stage="final_judge",
        wire=None,
        shard_id=shard_id,
    )
    unsigned = active.model_dump(mode="json", exclude={"identity_sha256"})
    unsigned.update(provider="kimi", model="kimi-k2.6")
    return PortfolioBudgetCallIdentity.model_validate(
        {
            **unsigned,
            "identity_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        },
        strict=True,
    )


def _write_legacy_v1_budget_ledger(tmp_path, *, settled: bool):
    """Write one canonical v1 ledger without passing through active-v2 APIs."""

    authority_unsigned = {
        "schema_version": 1,
        "kind": "portfolio-budget-authority",
        "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION_V1,
        "matrix_run_id": "portfolio-mini",
        "currency": "CNY",
        "phase_cap_cny": "1.000000000000",
        "prior_observed_cost_cny": "0.100000000000",
        "accounting_policy": "settled_actual_plus_unresolved_reserve",
        "exception_policy": "retain_full_reserve",
    }
    authority = {
        **authority_unsigned,
        "authority_sha256": sha256_bytes(canonical_json_bytes(authority_unsigned)),
    }
    identity = _budget_identity()
    profile = make_portfolio_provider_pricing_profile(identity)
    reserved_cost = portfolio_execution_module.calculate_portfolio_usage_cost_cny(
        input_tokens=profile.provider_max_input_tokens,
        output_tokens=profile.max_output_tokens,
        input_cny_per_million_tokens=(profile.reserve_input_cny_per_million_tokens),
        output_cny_per_million_tokens=(profile.reserve_output_cny_per_million_tokens),
    )
    reservation_unsigned = {
        "schema_version": 1,
        "kind": "portfolio-budget-reservation",
        "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION_V1,
        "authority_sha256": authority["authority_sha256"],
        "ledger_event_index": 1,
        "previous_event_sha256": None,
        "identity": identity.model_dump(mode="json"),
        "pricing_profile": profile.model_dump(mode="json"),
        "reserved_cost_cny": format(reserved_cost, ".12f"),
        "disposition": "reserved_before_provider_call",
    }
    reservation = {
        **reservation_unsigned,
        "reservation_sha256": sha256_bytes(canonical_json_bytes(reservation_unsigned)),
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "budget-authority.json").write_bytes(canonical_json_bytes(authority))
    events = tmp_path / "events"
    events.mkdir()
    (events / "00000001.json").write_bytes(canonical_json_bytes(reservation))
    if settled:
        actual_cost = (
            portfolio_execution_module.calculate_portfolio_settled_usage_cost_cny(
                profile,
                input_tokens=10_000,
                output_tokens=100,
            )
        )
        settlement_unsigned = {
            "schema_version": 1,
            "kind": "portfolio-budget-settlement",
            "policy_version": PORTFOLIO_BUDGET_POLICY_VERSION_V1,
            "authority_sha256": authority["authority_sha256"],
            "ledger_event_index": 2,
            "previous_event_sha256": reservation["reservation_sha256"],
            "reservation_sha256": reservation["reservation_sha256"],
            "identity_sha256": identity.identity_sha256,
            "actual_input_tokens": 10_000,
            "actual_output_tokens": 100,
            "actual_cost_cny": format(actual_cost, ".12f"),
            "provider_request_id": "request-legacy-v1",
            "response_sha256": _SHA_F,
            "outcome": "success",
            "disposition": "settled_from_actual_usage",
        }
        settlement = {
            **settlement_unsigned,
            "settlement_sha256": sha256_bytes(
                canonical_json_bytes(settlement_unsigned)
            ),
        }
        (events / "00000002.json").write_bytes(canonical_json_bytes(settlement))
    return reservation


def _create(root, breaker, *, subtype: str = "provider_pre_response"):
    return create_retryable_attempt_receipt(
        root,
        breaker=breaker,
        matrix_run_id="portfolio-mini",
        shard_id="00-batch-s1",
        config="s1",
        instance_sha256=_SHA_A,
        query_id="query-001",
        query_ordinal=7,
        request_sha256=_SHA_B,
        failure_stage="assistant_action",
        failure_subtype=subtype,
        circuit_id="assistant:qwen:qwen3-vl-flash",
        captured_provider_response_count=1,
        captured_input_tokens=11,
        captured_output_tokens=3,
        assistant_response_sha256=_SHA_C,
        assistant_receipt_sha256=_SHA_D,
        forfeited_reservation_sha256="0" * 64,
        budget_forfeit_sha256="1" * 64,
        validated_route_identity=ValidatedRouteIdentity(
            selected_capability="product.exact_match",
            skill_slug="exact-match",
            route_trace_sha256=_SHA_E,
            bank_sha256=_SHA_A,
        ),
    )


def test_attempt_receipts_are_create_only_ordered_and_bound(tmp_path) -> None:
    breaker = ProviderPreResponseCircuitBreaker()

    first, first_path = _create(tmp_path, breaker)
    # A resumed shard runs in a fresh process, so the second attempt must
    # reconstruct the first failure from its create-only receipt.
    second, second_path = _create(tmp_path, ProviderPreResponseCircuitBreaker())

    assert first.attempt_index == 1
    assert first.circuit_open is False
    assert second.attempt_index == 2
    assert second.circuit_open is True
    assert first_path.name == "0007-attempt-01.json"
    assert second_path.name == "0007-attempt-02.json"
    assert load_query_attempt_receipts(
        tmp_path,
        query_ordinal=7,
        query_id="query-001",
        instance_sha256=_SHA_A,
    ) == (first, second)
    with pytest.raises(PortfolioAttemptLimitError):
        require_retryable_attempt_available(
            tmp_path,
            query_ordinal=7,
            query_id="query-001",
            instance_sha256=_SHA_A,
        )
    with pytest.raises(PortfolioAttemptLimitError):
        _create(tmp_path, breaker)


def test_attempt_receipt_never_persists_raw_exception_text(tmp_path) -> None:
    breaker = ProviderPreResponseCircuitBreaker()
    receipt, path = create_retryable_attempt_receipt(
        tmp_path,
        breaker=breaker,
        matrix_run_id="portfolio-mini",
        shard_id="00-batch-s1",
        config="s1",
        instance_sha256=_SHA_A,
        query_id="query-001",
        query_ordinal=7,
        request_sha256=_SHA_B,
        failure_stage="shared_route",
        failure_subtype="provider_pre_response",
        circuit_id="assistant:qwen:qwen3-vl-flash",
        forfeited_reservation_sha256="0" * 64,
        budget_forfeit_sha256="1" * 64,
        exception_type="AssistantProviderPreResponseError",
    )

    raw = json.loads(path.read_bytes())
    assert raw["exception_type"] == "AssistantProviderPreResponseError"
    assert "exception_message" not in raw
    assert "raw_exception" not in raw
    assert "credential" not in path.read_text(encoding="utf-8").lower()
    assert PortfolioAttemptReceipt.model_validate(raw, strict=True) == receipt


def test_legacy_v1_route_receipt_without_raw_evidence_remains_replayable(
    tmp_path,
) -> None:
    current, _ = _create(tmp_path, ProviderPreResponseCircuitBreaker())
    payload = current.model_dump(mode="json")
    payload.update(
        {
            "policy_version": "portfolio-shard-attempt-v1",
            "failure_stage": "assistant_route",
            "failure_subtype": "route_contract_invalid_json",
            "circuit_id": "assistant_route_contract:qwen:model",
            "validated_route_identity": None,
        }
    )
    payload.pop("receipt_sha256")
    payload.pop("forfeited_reservation_sha256")
    payload.pop("budget_forfeit_sha256")
    replayed = PortfolioAttemptReceipt.model_validate(
        {
            **payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )

    assert replayed.policy_version == "portfolio-shard-attempt-v1"
    assert replayed.route_call_evidence is None


@pytest.mark.parametrize(
    "missing_field",
    [
        "forfeited_reservation_sha256",
        "budget_forfeit_sha256",
        "both",
    ],
)
def test_active_provider_pre_response_receipt_requires_complete_forfeit_binding(
    tmp_path,
    missing_field: str,
) -> None:
    receipt, _ = _create(tmp_path, ProviderPreResponseCircuitBreaker())
    payload = receipt.model_dump(mode="json")
    payload.pop("receipt_sha256")
    if missing_field in {"forfeited_reservation_sha256", "both"}:
        payload.pop("forfeited_reservation_sha256")
    if missing_field in {"budget_forfeit_sha256", "both"}:
        payload.pop("budget_forfeit_sha256")

    with pytest.raises(ValueError, match="forfeit"):
        PortfolioAttemptReceipt.model_validate(
            {
                **payload,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )


def test_active_orphan_receipt_counts_forfeited_uncaptured_call(tmp_path) -> None:
    receipt, _ = create_retryable_attempt_receipt(
        tmp_path,
        breaker=ProviderPreResponseCircuitBreaker(),
        matrix_run_id="portfolio-mini",
        shard_id="00-batch-s1",
        config="s1",
        instance_sha256=_SHA_A,
        query_id="query-001",
        query_ordinal=7,
        request_sha256=_SHA_B,
        failure_stage="assistant_action",
        failure_subtype="orphaned_provider_call",
        circuit_id="orphaned:qwen:qwen3-vl-flash",
        forfeited_reservation_sha256="0" * 64,
        budget_forfeit_sha256="1" * 64,
        exception_type="PortfolioBudgetOrphanedCallError",
    )

    assert (
        portfolio_execution_module.portfolio_attempt_provider_call_count(receipt) == 1
    )


def test_provider_circuits_are_scoped_and_reset_only_by_valid_response() -> None:
    breaker = ProviderPreResponseCircuitBreaker()

    assert breaker.record_retryable_failure("assistant:qwen:model") == (1, False)
    assert breaker.record_retryable_failure("judge:kimi:model") == (1, False)
    assert breaker.record_retryable_failure("assistant:qwen:model") == (2, True)

    breaker.record_valid_response("assistant:qwen:model")
    assert breaker.record_retryable_failure("assistant:qwen:model") == (1, False)
    assert breaker.record_retryable_failure("judge:kimi:model") == (2, True)


def test_attempt_receipt_rejects_unordered_filename_gap(tmp_path) -> None:
    breaker = ProviderPreResponseCircuitBreaker()
    _, first_path = _create(tmp_path, breaker)
    gap_path = first_path.with_name("0007-attempt-02.json")
    first_path.rename(gap_path)

    with pytest.raises(ValueError, match="order or binding mismatch"):
        load_query_attempt_receipts(
            tmp_path,
            query_ordinal=7,
            query_id="query-001",
            instance_sha256=_SHA_A,
        )


def test_budget_reservation_is_create_only_and_success_settles_actual_usage(
    tmp_path,
) -> None:
    authority, authority_path = initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
        prior_observed_cost_cny=Decimal("0.1"),
    )
    identity = _budget_identity()
    reservation, reservation_path = reserve_portfolio_provider_call(
        tmp_path,
        identity=identity,
    )

    assert authority_path.read_bytes().endswith(b"\n")
    assert authority.phase_cap_cny == Decimal("1.000000000000")
    assert identity.model == PORTFOLIO_QWEN_MODEL
    assert identity.instance_sha256 == _SHA_A
    assert identity.request_sha256 == _SHA_B
    assert identity.wire_request_sha256 == _SHA_E
    # Reserve uses the 258,048-token provider ceiling and highest 0.6/6 tier,
    # not the 32,768-token execution check or the caller's expected usage.
    assert reservation.reserved_cost_cny == Decimal("0.179404800000")
    assert reservation.pricing_profile.max_input_tokens == 32_768
    assert reservation.pricing_profile.provider_max_input_tokens == 258_048
    assert reservation_path.name == "00000001.json"

    settlement, settlement_path = settle_portfolio_provider_call_success(
        tmp_path,
        reservation_sha256=reservation.reservation_sha256,
        actual_input_tokens=10_000,
        actual_output_tokens=100,
        provider_request_id="request-qwen-001",
        response_sha256=_SHA_F,
    )
    assert settlement.actual_cost_cny == Decimal("0.001650000000")
    assert settlement.provider_request_id == "request-qwen-001"
    assert settlement.response_sha256 == _SHA_F
    assert settlement_path.name == "00000002.json"

    recovered = load_portfolio_budget_ledger(tmp_path)
    assert recovered.unresolved_reservations == ()
    assert recovered.settled_actual_cost_cny == Decimal("0.001650000000")
    assert recovered.accountable_cost_cny == Decimal("0.101650000000")
    assert recovered.remaining_cost_cny == Decimal("0.898350000000")
    with pytest.raises(PortfolioBudgetDuplicateCallError):
        reserve_portfolio_provider_call(tmp_path, identity=identity)
    with pytest.raises(PortfolioBudgetSettlementError, match="already"):
        settle_portfolio_provider_call_success(
            tmp_path,
            reservation_sha256=reservation.reservation_sha256,
            actual_input_tokens=10_000,
            actual_output_tokens=100,
            provider_request_id="request-qwen-001",
            response_sha256=_SHA_F,
        )


@pytest.mark.parametrize(
    "reason",
    [
        "provider_call_ended_without_captured_response",
        "orphan_recovered_after_owner_exit",
    ],
)
def test_budget_forfeit_closes_reservation_without_releasing_full_reserve(
    tmp_path,
    reason: str,
) -> None:
    authority, _ = initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
        prior_observed_cost_cny=Decimal("0.1"),
    )
    reservation, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(),
    )
    reserved_state = load_portfolio_budget_ledger(tmp_path)

    forfeit, path = forfeit_portfolio_provider_call(
        tmp_path,
        reservation_sha256=reservation.reservation_sha256,
        reason=reason,
    )
    recovered = load_portfolio_budget_ledger(tmp_path)

    assert authority.policy_version == PORTFOLIO_BUDGET_POLICY_VERSION
    assert authority.accounting_policy == (
        "settled_actual_plus_forfeited_full_reserve_plus_unresolved_reserve"
    )
    assert authority.exception_policy == (
        "append_full_reserve_forfeit_without_captured_response"
    )
    assert path.name == "00000002.json"
    assert forfeit.reason == reason
    assert forfeit.reservation_sha256 == reservation.reservation_sha256
    assert forfeit.identity_sha256 == reservation.identity.identity_sha256
    assert forfeit.forfeited_cost_cny == reservation.reserved_cost_cny
    assert forfeit.captured_provider_response is False
    assert forfeit.actual_usage_status == "unknown"
    assert recovered.reservations == (reservation,)
    assert recovered.settlements == ()
    assert recovered.forfeits == (forfeit,)
    assert recovered.unresolved_reservations == ()
    assert recovered.settled_actual_cost_cny == Decimal("0.000000000000")
    assert recovered.forfeited_reserved_cost_cny == Decimal("0.179404800000")
    assert recovered.unresolved_reserved_cost_cny == Decimal("0.000000000000")
    assert recovered.accountable_cost_cny == reserved_state.accountable_cost_cny
    assert recovered.remaining_cost_cny == reserved_state.remaining_cost_cny


def test_budget_settlement_and_forfeit_are_mutually_exclusive(tmp_path) -> None:
    settled_root = tmp_path / "settled"
    initialize_portfolio_budget_ledger(
        settled_root,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
    )
    settled_reservation, _ = reserve_portfolio_provider_call(
        settled_root,
        identity=_budget_identity(),
    )
    settle_portfolio_provider_call_success(
        settled_root,
        reservation_sha256=settled_reservation.reservation_sha256,
        actual_input_tokens=10_000,
        actual_output_tokens=100,
        provider_request_id="request-settled-first",
        response_sha256=_SHA_F,
    )
    with pytest.raises(PortfolioBudgetForfeitError, match="terminal outcome"):
        forfeit_portfolio_provider_call(
            settled_root,
            reservation_sha256=settled_reservation.reservation_sha256,
            reason="provider_call_ended_without_captured_response",
        )

    forfeited_root = tmp_path / "forfeited"
    initialize_portfolio_budget_ledger(
        forfeited_root,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
    )
    forfeited_reservation, _ = reserve_portfolio_provider_call(
        forfeited_root,
        identity=_budget_identity(),
    )
    forfeit_portfolio_provider_call(
        forfeited_root,
        reservation_sha256=forfeited_reservation.reservation_sha256,
        reason="provider_call_ended_without_captured_response",
    )
    with pytest.raises(PortfolioBudgetSettlementError, match="terminal outcome"):
        settle_portfolio_provider_call_success(
            forfeited_root,
            reservation_sha256=forfeited_reservation.reservation_sha256,
            actual_input_tokens=10_000,
            actual_output_tokens=100,
            provider_request_id="request-forfeited-first",
            response_sha256=_SHA_F,
        )
    with pytest.raises(PortfolioBudgetForfeitError, match="terminal outcome"):
        forfeit_portfolio_provider_call(
            forfeited_root,
            reservation_sha256=forfeited_reservation.reservation_sha256,
            reason="orphan_recovered_after_owner_exit",
        )


def test_budget_forfeit_api_rejects_unfrozen_reason_without_appending(tmp_path) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
    )
    reservation, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(),
    )

    with pytest.raises(ValueError, match="frozen Portfolio forfeit reason"):
        forfeit_portfolio_provider_call(
            tmp_path,
            reservation_sha256=reservation.reservation_sha256,
            reason="release_reserve",
        )

    recovered = load_portfolio_budget_ledger(tmp_path)
    assert recovered.forfeits == ()
    assert recovered.unresolved_reservations == (reservation,)
    assert recovered.last_event_index == 1


def test_settled_cost_projection_uses_per_call_tiers_and_identity(tmp_path) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("5"),
        prior_observed_cost_cny=Decimal("0"),
    )
    qwen, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(),
    )
    qwen_settlement, _ = settle_portfolio_provider_call_success(
        tmp_path,
        reservation_sha256=qwen.reservation_sha256,
        actual_input_tokens=32_500,
        actual_output_tokens=10,
        provider_request_id="request-qwen-tier-2",
        response_sha256=_SHA_F,
    )
    judge, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_legacy_kimi_budget_identity(),
        final_judge_max_output_tokens=(
            PORTFOLIO_KIMI_FINAL_JUDGE_RESERVE_OUTPUT_TOKENS
        ),
    )
    judge_settlement, _ = settle_portfolio_provider_call_success(
        tmp_path,
        reservation_sha256=judge.reservation_sha256,
        actual_input_tokens=100,
        actual_output_tokens=10,
        provider_request_id="request-kimi-001",
        response_sha256=_SHA_E,
    )
    state = load_portfolio_budget_ledger(tmp_path)

    assert qwen_settlement.actual_cost_cny == Decimal("0.009780000000")
    assert judge_settlement.actual_cost_cny == Decimal("0.000920000000")
    assert portfolio_budget_settled_cost_cny(state) == Decimal("0.010700000000")
    assert portfolio_budget_settled_cost_cny(
        state,
        shard_ids={"00-batch-s1"},
    ) == Decimal("0.009780000000")
    assert portfolio_budget_settled_cost_cny(
        state,
        stages={"final_judge"},
    ) == Decimal("0.000920000000")


def test_budget_restart_keeps_exception_reserve_and_prechecks_cap(tmp_path) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("0.35"),
        prior_observed_cost_cny=Decimal("0.1"),
    )
    first, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(call_index=1),
    )
    # Simulate a process/provider exception by intentionally writing no
    # settlement. A fresh load must recover and fully charge the reserve.
    recovered = load_portfolio_budget_ledger(tmp_path)
    assert recovered.unresolved_reservations == (first,)
    assert recovered.unresolved_reserved_cost_cny == Decimal("0.179404800000")
    assert recovered.accountable_cost_cny == Decimal("0.279404800000")

    event_count = len(tuple((tmp_path / "events").glob("*.json")))
    with pytest.raises(PortfolioBudgetExceededError, match="not authorized"):
        reserve_portfolio_provider_call(
            tmp_path,
            identity=_budget_identity(call_index=2),
        )
    assert len(tuple((tmp_path / "events").glob("*.json"))) == event_count


@pytest.mark.parametrize("settled", [False, True])
def test_legacy_v1_budget_ledger_remains_readable_but_is_read_only(
    tmp_path,
    settled: bool,
) -> None:
    reservation = _write_legacy_v1_budget_ledger(tmp_path, settled=settled)

    recovered = load_portfolio_budget_ledger(tmp_path)

    assert recovered.authority.policy_version == PORTFOLIO_BUDGET_POLICY_VERSION_V1
    assert len(recovered.reservations) == 1
    assert len(recovered.settlements) == int(settled)
    assert recovered.forfeits == ()
    assert recovered.forfeited_reserved_cost_cny == Decimal("0.000000000000")
    assert len(recovered.unresolved_reservations) == int(not settled)
    with pytest.raises(PortfolioBudgetError, match="read-only"):
        reserve_portfolio_provider_call(
            tmp_path,
            identity=_budget_identity(call_index=2),
        )
    with pytest.raises(PortfolioBudgetError, match="read-only"):
        forfeit_portfolio_provider_call(
            tmp_path,
            reservation_sha256=reservation["reservation_sha256"],
            reason="orphan_recovered_after_owner_exit",
        )
    if not settled:
        with pytest.raises(PortfolioBudgetError, match="read-only"):
            settle_portfolio_provider_call_success(
                tmp_path,
                reservation_sha256=reservation["reservation_sha256"],
                actual_input_tokens=10_000,
                actual_output_tokens=100,
                provider_request_id="request-v1-read-only",
                response_sha256=_SHA_F,
            )


def test_budget_session_appends_from_validated_prefix_without_full_replay(
    monkeypatch,
    tmp_path,
) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
    )
    session = open_portfolio_budget_ledger_session(tmp_path)

    def reject_full_replay(_root):
        raise AssertionError("active session must not fully replay the ledger")

    monkeypatch.setattr(
        portfolio_execution_module,
        "load_portfolio_budget_ledger",
        reject_full_replay,
    )
    reservation, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(),
    )
    settlement, _ = settle_portfolio_provider_call_success(
        tmp_path,
        reservation_sha256=reservation.reservation_sha256,
        actual_input_tokens=10_000,
        actual_output_tokens=100,
        provider_request_id="request-qwen-session",
        response_sha256=_SHA_F,
    )

    assert session.state.last_event_index == 2
    assert session.state.settlements == (settlement,)
    assert session.state.unresolved_reservations == ()
    assert session.state.accountable_cost_cny == Decimal("0.001650000000")


def test_budget_session_accepts_forfeit_without_full_replay(
    monkeypatch,
    tmp_path,
) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
    )
    session = open_portfolio_budget_ledger_session(tmp_path)

    def reject_full_replay(_root):
        raise AssertionError("active session must not fully replay the ledger")

    monkeypatch.setattr(
        portfolio_execution_module,
        "load_portfolio_budget_ledger",
        reject_full_replay,
    )
    reservation, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(),
    )
    state_before = session.state
    forfeit, _ = forfeit_portfolio_provider_call(
        tmp_path,
        reservation_sha256=reservation.reservation_sha256,
        reason="provider_call_ended_without_captured_response",
    )

    assert session.state.last_event_index == 2
    assert session.state.forfeits == (forfeit,)
    assert session.state.settlements == ()
    assert session.state.unresolved_reservations == ()
    assert session.state.forfeited_reserved_cost_cny == Decimal("0.179404800000")
    assert session.state.accountable_cost_cny == state_before.accountable_cost_cny
    assert session.state.remaining_cost_cny == state_before.remaining_cost_cny


def test_budget_profiles_are_frozen_and_qwen_settlement_uses_input_tiers() -> None:
    action = _budget_identity(stage="assistant_action")
    qwen_profile = make_portfolio_provider_pricing_profile(action)

    assert calculate_portfolio_settled_usage_cost_cny(
        qwen_profile,
        input_tokens=32_000,
        output_tokens=1,
    ) == Decimal("0.004801500000")
    assert calculate_portfolio_settled_usage_cost_cny(
        qwen_profile,
        input_tokens=32_001,
        output_tokens=1,
    ) == Decimal("0.009603300000")
    assert calculate_portfolio_settled_usage_cost_cny(
        qwen_profile,
        input_tokens=128_001,
        output_tokens=1,
    ) == Decimal("0.076806600000")
    with pytest.raises(ValueError, match="Qwen stages"):
        make_portfolio_provider_pricing_profile(
            action,
            final_judge_max_output_tokens=8_192,
        )

    route_profile = make_portfolio_provider_pricing_profile(
        _budget_identity(stage="shared_route")
    )
    assert route_profile.profile_id == "qwen-shared-route-v2"
    assert route_profile.max_output_tokens == PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
    assert route_profile.max_output_tokens == 512
    assert PORTFOLIO_QWEN_ROUTE_REQUEST_MAX_OUTPUT_TOKENS == 64
    assert route_profile.reserve_output_cny_per_million_tokens == Decimal(
        "6.000000000000"
    )

    legacy_unsigned = {
        "schema_version": 1,
        "kind": "portfolio-provider-pricing-profile",
        "profile_id": "qwen-shared-route-v1",
        "provider": "qwen",
        "model": PORTFOLIO_QWEN_MODEL,
        "stage": "shared_route",
        "max_input_tokens": 32_768,
        "max_output_tokens": PORTFOLIO_QWEN_LEGACY_ROUTE_MAX_OUTPUT_TOKENS,
        "provider_max_input_tokens": 258_048,
        "provider_max_output_tokens": None,
        "reserve_input_cny_per_million_tokens": "0.600000000000",
        "reserve_output_cny_per_million_tokens": "6.000000000000",
        "settlement_pricing_policy": "qwen_input_length_tiers_v1",
    }
    legacy_profile = PortfolioProviderPricingProfile.model_validate_json(
        canonical_json_bytes(
            {
                **legacy_unsigned,
                "profile_sha256": sha256_bytes(canonical_json_bytes(legacy_unsigned)),
            }
        ),
        strict=True,
    )
    assert legacy_profile.profile_id == "qwen-shared-route-v1"
    assert legacy_profile.max_output_tokens == 128

    active_judge = _budget_identity(stage="final_judge")
    assert (active_judge.provider, active_judge.model) == (
        PORTFOLIO_ACTIVE_FINAL_JUDGE_PROVIDER,
        PORTFOLIO_ACTIVE_FINAL_JUDGE_MODEL,
    )
    with pytest.raises(
        PortfolioBudgetError,
        match="pricing and provider token ceilings are not frozen",
    ):
        make_portfolio_provider_pricing_profile(
            active_judge,
            final_judge_max_output_tokens=8_192,
        )

    judge = _legacy_kimi_budget_identity()
    assert judge.model == PORTFOLIO_KIMI_MODEL
    profile = make_portfolio_provider_pricing_profile(
        judge,
        final_judge_max_output_tokens=(
            PORTFOLIO_KIMI_FINAL_JUDGE_RESERVE_OUTPUT_TOKENS
        ),
    )
    assert profile.max_input_tokens == 229_376
    assert profile.max_output_tokens == 8_192
    for unsafe_output_reserve in (2_048, 16_384):
        with pytest.raises(ValueError, match="must equal"):
            make_portfolio_provider_pricing_profile(
                judge,
                final_judge_max_output_tokens=unsafe_output_reserve,
            )


def test_route_reservation_settles_captured_output_above_legacy_128_cap(
    tmp_path,
) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
    )
    reservation, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(stage="shared_route"),
    )

    assert reservation.max_output_tokens == PORTFOLIO_QWEN_ROUTE_MAX_OUTPUT_TOKENS
    assert reservation.reserved_cost_cny == Decimal("0.157900800000")

    settlement, _ = settle_portfolio_provider_call_success(
        tmp_path,
        reservation_sha256=reservation.reservation_sha256,
        actual_input_tokens=1_000,
        actual_output_tokens=256,
        provider_request_id="request-qwen-route-256",
        response_sha256=_SHA_F,
    )

    assert settlement.actual_output_tokens == 256
    assert settlement.actual_cost_cny == Decimal("0.000534000000")


def test_budget_success_requires_usage_and_provider_response_identity(tmp_path) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("3"),
    )
    reservation, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_legacy_kimi_budget_identity(),
        final_judge_max_output_tokens=8_192,
    )
    assert reservation.reserved_cost_cny == Decimal("1.712128000000")

    with pytest.raises(PortfolioBudgetSettlementError, match="zero usage"):
        settle_portfolio_provider_call_success(
            tmp_path,
            reservation_sha256=reservation.reservation_sha256,
            actual_input_tokens=0,
            actual_output_tokens=0,
            provider_request_id="request-kimi-001",
            response_sha256=_SHA_F,
        )
    with pytest.raises(ValueError, match="provider_request_id"):
        settle_portfolio_provider_call_success(
            tmp_path,
            reservation_sha256=reservation.reservation_sha256,
            actual_input_tokens=1,
            actual_output_tokens=0,
            provider_request_id=" ",
            response_sha256=_SHA_F,
        )
    assert load_portfolio_budget_ledger(tmp_path).unresolved_reservations == (
        reservation,
    )


def test_budget_ledger_rejects_tampered_self_hash(tmp_path) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
    )
    _, path = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(),
    )
    raw = json.loads(path.read_bytes())
    raw["reserved_cost_cny"] = "0.000000000001"
    path.write_bytes(
        (
            json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )

    with pytest.raises(ValueError, match="reservation differs|self hash"):
        load_portfolio_budget_ledger(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "expected_error", "resign"),
    [
        (
            "forfeited_cost_cny",
            "0.100000000000",
            "complete reservation",
            True,
        ),
        ("identity_sha256", "0" * 64, "identity drifted", True),
        ("reason", "release_reserve", "reason", True),
        ("previous_event_sha256", "0" * 64, "chain mismatch", True),
        ("disposition", "charged_full_reserve", "self hash", False),
    ],
)
def test_budget_ledger_rejects_tampered_forfeit(
    tmp_path,
    field: str,
    value: str,
    expected_error: str,
    resign: bool,
) -> None:
    initialize_portfolio_budget_ledger(
        tmp_path,
        matrix_run_id="portfolio-mini",
        phase_cap_cny=Decimal("1"),
    )
    reservation, _ = reserve_portfolio_provider_call(
        tmp_path,
        identity=_budget_identity(),
    )
    _, path = forfeit_portfolio_provider_call(
        tmp_path,
        reservation_sha256=reservation.reservation_sha256,
        reason="provider_call_ended_without_captured_response",
    )
    raw = json.loads(path.read_bytes())
    if not resign:
        # The value is intentionally unchanged: rewriting the unsigned payload
        # is unnecessary; corrupt only the persisted self hash.
        raw["forfeit_sha256"] = "0" * 64
    else:
        raw[field] = value
        unsigned = dict(raw)
        unsigned.pop("forfeit_sha256")
        raw["forfeit_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    path.write_bytes(canonical_json_bytes(raw))

    with pytest.raises(ValueError, match=expected_error):
        load_portfolio_budget_ledger(tmp_path)


def test_budget_api_rejects_float_authority(tmp_path) -> None:
    with pytest.raises(TypeError, match="Decimal"):
        initialize_portfolio_budget_ledger(
            tmp_path,
            matrix_run_id="portfolio-mini",
            phase_cap_cny=1.0,  # type: ignore[arg-type]
        )
