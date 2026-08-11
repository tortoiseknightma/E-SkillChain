from __future__ import annotations

from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from scripts import run_portfolio_treatment_smoke as smoke
from skillchain.evaluation.portfolio_execution import load_portfolio_budget_ledger


class _Dumpable(SimpleNamespace):
    def model_dump(self, *, mode: str):
        assert mode == "json"
        return self.dump


def _prepared_budget_smoke(runner, query_ids: tuple[str, ...]):
    target_bank = SimpleNamespace(bank_sha256="c" * 64)
    return SimpleNamespace(
        config="s1",
        query_ids=query_ids,
        target_bank=target_bank,
        parent_bank=None,
        banks={"s1": target_bank},
        target_bank_file_sha256="d" * 64,
        parent_bank_file_sha256=None,
        runtime=SimpleNamespace(registry=object()),
        catalog=object(),
        runner=runner,
        runtime_lock={"runtime_lock_sha256": "e" * 64},
        runtime_lock_file_sha256="f" * 64,
        public_by_id={query_id: object() for query_id in query_ids},
        private_by_id={
            query_id: SimpleNamespace(canonical_capability="product.exact_match")
            for query_id in query_ids
        },
        query_artifact_sha256="1" * 64,
        plan_sha256="2" * 64,
        judge_runtime=object(),
        rubric=object(),
        rubric_file_sha256="3" * 64,
    )


def _budget_arguments(
    tmp_path,
    *,
    prior_cost_cny: float,
    phase_cap_cny: float,
):
    return SimpleNamespace(
        config="s1",
        output_dir=tmp_path / "output",
        paired_route_source_root=None,
        prior_cost_cny=prior_cost_cny,
        phase_cap_cny=phase_cap_cny,
    )


def test_progress_output_failure_does_not_abort_smoke(monkeypatch) -> None:
    def closed_stdout(*args, **kwargs) -> None:
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr("builtins.print", closed_stdout)
    smoke._emit_progress({"completed": 1, "query_id": "dm-001"})


def test_development_bank_mapping_projects_each_target_stage() -> None:
    target = object()
    parent = object()

    llm_static = smoke._development_bank_mapping(
        "llm_static",
        target,
        parent,
    )
    assert tuple(llm_static) == ("llm_static", "s1", "s1s2", "full")
    assert all(value is target for value in llm_static.values())

    s1 = smoke._development_bank_mapping("s1", target, parent)
    assert s1["llm_static"] is parent
    assert all(s1[name] is target for name in ("s1", "s1s2", "full"))

    s1s2 = smoke._development_bank_mapping("s1s2", target, parent)
    assert all(s1s2[name] is parent for name in ("llm_static", "s1"))
    assert all(s1s2[name] is target for name in ("s1s2", "full"))

    full = smoke._development_bank_mapping("full", target, parent)
    assert all(full[name] is parent for name in ("llm_static", "s1", "s1s2"))
    assert full["full"] is target


def test_default_queries_cover_all_six_capabilities_once() -> None:
    capability_by_query = {
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
    capabilities = tuple(
        capability_by_query[query_id] for query_id in smoke.DEFAULT_QUERY_IDS
    )
    private_by_id = {
        query_id: SimpleNamespace(canonical_capability=capability)
        for query_id, capability in zip(
            smoke.DEFAULT_QUERY_IDS,
            capabilities,
            strict=True,
        )
    }
    observed = smoke._validate_query_capability_coverage(
        private_by_id,
        smoke.DEFAULT_QUERY_IDS,
    )
    assert observed == capabilities
    assert frozenset(observed) == smoke.REQUIRED_SMOKE_CAPABILITIES


def test_query_coverage_rejects_duplicate_capabilities() -> None:
    private_by_id = {
        query_id: SimpleNamespace(canonical_capability="product.exact_match")
        for query_id in smoke.DEFAULT_QUERY_IDS
    }
    with pytest.raises(ValueError, match="optimization25"):
        smoke._validate_query_capability_coverage(
            private_by_id,
            smoke.DEFAULT_QUERY_IDS,
        )


def test_cost_includes_shared_route_once_and_tracks_phase_cap() -> None:
    value = smoke.calculate_smoke_cost(
        assistant_input_tokens=100,
        assistant_output_tokens=20,
        shared_route_input_tokens=40,
        shared_route_output_tokens=5,
        judge_input_tokens=60,
        judge_output_tokens=10,
        prior_cost_cny=0.25,
        phase_cap_cny=0.251,
    )
    expected_qwen = ((140 * 0.15) + (25 * 1.5)) / 1_000_000
    expected_kimi = ((60 * 6.5) + (10 * 27.0)) / 1_000_000
    assert value["qwen_cost_cny"] == pytest.approx(expected_qwen)
    assert value["kimi_cost_cny"] == pytest.approx(expected_kimi)
    assert value["observed_cost_cny"] == pytest.approx(expected_qwen + expected_kimi)
    assert value["cumulative_cost_cny"] == pytest.approx(
        0.25 + expected_qwen + expected_kimi
    )
    assert value["cap_reached"] is False


def test_judge_retry_accounting_uses_aggregate_usage_and_attempt_receipts() -> None:
    totals = {
        "judge_model_calls": 0,
        "judge_captured_response_count": 0,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
    }
    judge = SimpleNamespace(
        attempts=2,
        captured_response_count=2,
        aggregate_usage=SimpleNamespace(input_tokens=50, output_tokens=77),
    )

    smoke._accumulate_judge_accounting(totals, judge)

    assert totals == {
        "judge_model_calls": 2,
        "judge_captured_response_count": 2,
        "judge_input_tokens": 50,
        "judge_output_tokens": 77,
    }


def test_execute_initializes_ledger_and_passes_required_budget_contexts(
    monkeypatch,
    tmp_path,
) -> None:
    query_ids = ("dm-001",)
    captured = {}
    request = _Dumpable(
        request_sha256="4" * 64,
        dump={"request_sha256": "4" * 64},
    )
    response = _Dumpable(
        error_code=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        selected_capability="product.exact_match",
        skill_slug="exact-match",
        response_text="grounded response",
        visible_cards=(),
        dump={
            "response_text": "grounded response",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    )
    receipt = _Dumpable(
        model_calls=(object(),),
        dump={"model_call_count": 1},
    )

    class Runner:
        def execute(self, received_request, **kwargs):
            assert received_request is request
            context = kwargs["budget_context"]
            captured["assistant_context"] = context
            state = load_portfolio_budget_ledger(context.ledger_root)
            captured["assistant_authority"] = state.authority
            return SimpleNamespace(response=response, receipt=receipt)

    prepared = _prepared_budget_smoke(Runner(), query_ids)
    arguments = _budget_arguments(
        tmp_path,
        prior_cost_cny=0.25,
        phase_cap_cny=5.0,
    )
    judge = _Dumpable(
        attempts=1,
        captured_response_count=1,
        aggregate_usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        outcome=SimpleNamespace(
            status="scored",
            scores=SimpleNamespace(j_project=80.0),
        ),
        dump={"status": "scored", "j_project": 80.0},
    )

    monkeypatch.setattr(smoke, "_prepare_smoke", lambda *_: prepared)
    monkeypatch.setattr(
        smoke,
        "_backbone_and_budget",
        lambda *_: (object(), object(), object()),
    )
    monkeypatch.setattr(smoke, "_request", lambda **_: request)
    monkeypatch.setattr(
        smoke,
        "make_active_portfolio_evaluator_isolation_lock",
        lambda: object(),
    )
    monkeypatch.setattr(smoke, "_assistant_result", lambda *_: object())
    monkeypatch.setattr(
        smoke,
        "build_final_evaluation_packet",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        smoke,
        "calculate_portfolio_skill_adherence",
        lambda **_: 0.5,
    )

    def fake_judge(_packet, _isolation, **kwargs):
        context = kwargs["budget_context"]
        captured["judge_context"] = context
        state = load_portfolio_budget_ledger(context.ledger_root)
        captured["judge_authority"] = state.authority
        return judge

    monkeypatch.setattr(smoke, "run_visual_final_judge", fake_judge)

    summary = smoke._execute_treatment_smoke(arguments, query_ids)

    assistant_context = captured["assistant_context"]
    judge_context = captured["judge_context"]
    authority = captured["assistant_authority"]
    assert isinstance(assistant_context, smoke.PortfolioAssistantBudgetContext)
    assert isinstance(judge_context, smoke.FinalJudgeBudgetContext)
    assert assistant_context.ledger_root == judge_context.ledger_root
    assert assistant_context.instance_sha256 == request.request_sha256
    assert assistant_context.attempt_index == judge_context.attempt_index == 1
    assert judge_context.matrix_run_id == authority.matrix_run_id
    assert judge_context.query_id == "dm-001"
    assert judge_context.request_sha256 == request.request_sha256
    assert authority.phase_cap_cny == Decimal("5.000000000000")
    assert authority.prior_observed_cost_cny == Decimal("0.250000000000")
    assert captured["judge_authority"] == authority
    persisted = load_portfolio_budget_ledger(arguments.output_dir / "budget-ledger")
    assert persisted.authority == authority
    assert summary["budget_authority_sha256"] == authority.authority_sha256
    assert summary["status"] == "complete"
    assert summary["query_ids_completed"] == ["dm-001"]
    assert summary["query_ids_incomplete"] == []
    assert summary["incomplete_score_disposition"] == "not_applicable"


def test_budget_excess_keeps_unfinished_queries_incomplete_and_not_scored(
    monkeypatch,
    tmp_path,
) -> None:
    query_ids = ("dm-001", "dm-002")
    observed = {}
    request = _Dumpable(
        request_sha256="5" * 64,
        dump={"request_sha256": "5" * 64},
    )

    class Runner:
        def execute(self, _request, **kwargs):
            context = kwargs["budget_context"]
            state = load_portfolio_budget_ledger(context.ledger_root)
            observed["authority"] = state.authority
            observed["provider_calls"] = 0
            raise smoke.PortfolioBudgetExceededError(
                "first Assistant reservation exceeds the phase cap"
            )

    prepared = _prepared_budget_smoke(Runner(), query_ids)
    arguments = _budget_arguments(
        tmp_path,
        prior_cost_cny=0.25,
        phase_cap_cny=0.3,
    )
    monkeypatch.setattr(smoke, "_prepare_smoke", lambda *_: prepared)
    monkeypatch.setattr(
        smoke,
        "_backbone_and_budget",
        lambda *_: (object(), object(), object()),
    )
    monkeypatch.setattr(smoke, "_request", lambda **_: request)
    monkeypatch.setattr(
        smoke,
        "make_active_portfolio_evaluator_isolation_lock",
        lambda: object(),
    )

    summary = smoke._execute_treatment_smoke(arguments, query_ids)

    assert summary["status"] == "budget_cap_reached"
    assert summary["query_count_completed"] == 0
    assert summary["query_count_completed"] < summary["query_count_requested"]
    assert summary["query_ids_completed"] == []
    assert summary["query_ids_incomplete"] == ["dm-001", "dm-002"]
    assert summary["incomplete_score_disposition"] == "not_scored_no_fixed_zero"
    assert summary["mean_j_project"] == 0.0
    assert (arguments.output_dir / "results.jsonl").read_bytes() == b""
    assert list((arguments.output_dir / "assistant").iterdir()) == []
    assert list((arguments.output_dir / "final").iterdir()) == []
    authority = observed["authority"]
    assert authority.phase_cap_cny == Decimal("0.300000000000")
    assert authority.prior_observed_cost_cny == Decimal("0.250000000000")
    ledger = load_portfolio_budget_ledger(arguments.output_dir / "budget-ledger")
    assert ledger.last_event_index == 0
    assert ledger.accountable_cost_cny == Decimal("0.250000000000")
    assert observed["provider_calls"] == 0


def _argv(tmp_path, *, execute: bool = False) -> list[str]:
    values = [
        "--config",
        "s1",
        "--target-bank",
        str(tmp_path / "bank.json"),
        "--runtime-data-root",
        str(tmp_path / "runtime"),
        "--output-dir",
        str(tmp_path / "output"),
        "--prior-cost-cny",
        "0.25",
        "--phase-cap-cny",
        "1.0",
    ]
    if execute:
        values.append("--execute")
    return values


def test_default_mode_is_a_zero_call_dry_run(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry run crossed the explicit execute gate")

    monkeypatch.setattr(smoke, "_execute_treatment_smoke", forbidden)
    assert smoke.main(_argv(tmp_path)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry_run_only"
    assert output["model_calls_performed"] == 0
    assert output["query_ids"] == list(smoke.DEFAULT_QUERY_IDS)
    assert output["matrix_ready"] is False
    assert output["official_treatment_claim_allowed"] is False
    assert not (tmp_path / "output").exists()


def test_execute_flag_is_the_only_dispatch_to_real_orchestration(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    captured = {}

    def fake_execute(arguments, query_ids):
        captured["execute"] = arguments.execute
        captured["query_ids"] = query_ids
        return {
            "status": "complete",
            "formal_eligible": False,
            "matrix_ready": False,
        }

    monkeypatch.setattr(smoke, "_execute_treatment_smoke", fake_execute)
    monkeypatch.setattr(smoke, "_require_active_judge_execute_ready", lambda: None)
    assert smoke.main(_argv(tmp_path, execute=True)) == 0
    assert captured == {
        "execute": True,
        "query_ids": smoke.DEFAULT_QUERY_IDS,
    }
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert output["matrix_ready"] is False


def test_execute_fails_closed_while_gemini_pricing_is_unfrozen(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("pricing gate did not stop orchestration")

    monkeypatch.setattr(smoke, "_execute_treatment_smoke", forbidden)

    assert smoke.main(_argv(tmp_path, execute=True)) == 2
    assert "pricing-and-ceilings-not-frozen" in capsys.readouterr().err
