from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.prepare_portfolio_evolution_inputs as evolution
from skillchain.llm import LLMUsage
from skillchain.evaluation.portfolio_inputs import (
    VerifiedPortfolioDevMiniInputs,
)
from skillchain.synthesis.store import canonical_json_bytes
from skillchain.tools.serialization import parse_canonical_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_EXECUTION_ROOT = (
    REPOSITORY_ROOT / "runs" / "portfolio" / "portfolio-dev-mini-200x5-execution-v7"
)
S1_SHARD_ID = "02-dev-mini-001-r3-02-s1"
S1_AUDIT_SHA256 = "5a3eb187f59e668d3bf23b8a389aadc2c8eb92593a5953bb36cb7782206772c6"
S1S2_SHARD_ID = "03-dev-mini-001-r3-03-s1s2"
S1S2_AUDIT_SHA256 = "024651065565244213b7831b8c9ff280a59f8200017cda679febdb0eb4721e0e"


@pytest.fixture(scope="module")
def active_inputs() -> VerifiedPortfolioDevMiniInputs:
    return evolution._load_active_inputs()


@pytest.fixture
def use_cached_active_inputs(
    monkeypatch: pytest.MonkeyPatch,
    active_inputs: VerifiedPortfolioDevMiniInputs,
) -> VerifiedPortfolioDevMiniInputs:
    monkeypatch.setattr(evolution, "_load_active_inputs", lambda: active_inputs)
    return active_inputs


def _assert_model_consumable_packet(
    bundle: evolution.CreatedEvolutionInputBundle,
) -> None:
    assert {item.name for item in bundle.root.iterdir()} == {
        "manifest.json",
        "packet.json",
        "records.jsonl",
    }
    packet_bytes = bundle.packet_path.read_bytes()
    packet_raw = parse_canonical_json(
        packet_bytes,
        label="test evolution packet",
    )
    assert isinstance(packet_raw, dict)
    assert set(packet_raw) == {"manifest", "records", "packet_sha256"}
    assert len(packet_raw["records"]) == 25
    assert canonical_json_bytes(bundle.packet) == packet_bytes
    assert bundle.packet.manifest == bundle.manifest
    assert bundle.packet.records == bundle.records
    reloaded = evolution.load_prepared_evolution_inputs(
        bundle.root,
        expected_manifest_file_sha256=bundle.manifest_file_sha256,
    )
    assert reloaded.packet_file_sha256 == bundle.packet_file_sha256
    assert reloaded.packet_sha256 == bundle.packet_sha256


def test_s3_judge_processor_follows_schema_versioned_launch_role() -> None:
    assert (
        evolution._judge_processor_for_launch(
            SimpleNamespace(final_provider="gemini", final_model="gemini-3.6-flash")
        )
        == "aifast-gemini-judge"
    )
    assert (
        evolution._judge_processor_for_launch(
            SimpleNamespace(final_provider="kimi", final_model="kimi-k2.6")
        )
        == "dashscope-kimi-judge"
    )
    with pytest.raises(evolution.PortfolioEvolutionInputError, match="unknown"):
        evolution._judge_processor_for_launch(
            SimpleNamespace(final_provider="gemini", final_model="unbound")
        )


def test_prepare_s1_freezes_exact_25_query_six_capability_packet(
    tmp_path: Path,
    use_cached_active_inputs: VerifiedPortfolioDevMiniInputs,
) -> None:
    del use_cached_active_inputs
    bundle = evolution.prepare_s1_inputs(
        accepted_batch_id="dev-mini-001-r3",
        output_dir=tmp_path / "s1",
    )

    assert bundle.manifest.stage == "s1"
    assert bundle.manifest.component == "creator"
    assert len(bundle.records) == 25
    assert {item.canonical_capability for item in bundle.records} == set(
        evolution.EXPECTED_CAPABILITIES
    )
    assert bundle.manifest.model_calls_performed == 0
    assert bundle.manifest.evolution_component_invoked is False
    assert bundle.manifest.bank_generated is False
    _assert_model_consumable_packet(bundle)


def test_prepare_s2_freezes_route_attribution_from_pinned_s1_shard(
    tmp_path: Path,
    use_cached_active_inputs: VerifiedPortfolioDevMiniInputs,
) -> None:
    del use_cached_active_inputs
    bundle = evolution.prepare_s2_inputs(
        execution_root=SOURCE_EXECUTION_ROOT,
        source_shard_id=S1_SHARD_ID,
        expected_source_shard_audit_sha256=S1_AUDIT_SHA256,
        output_dir=tmp_path / "s2",
    )

    assert bundle.manifest.stage == "s2"
    assert bundle.manifest.source_config == "s1"
    assert bundle.manifest.source_shard_audit_sha256 == S1_AUDIT_SHA256
    assert sum(item.route_correct for item in bundle.records) == 21
    assert all(item.source_bank_sha256 for item in bundle.records)
    _assert_model_consumable_packet(bundle)


def test_prepare_s3_rejects_pinned_shard_after_runtime_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_cached_active_inputs: VerifiedPortfolioDevMiniInputs,
) -> None:
    del use_cached_active_inputs

    def cached_runtime_for(
        self: VerifiedPortfolioDevMiniInputs,
        processor: str,
    ):
        return next(
            runtime
            for runtime in self.remote_runtimes
            if runtime.processor == processor
        )

    monkeypatch.setattr(
        VerifiedPortfolioDevMiniInputs,
        "runtime_for",
        cached_runtime_for,
    )

    with pytest.raises(
        evolution.PortfolioEvolutionInputError,
        match="source final checkpoint binding differs",
    ):
        evolution.prepare_s3_inputs(
            execution_root=SOURCE_EXECUTION_ROOT,
            source_shard_id=S1S2_SHARD_ID,
            expected_source_shard_audit_sha256=S1S2_AUDIT_SHA256,
            output_dir=tmp_path / "s3",
        )


def test_final_audit_counts_retry_attempts_and_safe_reasoning_metadata() -> None:
    projections = []
    for index in range(25):
        retried = index == 0
        projections.append(
            {
                "final_kind": "visual_final_judge",
                "judge_status": "scored",
                "j_project": 80.0,
                "_query_id": f"dm-{index + 1:03d}",
                "_judge_usage": LLMUsage(input_tokens=10, output_tokens=2),
                "_judge_shape": "assessment_array",
                "_judge_attempts": 2 if retried else 1,
                "_judge_captured_response_count": 2 if retried else 1,
                "_judge_initial_empty_response": retried,
                "_judge_initial_retry_reason": (
                    "empty_final_response" if retried else None
                ),
                "_judge_initial_reasoning_present": True if retried else None,
                "_judge_initial_reasoning_tokens": 3 if retried else None,
                "_judge_initial_reasoning_bytes": 4 if retried else None,
                "_judge_initial_reasoning_sha256": "a" * 64 if retried else None,
                "_judge_terminal_response_captured": True,
                "_judge_terminal_reasoning_present": False,
                "_judge_terminal_reasoning_tokens": None,
                "_judge_terminal_reasoning_bytes": 0,
                "_judge_terminal_reasoning_sha256": None,
            }
        )
    judge_response_receipts = [
        {
            "query_id": item["_query_id"],
            "attempts": item["_judge_attempts"],
            "initial_empty_response": item["_judge_initial_empty_response"],
            "initial_reasoning_present": item["_judge_initial_reasoning_present"],
            "initial_reasoning_tokens": item["_judge_initial_reasoning_tokens"],
            "initial_reasoning_bytes": item["_judge_initial_reasoning_bytes"],
            "initial_reasoning_sha256": item["_judge_initial_reasoning_sha256"],
            "terminal_response_captured": item["_judge_terminal_response_captured"],
            "terminal_reasoning_present": item["_judge_terminal_reasoning_present"],
            "terminal_reasoning_tokens": item["_judge_terminal_reasoning_tokens"],
            "terminal_reasoning_bytes": item["_judge_terminal_reasoning_bytes"],
            "terminal_reasoning_sha256": item["_judge_terminal_reasoning_sha256"],
        }
        for item in projections
    ]
    audit = {
        "fixed_zero_count": 0,
        "judge_invoked_count": 25,
        "judge_status_counts": {"scored": 25},
        "judge_input_tokens": 250,
        "judge_output_tokens": 50,
        "mean_j_project_all_rows": 80.0,
        "min_j_project": 80.0,
        "max_j_project": 80.0,
        "judge_raw_dimensions_shape_counts": {"assessment_array": 25},
        "judge_model_calls": 26,
        "judge_captured_response_count": 26,
        "judge_retried_row_count": 1,
        "judge_initial_empty_response_count": 1,
        "judge_reasoning_present_response_count": 1,
        "judge_reasoning_tokens_reported_total": 3,
        "judge_reasoning_tokens_unavailable_response_count": 25,
        "judge_reasoning_bytes_total": 4,
        "judge_response_receipts": judge_response_receipts,
    }
    context = SimpleNamespace(
        audit=audit,
        runtime_lock={"final_judge_result_schema_version": 5},
    )

    evolution._validate_final_audit(context, tuple(projections))

    context.audit = {**audit, "judge_model_calls": 25}
    with pytest.raises(
        evolution.PortfolioEvolutionInputError,
        match="retry/reasoning metadata differs",
    ):
        evolution._validate_final_audit(context, tuple(projections))
