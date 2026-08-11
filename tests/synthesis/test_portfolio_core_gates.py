"""Focused text-free r2 validation gate tests; all filesystem writes use tmp_path."""

from __future__ import annotations

import runpy
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from skillchain.synthesis import portfolio_core_gates as gates_module
from skillchain.synthesis.portfolio_core_gates import (
    CORE_R2_TEXT_FREE_VALIDATION_GATE_SPEC,
    CoreR2ValidationGateError,
    CoreR2ValidationGateBundle,
    ValidationGatePublicationConflict,
    ValidationGatePublicationSafetyError,
    build_core_r2_validation_gate_bundle,
    canonical_core_r2_validation_gate_bundle_bytes,
    publish_core_r2_validation_gates,
    validate_core_r2_validation_gate_bundle,
)
from skillchain.synthesis.portfolio_core_r2 import build_r2_core_bridge


@pytest.fixture(scope="module")
def r2_inputs():
    """Full synthetic r2 metadata fixture; it never opens a source image."""

    namespace = runpy.run_path(str(Path(__file__).with_name("test_r2_planning.py")))
    result = namespace["_full_r2_fixture"]()
    plan_sha256 = namespace["_plan_sha256"](result.plan)
    bridge = build_r2_core_bridge(
        result,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    return result, bridge, plan_sha256


@pytest.fixture(scope="module")
def gate_bundle(r2_inputs) -> CoreR2ValidationGateBundle:
    result, bridge, plan_sha256 = r2_inputs
    return build_core_r2_validation_gate_bundle(
        result,
        bridge,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )


def test_text_free_gate_bundle_has_exact_sizes_atomic_batches_and_all_coverage(
    r2_inputs,
    gate_bundle: CoreR2ValidationGateBundle,
) -> None:
    result, bridge, plan_sha256 = r2_inputs
    validate_core_r2_validation_gate_bundle(
        gate_bundle,
        result,
        bridge,
        trusted_plan_sha256=plan_sha256,
    )

    assert gate_bundle.audit.gate_sizes == {
        "route_gate": 75,
        "body_gate": 75,
        "shadow_val": 50,
    }
    assert len(gate_bundle.entries) == 200
    assert {entry.plan_id for entry in gate_bundle.entries} == {
        plan_id
        for plan_id, split in result.final_split_by_plan_id.items()
        if split == "val"
    }
    by_batch: dict[str, list] = defaultdict(list)
    for entry in gate_bundle.entries:
        by_batch[entry.generator_batch_id].append(entry)
    assert len(by_batch) == 8
    assert all(len(items) == 25 and len({item.gate for item in items}) == 1 for items in by_batch.values())

    patterns = {
        "constraint_correction",
        "direct_request",
        "goal_change_or_multi_query",
        "no_result_relaxation",
        "underspecified_clarification",
    }
    for gate, capability_minimum in {"route_gate": 3, "body_gate": 3, "shadow_val": 2}.items():
        assert all(
            gate_bundle.audit.capability_counts[gate][capability]
            >= capability_minimum
            for capability in (
                "product.exact_match",
                "product.multi_search",
                "product.style_recommendation",
                "knowledge.visual_encyclopedia",
                "utility.document_reading",
                "utility.recipe_guidance",
            )
        )
        assert gate_bundle.audit.boundary_counts[gate] >= 1
        assert set(gate_bundle.audit.interaction_counts[gate]) == patterns
        assert all(gate_bundle.audit.interaction_counts[gate][pattern] >= 1 for pattern in patterns)

    payload = canonical_core_r2_validation_gate_bundle_bytes(gate_bundle)
    assert b'"text"' not in payload
    assert b'"turns"' not in payload
    assert b'"image_path"' not in payload
    assert gate_bundle.spec == CORE_R2_TEXT_FREE_VALIDATION_GATE_SPEC


def test_gate_bundle_is_deterministic_and_rejects_mapping_or_sha_tampering(
    r2_inputs,
    gate_bundle: CoreR2ValidationGateBundle,
) -> None:
    result, bridge, plan_sha256 = r2_inputs
    repeated = build_core_r2_validation_gate_bundle(
        result,
        bridge,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    assert repeated == gate_bundle
    assert canonical_core_r2_validation_gate_bundle_bytes(repeated) == (
        canonical_core_r2_validation_gate_bundle_bytes(gate_bundle)
    )

    altered_entries = list(gate_bundle.entries)
    altered_entries[0] = altered_entries[0].model_copy(
        update={"gate": "body_gate" if altered_entries[0].gate != "body_gate" else "route_gate"}
    )
    with pytest.raises(CoreR2ValidationGateError, match="gate audit is invalid|deterministic solver"):
        validate_core_r2_validation_gate_bundle(
            replace(gate_bundle, entries=tuple(altered_entries)),
            result,
            bridge,
            trusted_plan_sha256=plan_sha256,
        )
    with pytest.raises(CoreR2ValidationGateError, match="manifest binding drifted"):
        validate_core_r2_validation_gate_bundle(
            replace(
                gate_bundle,
                manifest=gate_bundle.manifest.model_copy(
                    update={"gate_rows_sha256": "f" * 64}
                ),
            ),
            result,
            bridge,
            trusted_plan_sha256=plan_sha256,
        )
    with pytest.raises(CoreR2ValidationGateError, match="plan/constraint binding"):
        build_core_r2_validation_gate_bundle(
            result,
            bridge,
            trusted_plan_sha256="0" * 64,
            seed=20260805,
        )


def test_create_only_publication_is_idempotent_and_refuses_existing_drift(
    tmp_path: Path,
    r2_inputs,
    gate_bundle: CoreR2ValidationGateBundle,
) -> None:
    result, bridge, plan_sha256 = r2_inputs
    output = tmp_path / "validation-gates.json"
    first = publish_core_r2_validation_gates(
        gate_bundle,
        output_path=output,
        result=result,
        bridge=bridge,
        trusted_plan_sha256=plan_sha256,
    )
    second = publish_core_r2_validation_gates(
        gate_bundle,
        output_path=output,
        result=result,
        bridge=bridge,
        trusted_plan_sha256=plan_sha256,
    )
    assert first == second == output
    canonical = canonical_core_r2_validation_gate_bundle_bytes(gate_bundle)
    assert output.read_bytes() == canonical

    drifted = output.read_bytes()[:-1] + b" "
    output.write_bytes(drifted)
    with pytest.raises(ValidationGatePublicationConflict, match="differs"):
        publish_core_r2_validation_gates(
            gate_bundle,
            output_path=output,
            result=result,
            bridge=bridge,
            trusted_plan_sha256=plan_sha256,
        )
    assert output.read_bytes() == drifted


def test_reparse_parent_is_rejected_before_any_publish(
    tmp_path: Path,
    r2_inputs,
    gate_bundle: CoreR2ValidationGateBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, bridge, plan_sha256 = r2_inputs
    monkeypatch.setattr(gates_module, "_is_reparse", lambda _: True)
    with pytest.raises(ValidationGatePublicationSafetyError, match="non-reparse"):
        publish_core_r2_validation_gates(
            gate_bundle,
            output_path=tmp_path / "validation-gates.json",
            result=result,
            bridge=bridge,
            trusted_plan_sha256=plan_sha256,
        )
