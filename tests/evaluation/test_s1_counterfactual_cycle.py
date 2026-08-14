from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillchain.evaluation.core_fast.engine import FastPathError
from skillchain.evolution.s1_sparse_patch import (
    DualPolicyPatchPayloadV1,
    PolicySurfaceDraftV1,
    compile_policy_surface_branch,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_s1_counterfactual_cycle.py"
_script_spec = importlib.util.spec_from_file_location(
    "run_s1_counterfactual_cycle", SCRIPT_PATH
)
assert _script_spec is not None and _script_spec.loader is not None
cycle_runner = importlib.util.module_from_spec(_script_spec)
_script_spec.loader.exec_module(cycle_runner)

_prepare_path = ROOT / "scripts" / "prepare_s1_counterfactual_cycle.py"
_prepare_spec = importlib.util.spec_from_file_location(
    "prepare_s1_counterfactual_cycle", _prepare_path
)
assert _prepare_spec is not None and _prepare_spec.loader is not None
cycle_preparer = importlib.util.module_from_spec(_prepare_spec)
_prepare_spec.loader.exec_module(cycle_preparer)

_fixture_spec = importlib.util.spec_from_file_location(
    "core_fast_fixture_module", Path(__file__).with_name("test_core_fast.py")
)
assert _fixture_spec is not None and _fixture_spec.loader is not None
fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(fixture_module)


def _branch(parent, capability: str, surface: str, round_id: str):
    parent_skill = next(
        item for item in parent.skills if item.capability_id == capability
    )
    patch = PolicySurfaceDraftV1(
        action="patch",
        policy_text=f"Apply the frozen {round_id} rule only in its visible state.",
    )
    inherit = PolicySurfaceDraftV1(action="inherit", policy_text=None)
    proposal = DualPolicyPatchPayloadV1(
        capability_id=capability,
        parent_skill_sha256=parent_skill.skill_sha256,
        action_policy=patch if surface == "action-policy" else inherit,
        response_policy=patch if surface == "response-policy" else inherit,
    )
    compiled = compile_policy_surface_branch(
        parent_bank=parent, proposal=proposal, surface=surface
    )
    assert compiled is not None
    return {
        "round_id": round_id,
        "artifact_file_sha256": round_id[-1] * 64,
        "artifact": {
            "capability": capability,
            "replay_gate": {"macro_delta_pp": 1.0, "hard_error_delta_pp": 0.0},
            "body_gate": {"macro_delta_pp": 2.5, "hard_error_delta_pp": 0.0},
        },
        "bank": compiled.bank,
    }


def test_r33_fanin_composes_only_disjoint_accepted_parent_bound_skills() -> None:
    parent = fixture_module._bank()
    recipe = _branch(parent, "utility.recipe_guidance", "action-policy", "r31")
    multi = _branch(parent, "product.multi_search", "response-policy", "r32")
    combined = cycle_runner._compose(parent, [recipe, multi])
    parent_skills = {item.capability_id: item for item in parent.skills}
    combined_skills = {item.capability_id: item for item in combined.skills}
    assert {
        capability
        for capability in parent_skills
        if parent_skills[capability].skill_sha256
        != combined_skills[capability].skill_sha256
    } == {"utility.recipe_guidance", "product.multi_search"}
    with pytest.raises(FastPathError, match="overlap"):
        cycle_runner._compose(parent, [recipe, recipe])


def test_s1_finalist_test_rejects_zero_finalist_before_provider_calls(
    tmp_path: Path,
) -> None:
    cycle_unsigned = {
        "schema_version": 1,
        "kind": "core-fast-s1-counterfactual-cycle-definition",
        "cycle_id": "s1-counterfactual-v1",
        "round_specs": {},
        "dashscope_stage_budget_cny": 10.0,
        "budget_estimate": {
            "within_cny10": True,
            "projected_total_dashscope_cny": 1.0,
            "parent_opt800_observed_cny": 0.0,
            "assistant_cny_per_outer_ceiling": 0.0025,
        },
    }
    cycle = {
        **cycle_unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(cycle_unsigned)),
    }
    cycle_path = tmp_path / "cycle-definition.json"
    cycle_path.write_bytes(canonical_json_bytes(cycle))
    finalist_unsigned = {
        "schema_version": 1,
        "kind": "core-fast-s1-cycle-finalist",
        "cycle_definition_sha256": sha256_bytes(cycle_path.read_bytes()),
        "test300_allowed": False,
        "finalist": None,
    }
    finalist = {
        **finalist_unsigned,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(finalist_unsigned)),
    }
    finalist_path = tmp_path / "cycle-finalist.json"
    finalist_path.write_bytes(canonical_json_bytes(finalist))
    with pytest.raises(FastPathError, match="one frozen cycle finalist"):
        cycle_runner.finalist_test(
            SimpleNamespace(
                cycle_definition=cycle_path,
                finalist_receipt=finalist_path,
                output_root=tmp_path / "test300",
            )
        )


def test_all_round_preflight_rejects_a_byte_exact_protected_target() -> None:
    spec = SimpleNamespace(
        s1_settings=SimpleNamespace(
            target_capabilities=("product.style_recommendation",),
            target_surface="action-policy",
        ),
        s1_parent=SimpleNamespace(
            protected_skill_sha256={
                "product.style_recommendation": "a" * 64,
            }
        ),
    )
    with pytest.raises(ValueError, match="byte-exact protected"):
        cycle_preparer._treatment_probe(  # noqa: SLF001
            parent=fixture_module._bank(),
            spec=spec,
            selected=(),
        )
