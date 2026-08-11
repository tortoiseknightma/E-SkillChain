from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillchain.task_spec import (
    DEFAULT_TASK_SPEC_PATH,
    DEFAULT_TASK_SPEC_SHA256,
    MVP_TASK_SPEC_V1_PATH,
    MVP_TASK_SPEC_V1_SHA256,
    TaskSpecError,
    load_default_task_specification,
    load_mvp_task_specification_v1,
    load_task_specification,
)
from skillchain.taxonomy import load_default_taxonomy_registry
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _raw_spec() -> dict:
    return json.loads(DEFAULT_TASK_SPEC_PATH.read_text(encoding="utf-8"))


def _rehash(raw: dict) -> None:
    payload = dict(raw)
    payload.pop("task_spec_sha256", None)
    raw["task_spec_sha256"] = sha256_bytes(canonical_json_bytes(payload))


def _write_spec(path: Path, raw: dict) -> None:
    _rehash(raw)
    path.write_bytes(_canonical_pretty(raw))


def test_default_task_spec_is_complete_strict_and_taxonomy_bound():
    taxonomy = load_default_taxonomy_registry()
    specification = load_default_task_specification()

    assert DEFAULT_TASK_SPEC_PATH.name == "ecommerce-task-spec-v0.json"
    assert specification.task_spec_version == "ecommerce-task-spec-v0"
    assert specification.task_spec_sha256 == DEFAULT_TASK_SPEC_SHA256
    assert specification.taxonomy_sha256 == taxonomy.taxonomy_sha256
    assert set(specification.capabilities_by_id) == set(taxonomy.capabilities_by_id)
    assert all(task.allowed_tools for task in specification.capabilities)
    assert all(task.success_criteria for task in specification.capabilities)
    assert all(task.failure_conditions for task in specification.capabilities)
    assert all(task.safety_constraints for task in specification.capabilities)
    assert all(task.indeterminate_conditions for task in specification.capabilities)
    assert all(
        task.fallback.must_state_uncertainty for task in specification.capabilities
    )
    assert {
        task.capability_id: task.output_contract.card_requirement
        for task in specification.capabilities
    } == {
        capability.capability_id: (
            "required" if capability.requires_card else "forbidden"
        )
        for capability in taxonomy.capabilities
    }
    assert {
        rule.provenance.kind
        for task in specification.capabilities
        for group in (
            task.input_preconditions,
            task.success_criteria,
            task.failure_conditions,
            task.acceptable_answer_rules,
            task.safety_constraints,
            task.fallback.trigger_rules,
            task.fallback.response_rules,
            task.indeterminate_conditions,
        )
        for rule in group
    } <= {"public_source", "tool_contract", "project_choice"}


def test_v1_task_spec_replaces_multi_product_chain_and_defines_style_modes():
    legacy = load_default_task_specification()
    specification = load_mvp_task_specification_v1()

    assert legacy.task_spec_version == "ecommerce-task-spec-v0"
    assert MVP_TASK_SPEC_V1_PATH.is_file()
    assert specification.task_spec_sha256 == MVP_TASK_SPEC_V1_SHA256
    assert specification.task_spec_version == "ecommerce-task-spec-v1"
    assert specification.taxonomy_sha256 == legacy.taxonomy_sha256
    legacy_by_id = legacy.capabilities_by_id
    v1_by_id = specification.capabilities_by_id
    assert set(v1_by_id) == set(legacy_by_id)

    for capability_id in sorted(v1_by_id):
        current = v1_by_id[capability_id]
        prior = legacy_by_id[capability_id]
        if capability_id == "product.multi_search":
            assert current.allowed_tools == ("multi_product_search",)
            provenance = {
                rule.provenance.source_ref
                for rule in (
                    *current.success_criteria,
                    *current.safety_constraints,
                )
                if rule.provenance.kind == "tool_contract"
            }
            assert provenance == {"tool:multi_product_search@1.0.0"}
            continue
        if capability_id == "product.style_recommendation":
            assert current.allowed_tools == ("style_similar_search",)
            current_refs = {
                rule.provenance.source_ref
                for rule in (
                    *current.success_criteria,
                    *current.safety_constraints,
                    *current.fallback.trigger_rules,
                )
                if rule.provenance.kind == "tool_contract"
            }
            prior_refs = {
                rule.provenance.source_ref
                for rule in (
                    *prior.success_criteria,
                    *prior.safety_constraints,
                    *prior.fallback.trigger_rules,
                )
                if rule.provenance.kind == "tool_contract"
            }
            assert current_refs == {"tool:style_similar_search@2.3.0"}
            assert prior_refs == {"tool:style_similar_search@2.1.0"}
            assert (
                "cross-category coordination"
                in current.input_preconditions[0].statement
            )
            assert (
                "trace's declared style submode"
                in current.success_criteria[0].statement
            )
            assert "requested same-category or cross-category mode" in (
                current.fallback.trigger_rules[0].statement
            )
            continue
        assert current == prior


def test_task_spec_rejects_taxonomy_binding_change(tmp_path: Path):
    taxonomy = load_default_taxonomy_registry()
    raw = _raw_spec()
    raw["taxonomy_sha256"] = "f" * 64
    path = tmp_path / "wrong-taxonomy.json"
    _write_spec(path, raw)

    with pytest.raises(TaskSpecError, match="taxonomy SHA-256 mismatch"):
        load_task_specification(path, taxonomy=taxonomy)


def test_task_spec_expected_hash_rejects_coordinated_content_change(tmp_path: Path):
    taxonomy = load_default_taxonomy_registry()
    raw = _raw_spec()
    raw["capabilities"][0]["success_criteria"][0]["statement"] = (
        "A benign but unreviewed success criterion."
    )
    path = tmp_path / "changed.json"
    _write_spec(path, raw)

    with pytest.raises(TaskSpecError, match="expected SHA-256"):
        load_task_specification(
            path,
            taxonomy=taxonomy,
            expected_sha256=DEFAULT_TASK_SPEC_SHA256,
        )


@pytest.mark.parametrize(
    "contamination",
    [
        "Read the candidate Skill before deciding success.",
        "Use the rubric and gold labels.",
        "Tune this rule from evaluation results.",
        "Inspect trajectories before applying this rule.",
        "参考技能库和裁判结果。",
    ],
)
def test_task_spec_rejects_experiment_derived_information(
    tmp_path: Path, contamination: str
):
    taxonomy = load_default_taxonomy_registry()
    raw = _raw_spec()
    raw["capabilities"][0]["success_criteria"][0]["statement"] = contamination
    path = tmp_path / "contaminated.json"
    _write_spec(path, raw)

    with pytest.raises(TaskSpecError, match="violates schema"):
        load_task_specification(path, taxonomy=taxonomy)


def test_task_spec_rejects_unknown_tool_and_invalid_provenance(tmp_path: Path):
    taxonomy = load_default_taxonomy_registry()

    unknown_tool = _raw_spec()
    unknown_tool["capabilities"][0]["allowed_tools"] = ["shell_exec"]
    unknown_path = tmp_path / "unknown-tool.json"
    _write_spec(unknown_path, unknown_tool)
    with pytest.raises(TaskSpecError, match="violates schema"):
        load_task_specification(unknown_path, taxonomy=taxonomy)

    bad_source = _raw_spec()
    provenance = bad_source["capabilities"][0]["success_criteria"][0]["provenance"]
    provenance["kind"] = "public_source"
    provenance["source_ref"] = "project-choice:not-public"
    source_path = tmp_path / "bad-source.json"
    _write_spec(source_path, bad_source)
    with pytest.raises(TaskSpecError, match="violates schema"):
        load_task_specification(source_path, taxonomy=taxonomy)

    wrong_tool_version = _raw_spec()
    wrong_tool_version["capabilities"][0]["success_criteria"][0]["provenance"][
        "source_ref"
    ] = "tool:encyclopedia_lookup@9.9.9"
    version_path = tmp_path / "wrong-tool-version.json"
    _write_spec(version_path, wrong_tool_version)
    with pytest.raises(TaskSpecError, match="violates schema"):
        load_task_specification(version_path, taxonomy=taxonomy)


def test_task_spec_rejects_card_contract_inconsistent_with_taxonomy(tmp_path: Path):
    taxonomy = load_default_taxonomy_registry()
    raw = _raw_spec()
    output = raw["capabilities"][0]["output_contract"]
    output["response_kind"] = "product_cards"
    output["card_requirement"] = "required"
    output["card_fields"] = ["evidence_reference", "product_id", "title"]
    path = tmp_path / "wrong-card-contract.json"
    _write_spec(path, raw)

    with pytest.raises(TaskSpecError, match="card requirement mismatch"):
        load_task_specification(path, taxonomy=taxonomy)


def test_task_spec_rejects_noncanonical_and_invalid_expected_digest(tmp_path: Path):
    taxonomy = load_default_taxonomy_registry()
    path = tmp_path / "noncanonical.json"
    path.write_text(json.dumps(_raw_spec(), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(TaskSpecError, match="canonical pretty JSON"):
        load_task_specification(path, taxonomy=taxonomy)
    with pytest.raises(TaskSpecError, match="expected Task Specification SHA-256"):
        load_task_specification(
            DEFAULT_TASK_SPEC_PATH,
            taxonomy=taxonomy,
            expected_sha256="invalid",
        )
