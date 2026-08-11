from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import scripts.prepare_portfolio_stage1_handoff as handoff_cli
from skillchain.evaluation.assistant_runs import build_assistant_query_input
from skillchain.evaluation.portfolio_inputs import PortfolioQueryAssetBinding
from skillchain.evaluation.portfolio_stage1 import (
    PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS,
    PORTFOLIO_STAGE1_OUTPUT_FILES,
    PortfolioDevMiniTrajectoryBundle,
    PortfolioStage1Error,
    build_portfolio_s1_handoff,
    create_portfolio_s1_handoff,
    load_verified_portfolio_s1_handoff,
    require_verified_portfolio_s1_handoff,
)
from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.stage1 import TrajectoryBundle
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes
from skillchain.taxonomy import TAXONOMY_VERSION


ROOT = Path(__file__).resolve().parents[2]
AUTHORING_INPUT = (
    ROOT / "specs" / "authoring" / "authoring-packet-codex-high-v5.json"
)
AUTHORING_INPUT_FILE_SHA256 = (
    "b389da568575e7b1502923db0a4667e70954233924623856b4f96d6a4fdf9cf1"
)
_HASHES = tuple(character * 64 for character in "abcdef0123456789")
_QUERY_SPECS = (
    ("dm-006", "exact_match", "product.exact_match"),
    ("dm-009", "multi_product", "product.multi_search"),
    ("dm-013", "divergent_rec", "product.style_recommendation"),
    ("dm-016", "encyclopedia", "knowledge.visual_encyclopedia"),
    ("dm-019", "utility", "utility.document_reading"),
    ("dm-020", "utility", "utility.document_reading"),
)


def _query(query_id: str, intent: str, capability: str) -> Query:
    text = f"Portfolio handoff request {query_id}"
    suffix = query_id.removeprefix("dm-")
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version="ecommerce-task-spec-v1",
        query_id=query_id,
        asset_id=f"asset-{query_id}",
        image_path=f"query_images/{intent}/{query_id}.jpg",
        leakage_group_id=f"leakage-{suffix}",
        boundary_group_id=None,
        template_family=f"portfolio-handoff-{query_id}",
        generator_batch_id="dev-mini-001",
        text=text,
        turns=[ConversationTurn(role="user", content=text)],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=False,
        boundary_strategy=None,
        requires_card=intent in {"exact_match", "multi_product", "divergent_rec"},
        split="dev_mini",
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="portfolio-stage1-test",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _fake_context():
    queries = tuple(_query(*item) for item in _QUERY_SPECS)
    smoke = next(item for item in queries if item.query_id == "dm-019")
    assistant_smoke = build_assistant_query_input(smoke)
    assets = tuple(
        PortfolioQueryAssetBinding(
            query_id=item.query_id,
            asset_id=item.asset_id,
            image_path=item.image_path,
            image_sha256=sha256_bytes(item.query_id.encode("utf-8")),
        )
        for item in queries
    )
    inputs = SimpleNamespace(
        queries=queries,
        query_assets=assets,
        expected_plan_sha256=_HASHES[0],
        expected_accepted_ledger_sha256=_HASHES[1],
        expected_query_artifact_sha256=_HASHES[2],
        expected_capability_assignments_sha256=_HASHES[3],
        expected_seed_set_sha256=_HASHES[4],
        expected_base_catalog_sha256=_HASHES[5],
        expected_output_catalog_sha256=_HASHES[6],
    )
    checklist_value = SimpleNamespace(
        track="portfolio",
        status="planned_no_model_calls",
        model_calls_performed=0,
        execution_authorized=False,
        formal_eligible=False,
        planned_query_count=1,
        selected_query=assistant_smoke,
        checklist_sha256=_HASHES[7],
    )
    checklist = SimpleNamespace(
        expected_file_sha256=_HASHES[8],
        checklist=checklist_value,
    )
    return inputs, checklist


@pytest.fixture
def fake_context(monkeypatch: pytest.MonkeyPatch):
    inputs, checklist = _fake_context()
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_stage1."
        "require_verified_portfolio_dev_mini_inputs",
        lambda value: value,
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_stage1."
        "require_verified_portfolio_five_config_run_checklist",
        lambda value, *, inputs: value,
    )
    return inputs, checklist


def test_builds_distinct_five_intent_zero_call_handoff(fake_context) -> None:
    inputs, checklist = fake_context
    artifacts = build_portfolio_s1_handoff(
        inputs=inputs,
        checklist=checklist,
        authoring_input_path=AUTHORING_INPUT,
        expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
    )

    bundle = artifacts.trajectory_bundle
    packet = artifacts.creator_packet
    manifest = artifacts.manifest
    assert bundle.selected_query_ids == PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS
    assert tuple(item.query_id for item in bundle.trajectories) == (
        PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS
    )
    assert {item.canonical_intent for item in bundle.trajectories} == {
        "exact_match",
        "multi_product",
        "divergent_rec",
        "encyclopedia",
        "utility",
    }
    assert bundle.smoke_query_id == "dm-019"
    assert "dm-019" not in bundle.selected_query_ids
    assert bundle.model_calls_performed == 0
    assert bundle.creator_invoked is False
    assert bundle.bank_generated is False
    assert bundle.execution_authorized is False
    assert bundle.formal_eligible is False
    assert all(item.image_sha256 for item in bundle.trajectories)
    assert packet.authoring_input.consumers == ("llm_static", "s1")
    assert packet.trajectory_bundle == bundle
    assert manifest.model_calls_performed == 0
    assert manifest.bank_generated is False

    serialized = bundle.canonical_bytes()
    for forbidden in (
        b"label_provenance",
        b"judge",
        b"failure",
        b"attribution",
        b"tool_trace",
        b"gate",
        b"test_frozen",
    ):
        assert forbidden not in serialized

    with pytest.raises(ValidationError):
        TrajectoryBundle.model_validate(bundle.model_dump(mode="json"), strict=True)


def test_build_reverifies_all_sources_before_return(
    fake_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, checklist = fake_context
    calls = {"inputs": 0, "checklist": 0}

    def require_inputs(value):
        calls["inputs"] += 1
        return value

    def require_checklist(value, *, inputs):
        calls["checklist"] += 1
        return value

    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_stage1."
        "require_verified_portfolio_dev_mini_inputs",
        require_inputs,
    )
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_stage1."
        "require_verified_portfolio_five_config_run_checklist",
        require_checklist,
    )

    build_portfolio_s1_handoff(
        inputs=inputs,
        checklist=checklist,
        authoring_input_path=AUTHORING_INPUT,
        expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
    )

    assert calls == {"inputs": 2, "checklist": 2}


def test_create_load_require_exact_files_and_overwrite_rejection(
    tmp_path: Path,
    fake_context,
) -> None:
    inputs, checklist = fake_context
    output = tmp_path / "portfolio-s1-handoff"
    created = create_portfolio_s1_handoff(
        inputs=inputs,
        checklist=checklist,
        authoring_input_path=AUTHORING_INPUT,
        expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
        output_dir=output,
    )
    assert tuple(sorted(item.name for item in output.iterdir())) == tuple(
        sorted(PORTFOLIO_STAGE1_OUTPUT_FILES)
    )
    verified = load_verified_portfolio_s1_handoff(
        output,
        expected_manifest_file_sha256=created.manifest_file_sha256,
        inputs=inputs,
        checklist=checklist,
        authoring_input_path=AUTHORING_INPUT,
        expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
    )
    assert (
        require_verified_portfolio_s1_handoff(
            verified,
            inputs=inputs,
            checklist=checklist,
        )
        is verified
    )

    with pytest.raises(FileExistsError, match="create-only"):
        create_portfolio_s1_handoff(
            inputs=inputs,
            checklist=checklist,
            authoring_input_path=AUTHORING_INPUT,
            expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
            output_dir=output,
        )
    (output / "unexpected.json").write_bytes(b"{}\n")
    with pytest.raises(PortfolioStage1Error, match="exactly the three"):
        load_verified_portfolio_s1_handoff(
            output,
            expected_manifest_file_sha256=created.manifest_file_sha256,
            inputs=inputs,
            checklist=checklist,
            authoring_input_path=AUTHORING_INPUT,
            expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
        )


def test_strict_bundle_rejects_later_stage_pollution(fake_context) -> None:
    inputs, checklist = fake_context
    bundle = build_portfolio_s1_handoff(
        inputs=inputs,
        checklist=checklist,
        authoring_input_path=AUTHORING_INPUT,
        expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
    ).trajectory_bundle
    raw = bundle.model_dump(mode="json")
    raw["judge_scores"] = {"dm-006": 10}
    raw["trajectory_bundle_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in raw.items()
                if key != "trajectory_bundle_sha256"
            }
        )
    )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PortfolioDevMiniTrajectoryBundle.model_validate(raw, strict=True)


def test_cli_reports_preparation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs, checklist = _fake_context()
    created = SimpleNamespace(
        output_dir=tmp_path / "handoff",
        manifest_file_sha256=_HASHES[9],
        selected_query_count=5,
    )
    monkeypatch.setattr(handoff_cli, "load_current_portfolio_inputs", lambda: inputs)
    monkeypatch.setattr(
        handoff_cli,
        "load_verified_portfolio_five_config_run_checklist",
        lambda *_args, **_kwargs: checklist,
    )
    monkeypatch.setattr(
        handoff_cli,
        "create_portfolio_s1_handoff",
        lambda **_kwargs: created,
    )

    assert handoff_cli.main(["--output-dir", str(created.output_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "prepared_not_invoked"
    assert report["purpose"] == "input_handoff_smoke"
    assert report["selected_query_ids"] == list(
        PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS
    )
    assert report["model_calls_performed"] == 0
    assert report["creator_invoked"] is False
    assert report["bank_generated"] is False
    assert report["execution_authorized"] is False
    assert report["formal_eligible"] is False
