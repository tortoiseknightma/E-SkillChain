from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from skillchain.evaluation.assistant_runs import (
    MAIN_CONFIG_ORDER,
    build_assistant_query_input,
)
from skillchain.evaluation.portfolio_checklist import (
    PortfolioChecklistError,
    PortfolioFiveConfigRunChecklist,
    build_portfolio_five_config_run_checklist,
    create_portfolio_five_config_run_checklist,
    load_verified_portfolio_five_config_run_checklist,
    require_verified_portfolio_five_config_run_checklist,
)
from skillchain.evaluation.portfolio_inputs import PortfolioQueryAssetBinding
from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes
from skillchain.taxonomy import TASK_SPEC_VERSION, TAXONOMY_VERSION


_HASHES = tuple(character * 64 for character in "abcdef0123456")


def _query() -> Query:
    text = "上面写的什么？"
    capability = "utility.document_reading"
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=TASK_SPEC_VERSION,
        query_id="dm-019",
        asset_id=(
            "asset.v2.c2a3ec24f6cfc9a7787b0870035d845721b6e4814432be97840115c0f4d2c2a7"
        ),
        image_path="query_images/utility/document-0021.jpg",
        leakage_group_id="leakage-dm-019",
        boundary_group_id=None,
        template_family="generation-prompt-family-v1/block-001",
        generator_batch_id="dev-mini-001",
        text=text,
        turns=[ConversationTurn(role="user", content=text)],
        canonical_intent="utility",
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=False,
        boundary_strategy=None,
        requires_card=False,
        split="dev_mini",
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="portfolio-checklist-test",
                canonical_intent="utility",
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _runtime(processor: str):
    receipt = SimpleNamespace(
        receipt_sha256=_HASHES[8],
        base_selection_manifest_sha256=_HASHES[9],
        base_catalog_sha256=_HASHES[6],
    )
    return SimpleNamespace(
        processor=processor,
        authorization=SimpleNamespace(authorization_id="portfolio-owner-v2"),
        authorization_file_sha256=_HASHES[7],
        receipt_file_sha256=_HASHES[10],
        receipt=receipt,
        dataset_assets_sha256=_HASHES[11],
        plan_sha256=_HASHES[0],
        query_artifact_sha256=_HASHES[3],
        catalog=SimpleNamespace(catalog_sha256=_HASHES[12]),
    )


def _inputs():
    query = _query()
    assistant_query = build_assistant_query_input(query)
    asset = PortfolioQueryAssetBinding(
        query_id=query.query_id,
        asset_id=query.asset_id,
        image_path=query.image_path,
        image_sha256=(
            "e240c2ff6b95c29ac81fe7f56a95d46c497431fcecb12d07736369c64c7c9134"
        ),
    )
    processors = (
        "dashscope-qwen-assistant",
        "dashscope-kimi-feedback",
        "dashscope-kimi-judge",
    )
    return SimpleNamespace(
        queries=(query,),
        assistant_queries=(assistant_query,),
        query_assets=(asset,),
        remote_runtimes=tuple(_runtime(item) for item in processors),
        expected_plan_sha256=_HASHES[0],
        expected_plan_manifest_file_sha256=_HASHES[1],
        expected_accepted_ledger_sha256=_HASHES[2],
        expected_query_artifact_sha256=_HASHES[3],
        expected_capability_assignments_sha256=_HASHES[4],
        expected_seed_set_sha256=_HASHES[5],
        expected_base_catalog_sha256=_HASHES[6],
        expected_output_catalog_sha256=_HASHES[12],
    )


@pytest.fixture
def fake_inputs(monkeypatch: pytest.MonkeyPatch):
    inputs = _inputs()
    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_checklist."
        "require_verified_portfolio_dev_mini_inputs",
        lambda value: value,
    )
    return inputs


def _resign_checklist(payload: dict) -> dict:
    unsigned = deepcopy(payload)
    unsigned.pop("checklist_sha256", None)
    payload["checklist_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    return payload


def test_builder_emits_exact_blocked_zero_call_matrix(fake_inputs) -> None:
    checklist = build_portfolio_five_config_run_checklist(
        fake_inputs,
        matrix_run_id="portfolio-dev-mini-smoke-dm-019-v1",
    )

    assert checklist.selected_query.query_id == "dm-019"
    assert checklist.config_order == MAIN_CONFIG_ORDER
    assert tuple(item.config for item in checklist.instances) == MAIN_CONFIG_ORDER
    assert tuple(item.config_ordinal for item in checklist.instances) == tuple(
        range(5)
    )
    assert checklist.model_calls_performed == 0
    assert checklist.execution_authorized is False
    assert checklist.status == "planned_no_model_calls"
    assert checklist.assistant_runtime_lock_sha256 is None
    assert all(item.assistant_runtime_lock_sha256 is None for item in checklist.instances)
    assert all(item.bank_sha256 is None for item in checklist.instances)
    assert all(item.request_sha256 is None for item in checklist.instances)
    assert tuple(item.blockers for item in checklist.instances) == (
        ("assistant_runtime_lock_missing",),
        ("assistant_runtime_lock_missing", "llm_static_bank_missing"),
        ("assistant_runtime_lock_missing", "s1_bank_missing"),
        ("assistant_runtime_lock_missing", "s1s2_bank_missing"),
        ("assistant_runtime_lock_missing", "full_bank_missing"),
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["instances"].reverse(),
        lambda value: value["instances"].__setitem__(1, value["instances"][0]),
        lambda value: value["instances"].pop(),
        lambda value: value.__setitem__("model_calls_performed", 1),
        lambda value: value.__setitem__("execution_authorized", True),
        lambda value: value.__setitem__(
            "assistant_runtime_lock_sha256",
            "f" * 64,
        ),
        lambda value: value["instances"][1].__setitem__("bank_sha256", "f" * 64),
        lambda value: value["instances"][1].__setitem__(
            "request_sha256",
            "f" * 64,
        ),
    ),
)
def test_resigned_semantic_tampering_is_rejected(fake_inputs, mutate) -> None:
    checklist = build_portfolio_five_config_run_checklist(
        fake_inputs,
        matrix_run_id="portfolio-dev-mini-smoke-dm-019-v1",
    )
    payload = checklist.model_dump(mode="json")
    mutate(payload)
    _resign_checklist(payload)

    with pytest.raises(ValidationError):
        PortfolioFiveConfigRunChecklist.model_validate(payload, strict=True)


def test_create_load_require_and_overwrite_rejection(
    tmp_path: Path,
    fake_inputs,
) -> None:
    checklist = build_portfolio_five_config_run_checklist(
        fake_inputs,
        matrix_run_id="portfolio-dev-mini-smoke-dm-019-v1",
    )
    path = tmp_path / "run-checklist.json"
    created = create_portfolio_five_config_run_checklist(
        path,
        checklist,
        inputs=fake_inputs,
    )
    verified = load_verified_portfolio_five_config_run_checklist(
        path,
        expected_file_sha256=created.file_sha256,
        inputs=fake_inputs,
    )

    assert (
        require_verified_portfolio_five_config_run_checklist(
            verified,
            inputs=fake_inputs,
        )
        is verified
    )
    with pytest.raises(FileExistsError):
        create_portfolio_five_config_run_checklist(
            path,
            checklist,
            inputs=fake_inputs,
        )
    with pytest.raises(PortfolioChecklistError, match="external digest"):
        load_verified_portfolio_five_config_run_checklist(
            path,
            expected_file_sha256="f" * 64,
            inputs=fake_inputs,
        )
