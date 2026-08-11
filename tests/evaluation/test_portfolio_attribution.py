from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_portfolio_stage_gate import _load_smoke
from skillchain.evaluation.assistant_runs import MVP_TOOL_NAMES_V2
from skillchain.evaluation.evaluator_outputs import (
    FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
    FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
    parse_final_judge_output_v3,
)
from skillchain.evaluation.final_runtime import FinalJudgeEvaluationResult
from skillchain.evaluation.packets import JudgeOutcome, build_judge_scores
from skillchain.evaluation.portfolio_attribution import (
    PORTFOLIO_ATTRIBUTION_CAPABILITIES,
    PortfolioAttributionError,
    PortfolioBodyAttributionRecord,
    PortfolioRouteAttributionRecord,
    build_portfolio_parent_attribution,
    load_portfolio_parent_attribution_packet,
    verify_portfolio_parent_attribution_gate_binding,
)
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
    PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION,
    build_portfolio_stage_gate_result_set,
)
from skillchain.llm import LLMUsage
from skillchain.static_authoring import (
    BankCapabilityBinding,
    CompilerIdentity,
    StaticBankArtifact,
    StrictSkillArtifact,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)


OPTIMIZATION_QUERY_IDS = tuple(f"dm-{index:03d}" for index in range(1, 26))
QUERY_IDS = OPTIMIZATION_QUERY_IDS
CAPABILITY_BY_QUERY = {
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
CAPABILITIES = tuple(CAPABILITY_BY_QUERY[query_id] for query_id in QUERY_IDS)
SMOKE_POLICY = PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION


def _hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _self_hashed(payload: dict, field: str) -> dict:
    return {**payload, field: _hash(payload)}


def _bank() -> StaticBankArtifact:
    skills: list[StrictSkillArtifact] = []
    for capability in PORTFOLIO_ATTRIBUTION_CAPABILITIES:
        slug = capability.replace(".", "-").replace("_", "-")
        payload = {
            "slug": slug,
            "version": 1,
            "description": f"Handle {capability} requests.",
            "body": (
                "# Objective\n\nUse visible evidence.\n\n"
                "## Output contract\n\n"
                "- required_sections: answer,evidence,uncertainty\n\n"
                "- card_requirement: forbidden\n\n"
                "- card_fields:\n"
            ),
            "static_refs": [],
            "operators": [],
            "capability_id": capability,
            "parent_skill_sha256": None,
        }
        skills.append(
            StrictSkillArtifact.model_validate(
                _self_hashed(payload, "skill_sha256"),
                strict=True,
            )
        )
    skills.sort(key=lambda item: item.slug)
    by_capability = {item.capability_id: item.slug for item in skills}
    payload = {
        "schema_version": 2,
        "baseline_kind": "llm_static",
        "construction_identity_sha256": "1" * 64,
        "construction_identity_policy": "reviewed-draft-v1",
        "runtime_binding_policy": "registry-runtime-v1",
        "compiler": CompilerIdentity().model_dump(mode="json"),
        "tool_registry_sha256": "2" * 64,
        "tool_registry_runtime_sha256": "3" * 64,
        "skills": [item.model_dump(mode="json") for item in skills],
        "capability_map": [
            BankCapabilityBinding(
                capability_id=capability,
                skill_slug=by_capability[capability],
            ).model_dump(mode="json")
            for capability in sorted(by_capability)
        ],
    }
    return StaticBankArtifact.model_validate(
        _self_hashed(payload, "bank_sha256"),
        strict=True,
    )


def _request(
    *,
    config: str,
    bank_sha256: str,
    query_id: str,
    ordinal: int,
) -> dict:
    backbone_payload = {
        "provider": "qwen",
        "model": "fixture-qwen",
        "endpoint": "https://example.invalid/v1",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": None,
        "system_prompt_sha256": "4" * 64,
    }
    backbone = _self_hashed(backbone_payload, "identity_sha256")
    budget_payload = {
        "max_input_tokens": 1024,
        "max_output_tokens": 256,
        "max_tool_calls": 1,
        "max_turns": 2,
        "timeout_ms": 1000,
    }
    budget = _self_hashed(budget_payload, "budget_sha256")
    registry_payload = {
        "schema_version": 2,
        "policy_version": "assistant-registry-runtime-v2",
        "registry_sha256": "5" * 64,
        "registry_runtime_sha256": "6" * 64,
        "tools": [
            {
                "tool_name": tool_name,
                "tool_spec_sha256": "7" * 64,
                "runtime_binding_sha256": "8" * 64,
            }
            for tool_name in sorted(MVP_TOOL_NAMES_V2)
        ],
    }
    registry = _self_hashed(registry_payload, "lock_sha256")
    public_input = canonical_json_bytes(
        {
            "asset_id": f"asset-{ordinal}",
            "image_path": f"query/{ordinal}.jpg",
            "text": "fixture request",
        }
    ).decode("utf-8")
    query_payload = {
        "query_id": query_id,
        "public_input_json": public_input,
        "public_input_sha256": sha256_bytes(public_input.encode("utf-8")),
    }
    query = _self_hashed(query_payload, "query_sha256")
    treatment = {
        "config": config,
        "bank_sha256": bank_sha256,
        "router_stage": "s1" if config == "s1" else "s2",
        "body_stage": "s1",
    }
    payload = {
        "schema_version": 1,
        "matrix_run_id": f"fixture-{config}",
        "config": config,
        "query_ordinal": ordinal,
        "query": query,
        "treatment": treatment,
        "backbone": backbone,
        "budget": budget,
        "registry": registry,
    }
    return _self_hashed(payload, "request_sha256")


def _assistant_artifact(
    *,
    config: str,
    bank_sha256: str,
    query_id: str,
    capability: str,
    ordinal: int,
    success: bool,
) -> tuple[dict, dict]:
    request = _request(
        config=config,
        bank_sha256=bank_sha256,
        query_id=query_id,
        ordinal=ordinal,
    )
    selected = capability if success else None
    route_sha = hashlib.sha256(f"route-{ordinal}".encode()).hexdigest()
    usage = {"input_tokens": 11 if success else 0, "output_tokens": 5 if success else 0}
    response = {
        "schema_version": 1,
        "request_sha256": request["request_sha256"],
        "backbone_provider": request["backbone"]["provider"],
        "backbone_model": request["backbone"]["model"],
        "backbone_endpoint": request["backbone"]["endpoint"],
        "backbone_identity_sha256": request["backbone"]["identity_sha256"],
        "registry_sha256": request["registry"]["registry_sha256"],
        "registry_runtime_sha256": request["registry"]["registry_runtime_sha256"],
        "budget_sha256": request["budget"]["budget_sha256"],
        "response_text": (
            "answer: Grounded fixture.\nevidence: Visible evidence.\nuncertainty: None."
            if success
            else ""
        ),
        "visible_cards": [],
        "visible_tool_evidence": (
            [
                {
                    "tool_name": "document_ocr",
                    "status": "success",
                    "visible_text": "Grounded fixture evidence.",
                    "cards": [],
                    "citations": [],
                    "detections": [],
                    "error_code": None,
                }
            ]
            if success
            else []
        ),
        "tool_trace": [],
        "selected_capability": selected,
        "skill_slug": (
            capability.replace(".", "-").replace("_", "-") if success else None
        ),
        "route_trace_sha256": route_sha if success else None,
        "backbone_request_id": "provider-request" if success else None,
        "usage": usage,
        "turn_count": 1,
        "latency_ms": 3,
        "error_code": None if success else "runtime_error",
    }
    model_calls = (
        [
            {
                "call_index": 1,
                "provider": "qwen",
                "endpoint": request["backbone"]["endpoint"],
                "requested_model": request["backbone"]["model"],
                "response_model": request["backbone"]["model"],
                "provider_request_id": "provider-request",
                "input_tokens": 11,
                "output_tokens": 5,
                "finish_reason": "stop",
                "latency_ms": 2,
                "response_sha256": "9" * 64,
            }
        ]
        if success
        else []
    )
    receipt_payload = {
        "schema_version": 1,
        "policy_version": "runner-owned-assistant-v1",
        "request_sha256": request["request_sha256"],
        "asset_catalog_sha256": "a" * 64 if success else None,
        "query_asset_id": f"asset-{ordinal}" if success else None,
        "query_asset_sha256": "b" * 64 if success else None,
        "shared_route_reference": None,
        "model_calls": model_calls,
        "tool_trace": [],
        "aggregate_usage": usage,
        "runner_latency_ms": 3,
        "outcome": "success" if success else "runtime_error",
        "response_sha256": _hash(response),
    }
    receipt_hash_payload = dict(receipt_payload)
    receipt_hash_payload.pop("shared_route_reference")
    receipt = {
        **receipt_payload,
        "receipt_sha256": _hash(receipt_hash_payload),
    }
    payload = {
        "schema_version": 1,
        "kind": "portfolio-treatment-smoke-assistant",
        "policy_version": SMOKE_POLICY,
        "query_id": query_id,
        "config": config,
        "request": request,
        "response": response,
        "receipt": receipt,
        "shared_route_file": None,
        "shared_route_file_sha256": None,
    }
    return _self_hashed(payload, "row_sha256"), response


def _scored_final() -> FinalJudgeEvaluationResult:
    evaluation_id = "c" * 64
    prompt_sha256 = "d" * 64
    raw_response = json.dumps(
        {
            "schema_version": 1,
            "requires_card": False,
            "dimensions": [
                {"dimension": "CA", "score": 8},
                {"dimension": "CQ", "score": 16},
                {"dimension": "TCR", "score": 8},
            ],
        },
        separators=(",", ":"),
    )
    parsed = parse_final_judge_output_v3(
        raw_response,
        expected_requires_card=False,
    )
    scores = build_judge_scores(
        evaluation_id=evaluation_id,
        requires_card=False,
        raw_scores={"CA": 8, "CQ": 16, "TCR": 8},
    )
    raw_sha = sha256_bytes(raw_response.encode("utf-8"))
    outcome = JudgeOutcome(
        evaluation_id=evaluation_id,
        status="scored",
        attempts=1,
        max_attempts=1,
        scores=scores,
        judge_provider="kimi",
        judge_model="kimi-k2.6",
        prompt_sha256=prompt_sha256,
        raw_response_sha256=raw_sha,
        error_code=None,
    )
    payload = {
        "schema_version": 3,
        "result_kind": "visual-final-judge",
        "cache_namespace": "final-evaluator-v4",
        "formal_eligible": False,
        "evaluation_id": evaluation_id,
        "packet_sha256": "e" * 64,
        "prompt_sha256": prompt_sha256,
        "image_sha256": "f" * 64,
        "wire_sha256": "0" * 64,
        "asset_catalog_sha256": "1" * 64,
        "remote_authorization_id": "fixture-authorization",
        "remote_authorization_file_sha256": "2" * 64,
        "remote_receipt_file_sha256": "3" * 64,
        "remote_receipt_sha256": "4" * 64,
        "provider": "kimi",
        "model": "kimi-k2.6",
        "endpoint": "https://example.invalid/v1",
        "max_tokens": 256,
        "attempts": 1,
        "max_attempts": 1,
        "request_id": "judge-request",
        "raw_response_text": raw_response,
        "raw_response_sha256": raw_sha,
        "raw_response_bytes": len(raw_response.encode("utf-8")),
        "tool_calls": (),
        "tool_call_count": 0,
        "response_redaction_reason": None,
        "parsed_submission": parsed.submission,
        "parser_policy_version": FINAL_JUDGE_PARSER_POLICY_VERSION_V3,
        "parser_policy_sha256": FINAL_JUDGE_PARSER_POLICY_SHA256_V3,
        "raw_dimensions_shape": parsed.raw_dimensions_shape,
        "canonical_submission_sha256": sha256_bytes(
            canonical_json_bytes(parsed.submission.model_dump(mode="json"))
        ),
        "usage": LLMUsage(input_tokens=20, output_tokens=8),
        "finish_reason": "stop",
        "latency_ms": 4,
        "outcome": outcome,
    }
    unsigned = FinalJudgeEvaluationResult.model_construct(
        **payload,
        result_sha256="0" * 64,
    )
    result_sha256 = sha256_bytes(
        canonical_json_bytes(
            unsigned.model_dump(mode="json", exclude={"result_sha256"})
        )
    )
    return FinalJudgeEvaluationResult.model_validate(
        {**payload, "result_sha256": result_sha256},
        strict=True,
    )


def _write_smoke(
    root: Path,
    *,
    config: str,
    bank: StaticBankArtifact,
    scored_first: bool = False,
    legacy_double_lf: bool = False,
) -> None:
    (root / "assistant").mkdir(parents=True)
    (root / "final").mkdir()
    rows = []
    for ordinal, (query_id, capability) in enumerate(
        zip(QUERY_IDS, CAPABILITIES, strict=True)
    ):
        success = scored_first and ordinal == 0
        assistant, response = _assistant_artifact(
            config=config,
            bank_sha256=bank.bank_sha256,
            query_id=query_id,
            capability=capability,
            ordinal=ordinal,
            success=success,
        )
        assistant_content = canonical_json_bytes(assistant)
        (root / "assistant" / f"{query_id}.json").write_bytes(assistant_content)
        if success:
            final_model = _scored_final()
            final = final_model.model_dump(mode="json")
            final_status = "scored"
            assistant_error = None
            j_project = final_model.outcome.scores.j_project
        else:
            final_payload = {
                "schema_version": 1,
                "kind": "portfolio-treatment-smoke-final-fixed-zero",
                "query_id": query_id,
                "assistant_error_code": "runtime_error",
                "judge_invoked": False,
                "j_project": 0.0,
            }
            final = _self_hashed(final_payload, "result_sha256")
            final_status = "assistant_fixed_zero"
            assistant_error = "runtime_error"
            j_project = 0.0
        final_content = canonical_json_bytes(final)
        (root / "final" / f"{query_id}.json").write_bytes(final_content)
        row_payload = {
            "schema_version": 1,
            "kind": "portfolio-treatment-smoke-result",
            "policy_version": SMOKE_POLICY,
            "query_id": query_id,
            "canonical_capability": capability,
            "config": config,
            "target_bank_sha256": bank.bank_sha256,
            "assistant_file": f"assistant/{query_id}.json",
            "assistant_file_sha256": sha256_bytes(assistant_content),
            "final_file": f"final/{query_id}.json",
            "final_file_sha256": sha256_bytes(final_content),
            "shared_route_file": None,
            "shared_route_file_sha256": None,
            "selected_capability": response["selected_capability"],
            "route_correct": success,
            "assistant_error_code": assistant_error,
            "final_status": final_status,
            "hard_error": not success,
            "skill_adherence": 1.0 if success else 0.0,
            "j_project": j_project,
            "usage_and_cost": {},
        }
        rows.append(_self_hashed(row_payload, "result_sha256"))
    separator = b"\n" if legacy_double_lf else b""
    results_content = b"".join(canonical_json_bytes(row) + separator for row in rows)
    (root / "results.jsonl").write_bytes(results_content)
    summary_payload = {
        "schema_version": 1,
        "kind": "portfolio-treatment-development-smoke-summary",
        "policy_version": SMOKE_POLICY,
        "status": "complete",
        "config": config,
        "target_bank_sha256": bank.bank_sha256,
        "query_count_requested": len(QUERY_IDS),
        "query_count_completed": len(QUERY_IDS),
        "query_ids_requested": list(QUERY_IDS),
        "query_ids_completed": list(QUERY_IDS),
        "capabilities_completed": list(CAPABILITIES),
        "results_file": "results.jsonl",
        "results_file_sha256": sha256_bytes(results_content),
        "route_accuracy": sum(row["route_correct"] for row in rows) / len(rows),
        "mean_j_project": sum(row["j_project"] for row in rows) / len(rows),
        "hard_error_count": sum(row["hard_error"] for row in rows),
        "mean_skill_adherence": sum(row["skill_adherence"] for row in rows) / len(rows),
        "rubric_file_sha256": PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
    }
    (root / "summary.json").write_bytes(
        canonical_json_bytes(_self_hashed(summary_payload, "summary_sha256"))
    )


def _rewrite_summary(root: Path, changes: dict | None = None) -> None:
    summary = parse_canonical_json(
        (root / "summary.json").read_bytes(),
        label="summary",
    )
    assert isinstance(summary, dict)
    summary.pop("summary_sha256")
    summary.update(changes or {})
    (root / "summary.json").write_bytes(
        canonical_json_bytes(_self_hashed(summary, "summary_sha256"))
    )


def _rewrite_results(root: Path, rows: list[dict]) -> None:
    content = b"".join(canonical_json_bytes(row) for row in rows)
    (root / "results.jsonl").write_bytes(content)
    _rewrite_summary(root, {"results_file_sha256": sha256_bytes(content)})


def _read_result_rows(root: Path) -> list[dict]:
    rows = []
    for line in (root / "results.jsonl").read_bytes().splitlines(keepends=True):
        if line == b"\n":
            continue
        value = parse_canonical_json(line, label="result")
        assert isinstance(value, dict)
        rows.append(value)
    return rows


def test_builds_current_parent_s2_route_packet_and_round_trips(tmp_path: Path) -> None:
    bank = _bank()
    smoke = tmp_path / "s1-smoke"
    _write_smoke(
        smoke,
        config="s1",
        bank=bank,
    )

    packet = build_portfolio_parent_attribution(
        stage="s2_route_optimizer",
        smoke_root=smoke,
        parent_bank=bank,
        optimization_query_ids=OPTIMIZATION_QUERY_IDS,
    )

    assert packet.source_config == "s1"
    assert packet.source_bank_sha256 == bank.bank_sha256
    assert packet.query_ids == QUERY_IDS
    assert all(
        isinstance(item, PortfolioRouteAttributionRecord) for item in packet.records
    )
    assert {item.canonical_capability for item in packet.records} == set(
        PORTFOLIO_ATTRIBUTION_CAPABILITIES
    )
    result_kwargs = {
        "source_config": "s1",
        "bank_sha256": bank.bank_sha256,
        "adherence_contract_bank_sha256": bank.bank_sha256,
        "evaluation_query_ids": tuple(sorted(packet.query_ids)),
        "route_accuracy": 0.5,
        "mean_j": 50.0,
        "mean_skill_adherence": 0.5,
        "hard_error_count": 1,
        "rubric_file_sha256": PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
        "shared_route_artifact_sha256s": (None,) * len(QUERY_IDS),
        "source_summary_file_sha256": packet.source_summary_file_sha256,
        "source_summary_sha256": packet.source_summary_sha256,
        "source_results_file_sha256": packet.source_results_file_sha256,
        "source_result_sha256s": tuple(
            item.source_result_sha256
            for item in sorted(packet.records, key=lambda item: item.query_id)
        ),
    }
    gate_parent = build_portfolio_stage_gate_result_set(**result_kwargs)
    verify_portfolio_parent_attribution_gate_binding(packet, gate_parent)
    other_smoke = build_portfolio_stage_gate_result_set(
        **{
            **result_kwargs,
            "source_results_file_sha256": "0" * 64,
        }
    )
    with pytest.raises(PortfolioAttributionError, match="gate-parent smoke"):
        verify_portfolio_parent_attribution_gate_binding(packet, other_smoke)
    path = tmp_path / "route-attribution.json"
    content = packet.canonical_bytes()
    path.write_bytes(content)
    assert (
        load_portfolio_parent_attribution_packet(
            path,
            expected_file_sha256=sha256_bytes(content),
        )
        == packet
    )


def test_rejects_legacy_double_lf_smoke_encoding(tmp_path: Path) -> None:
    bank = _bank()
    smoke = tmp_path / "legacy-double-lf"
    _write_smoke(
        smoke,
        config="s1",
        bank=bank,
        legacy_double_lf=True,
    )
    with pytest.raises(PortfolioAttributionError, match="result line 2 is invalid"):
        build_portfolio_parent_attribution(
            stage="s2_route_optimizer",
            smoke_root=smoke,
            parent_bank=bank,
            optimization_query_ids=OPTIMIZATION_QUERY_IDS,
        )


def test_gate_ingest_revalidates_policy_capabilities_and_bound_outputs(
    tmp_path: Path,
) -> None:
    bank = _bank()
    valid = tmp_path / "valid-gate-smoke"
    _write_smoke(valid, config="s1", bank=bank)
    result, _ = _load_smoke(
        valid,
        expected_config="s1",
        expected_bank=bank,
        adherence_contract_bank=bank,
    )
    assert result.evaluation_query_ids == QUERY_IDS
    assert result.mean_skill_adherence == 0.0

    stale_policy = tmp_path / "stale-policy"
    _write_smoke(stale_policy, config="s1", bank=bank)
    _rewrite_summary(
        stale_policy,
        {"policy_version": "portfolio-treatment-development-smoke-v1"},
    )
    with pytest.raises(ValueError, match="summary binding mismatch"):
        _load_smoke(
            stale_policy,
            expected_config="s1",
            expected_bank=bank,
            adherence_contract_bank=bank,
        )

    wrong_capabilities = tmp_path / "wrong-capabilities"
    _write_smoke(wrong_capabilities, config="s1", bank=bank)
    rows = _read_result_rows(wrong_capabilities)
    first = dict(rows[0])
    first.pop("result_sha256")
    first["canonical_capability"] = "product.style_recommendation"
    rows[0] = _self_hashed(first, "result_sha256")
    _rewrite_results(wrong_capabilities, rows)
    capabilities = list(CAPABILITIES)
    capabilities[0] = "product.style_recommendation"
    _rewrite_summary(
        wrong_capabilities,
        {"capabilities_completed": capabilities},
    )
    with pytest.raises(ValueError, match="denominator mismatch"):
        _load_smoke(
            wrong_capabilities,
            expected_config="s1",
            expected_bank=bank,
            adherence_contract_bank=bank,
        )

    assistant_tamper = tmp_path / "assistant-tamper"
    _write_smoke(assistant_tamper, config="s1", bank=bank)
    assistant_path = assistant_tamper / "assistant" / "dm-001.json"
    assistant_path.write_bytes(assistant_path.read_bytes() + b" ")
    with pytest.raises(PortfolioAttributionError, match="Assistant"):
        _load_smoke(
            assistant_tamper,
            expected_config="s1",
            expected_bank=bank,
            adherence_contract_bank=bank,
        )

    final_tamper = tmp_path / "final-tamper"
    _write_smoke(final_tamper, config="s1", bank=bank)
    final_path = final_tamper / "final" / "dm-001.json"
    final_path.write_bytes(final_path.read_bytes() + b" ")
    with pytest.raises(PortfolioAttributionError, match="final"):
        _load_smoke(
            final_tamper,
            expected_config="s1",
            expected_bank=bank,
            adherence_contract_bank=bank,
        )


def test_builds_s3_response_and_score_attribution(tmp_path: Path) -> None:
    bank = _bank()
    smoke = tmp_path / "s1s2-smoke"
    _write_smoke(smoke, config="s1s2", bank=bank, scored_first=True)

    packet = build_portfolio_parent_attribution(
        stage="s3_body_refiner",
        smoke_root=smoke,
        parent_bank=bank,
        optimization_query_ids=OPTIMIZATION_QUERY_IDS,
    )

    assert packet.source_config == "s1s2"
    first = packet.records[0]
    assert isinstance(first, PortfolioBodyAttributionRecord)
    assert first.response_text.startswith("answer: Grounded fixture.")
    assert first.judge_status == "scored"
    assert first.j_project == 80.0
    assert [item.dimension for item in first.dimensions] == ["CA", "CQ", "TCR"]
    fixed = packet.records[1]
    assert isinstance(fixed, PortfolioBodyAttributionRecord)
    assert fixed.final_kind == "assistant_fixed_zero"
    assert fixed.j_project == 0.0
    path = tmp_path / "body-attribution.json"
    content = packet.canonical_bytes()
    path.write_bytes(content)
    assert (
        load_portfolio_parent_attribution_packet(
            path,
            expected_file_sha256=sha256_bytes(content),
        )
        == packet
    )


def test_rejects_summary_self_hash_tampering(tmp_path: Path) -> None:
    bank = _bank()
    smoke = tmp_path / "smoke"
    _write_smoke(smoke, config="s1", bank=bank)
    summary = parse_canonical_json(
        (smoke / "summary.json").read_bytes(),
        label="summary",
    )
    assert isinstance(summary, dict)
    summary["target_bank_sha256"] = "f" * 64
    (smoke / "summary.json").write_bytes(canonical_json_bytes(summary))

    with pytest.raises(PortfolioAttributionError, match="summary_sha256"):
        build_portfolio_parent_attribution(
            stage="s2_route_optimizer",
            smoke_root=smoke,
            parent_bank=bank,
            optimization_query_ids=OPTIMIZATION_QUERY_IDS,
        )


def test_rejects_result_self_hash_tampering(tmp_path: Path) -> None:
    bank = _bank()
    smoke = tmp_path / "smoke"
    _write_smoke(smoke, config="s1", bank=bank)
    rows = _read_result_rows(smoke)
    rows[0]["selected_capability"] = "product.exact_match"
    rows[0]["route_correct"] = True
    _rewrite_results(smoke, rows)

    with pytest.raises(PortfolioAttributionError, match="result line"):
        build_portfolio_parent_attribution(
            stage="s2_route_optimizer",
            smoke_root=smoke,
            parent_bank=bank,
            optimization_query_ids=OPTIMIZATION_QUERY_IDS,
        )


def test_rejects_assistant_and_final_self_hash_tampering(tmp_path: Path) -> None:
    bank = _bank()
    for target in ("assistant", "final"):
        smoke = tmp_path / target
        _write_smoke(smoke, config="s1", bank=bank)
        rows = _read_result_rows(smoke)
        path = smoke / rows[0][f"{target}_file"]
        artifact = parse_canonical_json(path.read_bytes(), label=target)
        assert isinstance(artifact, dict)
        if target == "assistant":
            artifact["response"]["latency_ms"] += 1
        else:
            artifact["j_project"] = 1.0
        content = canonical_json_bytes(artifact)
        path.write_bytes(content)
        rows[0][f"{target}_file_sha256"] = sha256_bytes(content)
        rows[0].pop("result_sha256")
        rows[0] = _self_hashed(rows[0], "result_sha256")
        _rewrite_results(smoke, rows)

        with pytest.raises(PortfolioAttributionError):
            build_portfolio_parent_attribution(
                stage="s2_route_optimizer",
                smoke_root=smoke,
                parent_bank=bank,
                optimization_query_ids=OPTIMIZATION_QUERY_IDS,
            )


def test_rejects_stale_parent_and_query_outside_optimization_25(
    tmp_path: Path,
) -> None:
    bank = _bank()
    smoke = tmp_path / "smoke"
    _write_smoke(smoke, config="s1", bank=bank)
    stale = bank.model_copy(update={"bank_sha256": "f" * 64})
    with pytest.raises(PortfolioAttributionError, match="current parent Bank"):
        build_portfolio_parent_attribution(
            stage="s2_route_optimizer",
            smoke_root=smoke,
            parent_bank=stale,
            optimization_query_ids=OPTIMIZATION_QUERY_IDS,
        )

    optimization_without_source = tuple(
        item for item in OPTIMIZATION_QUERY_IDS if item != "dm-001"
    ) + ("dm-026",)
    with pytest.raises(PortfolioAttributionError, match="frozen dm-001..dm-025"):
        build_portfolio_parent_attribution(
            stage="s2_route_optimizer",
            smoke_root=smoke,
            parent_bank=bank,
            optimization_query_ids=optimization_without_source,
        )
