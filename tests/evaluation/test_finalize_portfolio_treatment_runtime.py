from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import finalize_portfolio_treatment_runtime as finalizer_module
from scripts.finalize_portfolio_treatment_runtime import (
    FinalizationInputs,
    GateSource,
    PortfolioTreatmentFinalizationError,
    StageSource,
    _load_evolution_source,
    _load_gate,
    _require_current_mutation_implementation,
    _runtime_lock,
    finalize_portfolio_treatment_runtime,
)
from scripts.prepare_portfolio_evolution_inputs import (
    CapabilityCount,
    EXPECTED_CAPABILITIES,
    PortfolioEvolutionInputManifest,
    PortfolioEvolutionInputPacket,
    S1TrajectoryRecord,
    S2RouteAttributionRecord,
    S3BodyAttributionRecord,
)
from scripts.run_portfolio_evolution_model import (
    CodexProcessResult,
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    INVOCATION_POLICY_VERSION,
    S1_SPARSE_IMPLEMENTATION_VERSION,
    S1_SPARSE_INVOCATION_POLICY_VERSION,
    S3_IMPLEMENTATION_VERSION,
    S3_INVOCATION_POLICY_VERSION,
    _actionable_mutation_scope,
    run_portfolio_evolution_model,
)
from skillchain.evaluation.final_runtime import (
    CARD_REQUIREMENT_GUARD_POLICY_SHA256,
    CARD_REQUIREMENT_GUARD_POLICY_VERSION,
    FINAL_JUDGE_CACHE_NAMESPACE,
    FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE,
    FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE,
    FINAL_JUDGE_RESULT_SCHEMA_VERSION,
    FINAL_JUDGE_RETRY_POLICY_VERSION,
    FINAL_JUDGE_THINKING_BUDGET,
    FINAL_JUDGE_TRANSPORT_POLICY_SHA256,
    FINAL_JUDGE_TRANSPORT_POLICY_VERSION,
)
from skillchain.evaluation.portfolio_launch import (
    PORTFOLIO_BUDGET_POLICY_SHA256,
    PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256,
)
from skillchain.evaluation.portfolio_execution import (
    PORTFOLIO_BUDGET_POLICY_VERSION,
    PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS,
)
from skillchain.evaluation.portfolio_attribution import (
    PORTFOLIO_OPTIMIZATION_QUERY_IDS,
    PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION,
    PortfolioBodyAttributionRecord,
    PortfolioParentAttributionPacket,
    PortfolioRouteAttributionRecord,
)
from skillchain.evaluation.portfolio_treatment_io import (
    load_verified_portfolio_treatment_runtime,
)
from skillchain.evaluation.portfolio_s3_textopt import (
    build_empty_portfolio_s3_rejected_edit_buffer,
)


from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
    PortfolioTreatmentError,
    build_portfolio_stage_gate_result_set,
    build_portfolio_stage_gate_report,
)
from skillchain.static_authoring import StaticBankArtifact
from skillchain.tools.portfolio_runtime import (
    PORTFOLIO_SYSTEM_PROMPT,
    PORTFOLIO_TOOL_RUNTIME_POLICY,
    PortfolioRuntimeSources,
    build_portfolio_tool_runtime,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_INPUT = (
    ROOT / "specs" / "authoring" / "authoring-packet-primary-v5-candidate.json"
)
CODEX_INPUT = ROOT / "specs" / "authoring" / "authoring-packet-codex-high-v5.json"
STATIC_ROOT = (
    ROOT / "runs" / "formal-authoring" / "llm-static-codex-primary-20260724-high-v5"
)
STATIC_REVIEW = (
    ROOT
    / "specs"
    / "authoring"
    / "llm-static-codex-primary-20260724-high-v5-human-review.json"
)
AUTHOR_RAW = STATIC_ROOT / "author-content.raw.json"
OPTIMIZATION_BATCH_ID = "dev-mini-001-r3"
RUBRIC_SHA = PORTFOLIO_FINAL_RUBRIC_FILE_SHA256
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


def test_runtime_lock_binds_active_gemini_transport_and_pricing_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SimpleNamespace(
        registry_sha256="a" * 64,
        registry_runtime_sha256="b" * 64,
        specs=lambda: (),
    )
    tool = SimpleNamespace(
        runtime=SimpleNamespace(
            registry=registry,
            index=SimpleNamespace(
                runtime_data_sha256="c" * 64,
                source_sha256s=("d" * 64,),
            ),
        ),
        source_lock_file_sha256="e" * 64,
        source_lock_sha256="f" * 64,
        recipe_rows=1,
    )
    outputs = {
        config: SimpleNamespace(
            bank_sha256=str(index) * 64,
            canonical_bytes=lambda index=index: bytes([index]),
        )
        for index, config in enumerate(("llm_static", "s1", "s1s2", "full"), 1)
    }
    records = tuple(
        SimpleNamespace(config=config, receipt_sha256="9" * 64) for config in outputs
    )
    manifest = SimpleNamespace(chain_sha256="8" * 64)
    monkeypatch.setattr(
        finalizer_module,
        "build_portfolio_execution_artifact_aliases",
        lambda _manifest: (),
    )

    lock = _runtime_lock(
        tool=tool,
        outputs=outputs,
        records=records,
        manifest=manifest,
        manifest_file_sha256="7" * 64,
        iteration_report_sha256="6" * 64,
    )

    assert lock["final_judge_transport_policy_version"] == (
        FINAL_JUDGE_TRANSPORT_POLICY_VERSION
    )
    assert lock["final_judge_transport_policy_sha256"] == (
        FINAL_JUDGE_TRANSPORT_POLICY_SHA256
    )
    assert lock["final_judge_requested_response_format"] == "json_object"
    assert lock["final_judge_provider_pricing_status"] == (
        PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS
    )


def _file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _self_hashed(payload: dict, field: str) -> dict:
    return {
        **payload,
        field: sha256_bytes(canonical_json_bytes(payload)),
    }


def _clean_events(message: str, *, thread_id: str) -> bytes:
    events = (
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started", "thread_id": thread_id},
        {
            "type": "item.completed",
            "thread_id": thread_id,
            "item": {
                "id": "message-1",
                "type": "agent_message",
                "text": message,
            },
        },
        {
            "type": "turn.completed",
            "thread_id": thread_id,
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 30,
            },
        },
    )
    return b"".join(canonical_json_bytes(item) for item in events)


class _MockCodex:
    def __init__(
        self,
        raw_output: bytes,
        *,
        thread_id: str = "finalizer-test-thread",
    ):
        self.raw_output = raw_output
        self.thread_id = thread_id
        self.calls = 0

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        stdin: bytes,
        cwd: Path,
        environment,
        timeout_seconds: int,
    ) -> CodexProcessResult:
        del stdin, environment
        assert timeout_seconds > 0
        assert not tuple(cwd.iterdir())
        final_path = Path(command[command.index("--output-last-message") + 1])
        final_path.write_bytes(self.raw_output)
        self.calls += 1
        return CodexProcessResult(
            returncode=0,
            stdout=_clean_events(
                self.raw_output.decode("utf-8"),
                thread_id=self.thread_id,
            ),
            stderr=b"",
        )


def _capability_sequence() -> tuple[str, ...]:
    return tuple(EXPECTED_CAPABILITIES[index % 6] for index in range(25))


def _manifest(
    *,
    stage: str,
    record_kind: str,
    records: tuple[object, ...],
    source_bank_sha256: str | None,
) -> PortfolioEvolutionInputManifest:
    query_ids = tuple(f"dm-{index:03d}" for index in range(1, 26))
    counts = tuple(
        CapabilityCount(
            capability_id=capability,
            count=_capability_sequence().count(capability),
        )
        for capability in EXPECTED_CAPABILITIES
    )
    source_fields = (
        {}
        if stage == "s1"
        else {
            "source_config": "s1" if stage == "s2" else "s1s2",
            "source_execution_control_file_sha256": "1" * 64,
            "source_execution_control_sha256": "2" * 64,
            "source_launch_plan_file_sha256": "3" * 64,
            "source_launch_plan_sha256": "4" * 64,
            "source_runtime_lock_file_sha256": "5" * 64,
            "source_runtime_lock_sha256": "6" * 64,
            "source_shard_id": f"test-{stage}-shard",
            "source_shard_sha256": "7" * 64,
            "source_shard_summary_file_sha256": "8" * 64,
            "source_shard_summary_sha256": "9" * 64,
            "source_shard_audit_file_sha256": "a" * 64,
            "source_shard_audit_sha256": "b" * 64,
            "source_bank_sha256": source_bank_sha256,
        }
    )
    payload = {
        "schema_version": 1,
        "kind": "portfolio-evolution-input-manifest",
        "policy_version": "portfolio-evolution-input-v1",
        "track": "portfolio",
        "formal_eligible": False,
        "stage": stage,
        "component": {
            "s1": "creator",
            "s2": "route_optimizer",
            "s3": "body_refiner",
        }[stage],
        "status": "prepared_input_only",
        "source_split": "dev_mini",
        "accepted_batch_id": OPTIMIZATION_BATCH_ID,
        "record_kind": record_kind,
        "record_count": 25,
        "query_ids": query_ids,
        "query_order_sha256": sha256_bytes(canonical_json_bytes(list(query_ids))),
        "capability_counts": counts,
        "records_file": "records.jsonl",
        "records_file_sha256": sha256_bytes(
            b"".join(
                canonical_json_bytes(item.model_dump(mode="json")) for item in records
            )
        ),
        "portfolio_plan_sha256": "c" * 64,
        "accepted_ledger_sha256": "d" * 64,
        "query_artifact_sha256": "e" * 64,
        "capability_assignments_sha256": "f" * 64,
        "seed_set_sha256": "0" * 64,
        "runtime_catalog_sha256": "1" * 64,
        **source_fields,
        "model_calls_performed": 0,
        "evolution_component_invoked": False,
        "bank_generated": False,
        "execution_authorized": False,
    }
    draft = PortfolioEvolutionInputManifest.model_construct(
        **payload,
        manifest_sha256="0" * 64,
    )
    normalized = draft.model_dump(
        mode="json",
        exclude={"manifest_sha256"},
    )
    return PortfolioEvolutionInputManifest.model_validate(
        {
            **normalized,
            "manifest_sha256": sha256_bytes(canonical_json_bytes(normalized)),
        },
        strict=True,
    )


def _packet(
    *,
    stage: str,
    source_bank_sha256: str | None = None,
) -> PortfolioEvolutionInputPacket:
    query_ids = tuple(f"dm-{index:03d}" for index in range(1, 26))
    capabilities = _capability_sequence()
    records: list[object] = []
    for ordinal, (query_id, capability) in enumerate(
        zip(query_ids, capabilities, strict=True)
    ):
        if stage == "s1":
            public = canonical_json_bytes(
                {
                    "asset_id": f"asset-{ordinal}",
                    "image_path": f"images/{ordinal}.jpg",
                }
            ).decode("utf-8")
            payload = {
                "schema_version": 1,
                "kind": "portfolio-s1-user-trajectory",
                "record_ordinal": ordinal,
                "source_query_ordinal": ordinal,
                "accepted_batch_id": OPTIMIZATION_BATCH_ID,
                "query_id": query_id,
                "canonical_capability": capability,
                "canonical_intent": capability,
                "asset_id": f"asset-{ordinal}",
                "image_path": f"images/{ordinal}.jpg",
                "image_sha256": f"{ordinal + 1:064x}",
                "public_input_json": public,
                "public_input_sha256": sha256_bytes(public.encode("utf-8")),
                "query_sha256": f"{ordinal + 101:064x}",
            }
            records.append(
                S1TrajectoryRecord.model_validate(
                    _self_hashed(payload, "record_sha256"),
                    strict=True,
                )
            )
        elif stage == "s2":
            payload = {
                "schema_version": 1,
                "kind": "portfolio-s2-route-attribution",
                "record_ordinal": ordinal,
                "source_query_ordinal": ordinal,
                "accepted_batch_id": OPTIMIZATION_BATCH_ID,
                "query_id": query_id,
                "public_input_sha256": f"{ordinal + 1:064x}",
                "query_sha256": f"{ordinal + 101:064x}",
                "expected_capability": capability,
                "selected_capability": None,
                "route_correct": False,
                "skill_slug": None,
                "route_trace_sha256": None,
                "tool_names": [],
                "tool_trace": [],
                "assistant_error_code": "test_route_error",
                "source_bank_sha256": source_bank_sha256,
                "request_sha256": f"{ordinal + 201:064x}",
                "response_sha256": f"{ordinal + 301:064x}",
                "receipt_sha256": f"{ordinal + 401:064x}",
                "assistant_checkpoint_file_sha256": f"{ordinal + 501:064x}",
            }
            model_payload = {
                **payload,
                "tool_names": (),
                "tool_trace": (),
            }
            records.append(
                S2RouteAttributionRecord.model_validate(
                    {
                        **model_payload,
                        "record_sha256": sha256_bytes(canonical_json_bytes(payload)),
                    },
                    strict=True,
                )
            )
        else:
            payload = {
                "schema_version": 1,
                "kind": "portfolio-s3-body-attribution",
                "record_ordinal": ordinal,
                "source_query_ordinal": ordinal,
                "accepted_batch_id": OPTIMIZATION_BATCH_ID,
                "query_id": query_id,
                "public_input_sha256": f"{ordinal + 1:064x}",
                "query_sha256": f"{ordinal + 101:064x}",
                "canonical_capability": capability,
                "selected_capability": None,
                "skill_slug": None,
                "route_trace_sha256": None,
                "shared_route_artifact_file_sha256": None,
                "response_text": "",
                "visible_cards": [],
                "visible_tool_evidence": [],
                "tool_trace": [],
                "assistant_error_code": "test_assistant_error",
                "source_bank_sha256": source_bank_sha256,
                "request_sha256": f"{ordinal + 201:064x}",
                "response_sha256": f"{ordinal + 301:064x}",
                "receipt_sha256": f"{ordinal + 401:064x}",
                "assistant_checkpoint_file_sha256": f"{ordinal + 501:064x}",
                "final_kind": "assistant_fixed_zero",
                "evaluation_id": None,
                "judge_status": "not_invoked_assistant_error",
                "judge_error_code": None,
                "dimensions": [],
                "j_project": 0.0,
                "final_result_sha256": f"{ordinal + 601:064x}",
                "final_checkpoint_file_sha256": f"{ordinal + 701:064x}",
            }
            model_payload = {
                **payload,
                "visible_cards": (),
                "visible_tool_evidence": (),
                "tool_trace": (),
                "dimensions": (),
            }
            records.append(
                S3BodyAttributionRecord.model_validate(
                    {
                        **model_payload,
                        "record_sha256": sha256_bytes(canonical_json_bytes(payload)),
                    },
                    strict=True,
                )
            )
    typed_records = tuple(records)
    record_kind = {
        "s1": "portfolio-s1-user-trajectory",
        "s2": "portfolio-s2-route-attribution",
        "s3": "portfolio-s3-body-attribution",
    }[stage]
    manifest = _manifest(
        stage=stage,
        record_kind=record_kind,
        records=typed_records,
        source_bank_sha256=source_bank_sha256,
    )
    packet_records = [item.model_dump(mode="json") for item in typed_records]
    payload = {
        "manifest": manifest.model_dump(mode="json"),
        "records": packet_records,
    }
    return PortfolioEvolutionInputPacket.model_validate(
        {
            "manifest": manifest,
            "records": typed_records,
            "packet_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


def _write_packet(path: Path, packet: PortfolioEvolutionInputPacket) -> str:
    content = canonical_json_bytes(packet.model_dump(mode="json"))
    path.write_bytes(content)
    return sha256_bytes(content)


def _write_parent_packet(
    path: Path,
    *,
    stage: str,
    source_bank_sha256: str,
) -> str:
    query_ids = PORTFOLIO_OPTIMIZATION_QUERY_IDS
    records = []
    for ordinal, (query_id, capability) in enumerate(
        ((query_id, CAPABILITY_BY_QUERY[query_id]) for query_id in query_ids)
    ):
        selected_capability = capability
        if stage == "s2_route_optimizer" and query_id in {"dm-001", "dm-004"}:
            selected_capability = "product.multi_search"
        common = {
            "schema_version": 1,
            "record_ordinal": ordinal,
            "query_id": query_id,
            "canonical_capability": capability,
            "selected_capability": selected_capability,
            "route_correct": selected_capability == capability,
            "skill_slug": f"skill-{ordinal}",
            "route_trace_sha256": "1" * 64,
            "assistant_error_code": None,
            "source_result_sha256": sha256_bytes(
                f"source-result:{stage}:{ordinal}".encode()
            ),
            "source_assistant_file_sha256": "3" * 64,
            "source_assistant_row_sha256": "4" * 64,
            "source_request_sha256": "5" * 64,
            "source_response_sha256": "6" * 64,
            "source_receipt_sha256": "7" * 64,
            "source_final_file_sha256": "8" * 64,
            "source_final_result_sha256": "9" * 64,
        }
        if stage == "s2_route_optimizer":
            payload = {
                **common,
                "kind": "portfolio-current-parent-route-attribution",
                "tool_names": [],
            }
            record = PortfolioRouteAttributionRecord.model_validate(
                _self_hashed(payload, "record_sha256"),
                strict=True,
            )
        else:
            payload = {
                **common,
                "kind": "portfolio-current-parent-body-attribution",
                "response_text": "Supported response.",
                "visible_cards": [
                    {
                        "title": "Supported result",
                        "body": "Grounded evidence.",
                        "fields": [],
                    }
                ],
                "visible_tool_evidence": [
                    {
                        "tool_name": "image_product_search",
                        "status": "success",
                        "visible_text": "Grounded candidate evidence.",
                        "cards": [],
                        "citations": [],
                        "detections": [],
                        "error_code": None,
                    }
                ],
                "tool_trace": [],
                "backend_error_code": None,
                "hard_error": False,
                "final_kind": "visual_final_judge",
                "evaluation_id": "a" * 64,
                "judge_status": "scored",
                "judge_error_code": None,
                "dimensions": [
                    {
                        "dimension": "TCR",
                        "score": 5 if query_id in {"dm-001", "dm-004"} else 10,
                        "tier": (
                            "Average" if query_id in {"dm-001", "dm-004"} else "Good"
                        ),
                    }
                ],
                "j_project": 100.0,
            }
            record = PortfolioBodyAttributionRecord.model_validate(
                _self_hashed(payload, "record_sha256"),
                strict=True,
            )
        records.append(record)
    packet_payload = {
        "schema_version": 1,
        "kind": "portfolio-current-parent-attribution-packet",
        "policy_version": PORTFOLIO_PARENT_ATTRIBUTION_POLICY_VERSION,
        "stage": stage,
        "source_config": ("s1" if stage == "s2_route_optimizer" else "s1s2"),
        "source_bank_sha256": source_bank_sha256,
        "optimization_query_ids_sha256": sha256_bytes(
            canonical_json_bytes([f"dm-{index:03d}" for index in range(1, 26)])
        ),
        "source_summary_file_sha256": "b" * 64,
        "source_summary_sha256": "c" * 64,
        "source_results_file_sha256": "d" * 64,
        "query_ids": list(query_ids),
        "records": [item.model_dump(mode="json") for item in records],
    }
    packet = PortfolioParentAttributionPacket.model_validate(
        _self_hashed(packet_payload, "packet_sha256"),
        strict=True,
    )
    path.write_bytes(packet.canonical_bytes())
    return _file_sha(path)


def _tool_source(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "tool-source"
    root.mkdir()
    recipe_bytes = canonical_json_bytes(
        {
            "author": "test",
            "category": "test_dish",
            "dish": "test dish",
            "license_id": "LicenseRef-Test",
            "name": "Test dish",
            "recipeIngredient": ["ingredient"],
            "recipeInstructions": ["instruction"],
            "source_archive_sha256": "a" * 64,
            "source_record_id": "test:1",
            "source_uri": "https://example.invalid/test",
        }
    )
    rows = 1
    recipe_path = root / "recipe-evidence.jsonl"
    recipe_path.write_bytes(recipe_bytes)
    clean = ROOT / "data" / "clean"
    runtime = build_portfolio_tool_runtime(
        PortfolioRuntimeSources(
            selection_manifest=clean / "query_images" / "selection-manifest.json",
            dataset_assets=clean / "query_images" / "dataset-assets.jsonl",
            runtime_catalog_assets=(
                clean / "portfolio-mini-asset-catalog-v3" / "assets.jsonl"
            ),
            rpc_scenes=clean / "rpc-multi-product-query-v1" / "scenes.jsonl",
            inaturalist_manifest=(
                clean
                / "portfolio-source-pools"
                / "encyclopedia-inaturalist"
                / "manifest.jsonl"
            ),
            recipe_evidence=recipe_path,
        )
    )
    (root / "system-prompt.txt").write_bytes(PORTFOLIO_SYSTEM_PROMPT.encode("utf-8"))
    payload = {
        "schema_version": 1,
        "kind": "portfolio-assistant-runtime-lock",
        "policy_version": PORTFOLIO_TOOL_RUNTIME_POLICY,
        "track": "portfolio",
        "formal_eligible": False,
        "tool_registry_sha256": runtime.registry.registry_sha256,
        "tool_registry_runtime_sha256": runtime.registry.registry_runtime_sha256,
        "runtime_data_sha256": runtime.index.runtime_data_sha256,
        "source_sha256s": list(runtime.index.source_sha256s),
        "system_prompt_sha256": sha256_bytes(PORTFOLIO_SYSTEM_PROMPT.encode("utf-8")),
        "recipe_evidence_rows": rows,
    }
    lock = _self_hashed(payload, "runtime_lock_sha256")
    lock_bytes = canonical_json_bytes(lock)
    (root / "runtime-lock.json").write_bytes(lock_bytes)
    return (
        root,
        sha256_bytes(lock_bytes),
        runtime.registry.registry_runtime_sha256,
    )


def _gate_source(
    tmp_path: Path,
    *,
    config: str,
    parent_bank,
    candidate_bank,
    parent_attribution: PortfolioParentAttributionPacket | None = None,
) -> GateSource:
    source = tmp_path / f"{config}-gate-source"
    source.mkdir()
    query_ids = PORTFOLIO_OPTIMIZATION_QUERY_IDS
    parent_config = {
        "s1": "llm_static",
        "s1s2": "s1",
        "full": "s1s2",
    }[config]

    def result_set(
        *,
        role,
        source_config,
        bank,
        route,
        mean_j,
        hard,
        attribution=None,
    ):
        if attribution is not None:
            assert attribution.source_config == source_config
            assert attribution.source_bank_sha256 == bank.bank_sha256
            summary_file_sha256 = attribution.source_summary_file_sha256
            summary_sha256 = attribution.source_summary_sha256
            results_file_sha256 = attribution.source_results_file_sha256
            result_sha256s = tuple(
                item.source_result_sha256
                for item in sorted(
                    attribution.records,
                    key=lambda item: item.query_id,
                )
            )
        else:
            summary_file_sha256 = sha256_bytes(f"{config}:{role}:summary-file".encode())
            summary_sha256 = sha256_bytes(f"{config}:{role}:summary".encode())
            results_file_sha256 = sha256_bytes(f"{config}:{role}:results-file".encode())
            result_sha256s = tuple(
                sha256_bytes(f"{config}:{role}:{index}".encode())
                for index in range(len(query_ids))
            )
        return build_portfolio_stage_gate_result_set(
            source_config=source_config,
            bank_sha256=bank.bank_sha256,
            adherence_contract_bank_sha256=parent_bank.bank_sha256,
            evaluation_query_ids=query_ids,
            route_accuracy=route,
            mean_j=mean_j,
            mean_skill_adherence=0.8 if role == "parent" else 0.9,
            hard_error_count=hard,
            rubric_file_sha256=RUBRIC_SHA,
            shared_route_artifact_sha256s=(
                tuple(
                    sha256_bytes(f"shared-route:{query_id}".encode())
                    for query_id in query_ids
                )
                if source_config in {"s1s2", "full"}
                else (None,) * len(query_ids)
            ),
            source_summary_file_sha256=summary_file_sha256,
            source_summary_sha256=summary_sha256,
            source_results_file_sha256=results_file_sha256,
            source_result_sha256s=result_sha256s,
        )

    parent_results = result_set(
        role="parent",
        source_config=parent_config,
        bank=parent_bank,
        route=0.5,
        mean_j=50.0,
        hard=1,
        attribution=parent_attribution,
    ).canonical_bytes()
    candidate_results = result_set(
        role="candidate",
        source_config=config,
        bank=candidate_bank,
        route=0.6,
        mean_j=51.0,
        hard=0,
    ).canonical_bytes()
    parent_path = source / "parent-results.json"
    candidate_path = source / "candidate-results.json"
    parent_path.write_bytes(parent_results)
    candidate_path.write_bytes(candidate_results)
    report = build_portfolio_stage_gate_report(
        config=config,
        parent_bank_sha256=parent_bank.bank_sha256,
        candidate_bank_sha256=candidate_bank.bank_sha256,
        adherence_contract_bank_sha256=parent_bank.bank_sha256,
        evaluation_query_ids=query_ids,
        rubric_file_sha256=RUBRIC_SHA,
        paired_shared_route_artifact_sha256s=(
            tuple(
                sha256_bytes(f"shared-route:{query_id}".encode())
                for query_id in query_ids
            )
            if config == "full"
            else None
        ),
        parent_route_accuracy=0.5,
        candidate_route_accuracy=0.6,
        parent_mean_j=50.0,
        candidate_mean_j=51.0,
        parent_mean_skill_adherence=0.8,
        candidate_mean_skill_adherence=0.9,
        parent_hard_error_count=1,
        candidate_hard_error_count=0,
        decision="accepted",
        parent_result_file=f"gates/{config}/parent-results.json",
        parent_result_file_sha256=sha256_bytes(parent_results),
        candidate_result_file=f"gates/{config}/candidate-results.json",
        candidate_result_file_sha256=sha256_bytes(candidate_results),
    )
    report_path = source / "gate-report.json"
    report_path.write_bytes(report.canonical_bytes())
    return GateSource(
        key={"s1": "s1", "s1s2": "s2", "full": "s3"}[config],
        report_path=report_path,
        expected_report_file_sha256=_file_sha(report_path),
        parent_results_path=parent_path,
        candidate_results_path=candidate_path,
    )


@pytest.mark.parametrize("schema_version", (6, 7, 8, 9, 10))
def test_finalizer_dispatches_forward_sparse_bundle_only_to_sparse_s1(
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
) -> None:
    content = canonical_json_bytes(
        {
            "schema_version": schema_version,
            "kind": "portfolio-s1-feedback-bundle",
            "policy_version": f"portfolio-s1-feedback-bundle-v{schema_version}",
        }
    )

    class _LoadedSparseBundle:
        selected_count = 240
        parsed_count = 240
        provider_call_count = 243
        retry_claim_count = 3
        retry_claim_sha256s = ("1" * 64, "2" * 64, "3" * 64)
        fresh_output_count = 240
        historical_feedback_outputs_imported = 0
        run_sha256 = "4" * 64
        round3_run_sha256 = run_sha256
        run_file_sha256 = "5" * 64
        round3_run_file_sha256 = run_file_sha256
        authorization_sha256 = "6" * 64
        round3_authorization_sha256 = authorization_sha256
        control_sha256 = "7" * 64
        round3_control_sha256 = control_sha256
        round3_artifact_set_sha256 = "8" * 64
        entry_provenance = tuple(
            type(
                "_Provenance",
                (),
                {
                    "bound_artifact_sha256": f"{index:064x}",
                    "feedback_result_sha256": f"{index + 240:064x}",
                    "final_global_call_ordinal": index,
                },
            )()
            for index in range(1, 241)
        )

        @classmethod
        def model_validate_json(
            cls,
            candidate: bytes,
            *,
            strict: bool,
        ) -> "_LoadedSparseBundle":
            assert strict
            assert candidate == content
            return cls()

        def canonical_bytes(self) -> bytes:
            return content

        def model_projection_payload(self) -> dict[str, object]:
            return {
                "schema_version": 5,
                "selected_count": 240,
                "parsed_count": 240,
                "status": "complete_policy_filtered_feedback",
            }

    monkeypatch.setattr(
        finalizer_module,
        f"PortfolioS1FeedbackBundleV{schema_version}",
        _LoadedSparseBundle,
    )
    assert isinstance(
        finalizer_module._load_typed_feedback_bundle(content, sparse=True),
        _LoadedSparseBundle,
    )
    with pytest.raises(
        PortfolioTreatmentFinalizationError,
        match="implementation and typed Feedback bundle version differ",
    ):
        finalizer_module._load_typed_feedback_bundle(content, sparse=False)


def test_finalizer_rejects_bundle_v10_retry_provenance_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = canonical_json_bytes(
        {
            "schema_version": 10,
            "kind": "portfolio-s1-feedback-bundle",
            "policy_version": "portfolio-s1-feedback-bundle-v10",
        }
    )

    class _DriftedV10:
        selected_count = 240
        provider_call_count = 242
        retry_claim_count = 3
        retry_claim_sha256s = ("1" * 64, "2" * 64, "3" * 64)
        fresh_output_count = 240
        historical_feedback_outputs_imported = 0
        run_sha256 = round3_run_sha256 = "4" * 64
        run_file_sha256 = round3_run_file_sha256 = "5" * 64
        authorization_sha256 = round3_authorization_sha256 = "6" * 64
        control_sha256 = round3_control_sha256 = "7" * 64
        entry_provenance = tuple(range(240))

        @classmethod
        def model_validate_json(
            cls, candidate: bytes, *, strict: bool
        ) -> "_DriftedV10":
            assert strict and candidate == content
            return cls()

        def canonical_bytes(self) -> bytes:
            return content

    monkeypatch.setattr(
        finalizer_module,
        "PortfolioS1FeedbackBundleV10",
        _DriftedV10,
    )
    with pytest.raises(
        PortfolioTreatmentFinalizationError,
        match="fresh Round3 provenance drifted",
    ):
        finalizer_module._load_typed_feedback_bundle(content, sparse=True)


def test_bank_only_scaffold_cannot_be_named_into_a_real_s1_stage(
    tmp_path: Path,
) -> None:
    scaffold = tmp_path / "s1"
    scaffold.mkdir()
    (scaffold / "candidate-bank.json").write_bytes(b"{}\n")

    with pytest.raises(
        PortfolioTreatmentFinalizationError,
        match="detailed invocation receipt cannot be read safely",
    ):
        _load_evolution_source(
            StageSource(
                key="s1",
                root=scaffold,
                expected_invocation_receipt_file_sha256="0" * 64,
            ),
            parent_bank=None,
            prior_stage=None,
            prior_gate=None,
            semantic_bytes=b"{}\n",
            codex_bytes=b"{}\n",
            common_authoring_input_sha256="1" * 64,
            tool_registry_runtime_sha256="2" * 64,
        )


def test_legacy_real_s1_remains_compatible_but_old_s2_is_rejected() -> None:
    legacy = SimpleNamespace(
        implementation_id="portfolio-evolution-codex-cli",
        implementation_version="1.0.0",
        policy_version="portfolio-evolution-single-clean-turn-v1",
    )
    _require_current_mutation_implementation("s1", legacy)
    sparse = SimpleNamespace(
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=S1_SPARSE_IMPLEMENTATION_VERSION,
        policy_version=S1_SPARSE_INVOCATION_POLICY_VERSION,
        session_mode="ephemeral",
    )
    _require_current_mutation_implementation("s1", sparse)
    with pytest.raises(
        PortfolioTreatmentFinalizationError,
        match="s1 does not use its required stage-specific implementation",
    ):
        _require_current_mutation_implementation(
            "s1",
            SimpleNamespace(
                implementation_id=IMPLEMENTATION_ID,
                implementation_version=S1_SPARSE_IMPLEMENTATION_VERSION,
                policy_version=S1_SPARSE_INVOCATION_POLICY_VERSION,
                session_mode="resume",
            ),
        )
    with pytest.raises(
        PortfolioTreatmentFinalizationError,
        match="s2 does not use its required stage-specific implementation",
    ):
        _require_current_mutation_implementation("s2", legacy)


def test_s2_and_s3_require_distinct_archived_and_clean_turn_implementations() -> None:
    current_s2 = SimpleNamespace(
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        policy_version=INVOCATION_POLICY_VERSION,
        session_mode="new_persistent",
    )
    current_s3 = SimpleNamespace(
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=S3_IMPLEMENTATION_VERSION,
        policy_version=S3_INVOCATION_POLICY_VERSION,
        session_mode="ephemeral",
    )
    _require_current_mutation_implementation("s2", current_s2)
    _require_current_mutation_implementation("s3", current_s3)

    for invalid_s3 in (
        SimpleNamespace(
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            policy_version=INVOCATION_POLICY_VERSION,
            session_mode="ephemeral",
        ),
        SimpleNamespace(
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=S3_IMPLEMENTATION_VERSION,
            policy_version=S3_INVOCATION_POLICY_VERSION,
            session_mode="resume",
        ),
    ):
        with pytest.raises(
            PortfolioTreatmentFinalizationError,
            match="s3 does not use its required stage-specific implementation",
        ):
            _require_current_mutation_implementation("s3", invalid_s3)

    legacy_s3 = SimpleNamespace(
        implementation_id=IMPLEMENTATION_ID,
        implementation_version="1.4.0",
        policy_version="portfolio-evolution-gate-bound-ephemeral-turn-v4",
        session_mode="ephemeral",
    )
    with pytest.raises(
        PortfolioTreatmentFinalizationError,
        match="s3 does not use its required stage-specific implementation",
    ):
        _require_current_mutation_implementation("s3", legacy_s3)


def test_archived_s2_v13_evidence_remains_finalizer_compatible() -> None:
    stage_root = (
        ROOT
        / "runs"
        / "portfolio"
        / "portfolio-real-treatment-development-v3"
        / "stages"
        / "s2"
    )
    parent_path = (
        ROOT
        / "runs"
        / "portfolio"
        / "portfolio-real-treatment-development-v1"
        / "stages"
        / "s1"
        / "candidate-bank.json"
    )
    implementation_path = (
        ROOT
        / "runs"
        / "portfolio"
        / "portfolio-public-data-runtime-v15"
        / "stages"
        / "s1s2"
        / "implementation-source.py"
    )
    parent = StaticBankArtifact.model_validate_json(
        parent_path.read_bytes(),
        strict=True,
    )
    receipt_path = stage_root / "invocation-receipt.json"
    loaded = _load_evolution_source(
        StageSource(
            key="s2",
            root=stage_root,
            expected_invocation_receipt_file_sha256=_file_sha(receipt_path),
            implementation_source_path=implementation_path,
            expected_implementation_source_file_sha256=_file_sha(implementation_path),
        ),
        parent_bank=parent,
        prior_stage=None,
        prior_gate=None,
        semantic_bytes=b"{}\n",
        codex_bytes=b"{}\n",
        common_authoring_input_sha256="0" * 64,
        tool_registry_runtime_sha256=parent.tool_registry_runtime_sha256,
    )
    assert loaded.detailed.implementation_version == IMPLEMENTATION_VERSION
    assert loaded.detailed.policy_version == INVOCATION_POLICY_VERSION
    assert loaded.detailed.session_mode == "new_persistent"


def test_gate_loader_rejects_untyped_canonical_result_objects(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent-result.json"
    candidate = tmp_path / "candidate-result.json"
    parent.write_bytes(canonical_json_bytes({"rows": 6, "role": "parent"}))
    candidate.write_bytes(canonical_json_bytes({"rows": 6, "role": "candidate"}))
    report = build_portfolio_stage_gate_report(
        config="s1",
        parent_bank_sha256="1" * 64,
        candidate_bank_sha256="2" * 64,
        adherence_contract_bank_sha256="1" * 64,
        evaluation_query_ids=PORTFOLIO_OPTIMIZATION_QUERY_IDS,
        rubric_file_sha256=RUBRIC_SHA,
        paired_shared_route_artifact_sha256s=None,
        parent_route_accuracy=0.5,
        candidate_route_accuracy=0.6,
        parent_mean_j=50.0,
        candidate_mean_j=51.0,
        parent_mean_skill_adherence=0.8,
        candidate_mean_skill_adherence=0.9,
        parent_hard_error_count=1,
        candidate_hard_error_count=0,
        decision="accepted",
        parent_result_file="gates/s1/parent-result.json",
        parent_result_file_sha256=_file_sha(parent),
        candidate_result_file="gates/s1/candidate-result.json",
        candidate_result_file_sha256=_file_sha(candidate),
    )
    report_path = tmp_path / "gate-report.json"
    report_path.write_bytes(report.canonical_bytes())

    with pytest.raises(
        PortfolioTreatmentFinalizationError,
        match="parent gate results is invalid",
    ):
        _load_gate(
            GateSource(
                key="s1",
                report_path=report_path,
                expected_report_file_sha256=_file_sha(report_path),
                parent_results_path=parent,
                candidate_results_path=candidate,
            )
        )


@pytest.mark.parametrize("bind_rejected_buffer", (False, True))
def test_finalizer_publishes_a_loader_verified_real_treatment_runtime(
    tmp_path: Path,
    bind_rejected_buffer: bool,
) -> None:
    tool_root, tool_lock_sha, runtime_sha = _tool_source(tmp_path)
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"test Codex executable")
    s1_packet_path = tmp_path / "s1-packet.json"
    s1_packet_sha = _write_packet(s1_packet_path, _packet(stage="s1"))
    s1 = run_portfolio_evolution_model(
        stage="s1_creator",
        output_dir=tmp_path / "s1-stage",
        stage_input_path=s1_packet_path,
        expected_stage_input_file_sha256=s1_packet_sha,
        semantic_authoring_input_path=SEMANTIC_INPUT,
        expected_semantic_authoring_input_file_sha256=_file_sha(SEMANTIC_INPUT),
        codex_authoring_input_path=CODEX_INPUT,
        expected_codex_authoring_input_file_sha256=_file_sha(CODEX_INPUT),
        tool_registry_runtime_sha256=runtime_sha,
        codex_executable=executable,
        process_runner=_MockCodex(AUTHOR_RAW.read_bytes()),
    )
    s2_packet_path = tmp_path / "s2-packet.json"
    s2_packet_sha = _write_parent_packet(
        s2_packet_path,
        stage="s2_route_optimizer",
        source_bank_sha256=s1.candidate_bank.bank_sha256,
    )
    s2_packet = PortfolioParentAttributionPacket.model_validate_json(
        s2_packet_path.read_bytes(),
        strict=True,
    )
    s2_skill = next(
        item
        for item in s1.candidate_bank.skills
        if item.capability_id == "product.exact_match"
    )
    s2_raw = canonical_json_bytes(
        {
            "schema_version": 1,
            "changes": [
                {
                    "capability_id": s2_skill.capability_id,
                    "description": (
                        "Route only when the exact requested outcome and its "
                        "evidence contract match this capability."
                    ),
                }
            ],
        }
    )
    session_scratch = tmp_path / "s2-s3-session-scratch"
    session_scratch.mkdir()
    s2 = run_portfolio_evolution_model(
        stage="s2_route_optimizer",
        output_dir=tmp_path / "s2-stage",
        stage_input_path=s2_packet_path,
        expected_stage_input_file_sha256=s2_packet_sha,
        parent_bank_path=s1.candidate_bank_path,
        expected_parent_bank_file_sha256=_file_sha(s1.candidate_bank_path),
        codex_executable=executable,
        session_mode="new_persistent",
        session_scratch=session_scratch,
        process_runner=_MockCodex(
            s2_raw,
            thread_id="finalizer-s2-persistent-thread",
        ),
    )
    s1_gate = _gate_source(
        tmp_path,
        config="s1",
        parent_bank=s1.candidate_bank,
        candidate_bank=s1.candidate_bank,
    )
    s2_gate = _gate_source(
        tmp_path,
        config="s1s2",
        parent_bank=s1.candidate_bank,
        candidate_bank=s2.candidate_bank,
        parent_attribution=s2_packet,
    )
    s3_packet_path = tmp_path / "s3-packet.json"
    s3_packet_sha = _write_parent_packet(
        s3_packet_path,
        stage="s3_body_refiner",
        source_bank_sha256=s2.candidate_bank.bank_sha256,
    )
    s3_packet = PortfolioParentAttributionPacket.model_validate_json(
        s3_packet_path.read_bytes(),
        strict=True,
    )
    s3_skill = next(
        item
        for item in s2.candidate_bank.skills
        if item.capability_id == s2_skill.capability_id
    )
    _, s3_scope = _actionable_mutation_scope(
        "s3_body_refiner",
        s3_packet,
        parent_bank=s2.candidate_bank,
    )
    cluster = s3_scope["failure_clusters"][0]
    editable_headings = {
        "## Success criteria",
        "## Acceptable answer rules",
        "## Failure conditions",
        "## Fallback triggers",
        "## Fallback responses",
        "## Indeterminate conditions",
    }
    current_heading = ""
    target_rule_line = ""
    for line in s3_skill.body.splitlines():
        if line.startswith("## "):
            current_heading = line
        elif current_heading in editable_headings and line.startswith("- ["):
            target_rule_line = line
            break
    assert target_rule_line
    target_rule_id = target_rule_line[3 : target_rule_line.index("]")]
    s3_raw = canonical_json_bytes(
        {
            "schema_version": 1,
            "artifact_kind": "portfolio-s3-text-patch-proposal",
            "policy_version": "portfolio-s3-textopt-proposal-v1",
            "capability_id": cluster["capability_id"],
            "failure_cluster_id": cluster["failure_cluster_id"],
            "failure_cluster_sha256": cluster["cluster_sha256"],
            "edits": [
                {
                    "op": "replace_rule",
                    "rule_id": target_rule_id,
                    "expected_text_sha256": sha256_bytes(
                        target_rule_line.encode("utf-8")
                    ),
                    "replacement": (
                        "Lead with the supported result before presenting limitations."
                    ),
                    "evidence_ids": cluster["evidence_ids"],
                    "support_count": cluster["support_count"],
                    "addressed_dimensions": cluster["addressed_dimensions"],
                    "rationale": "Address the recurring low TCR evidence cluster.",
                }
            ],
            "reconsideration_rationale": None,
        }
    )
    rejected_buffer_arguments = {}
    if bind_rejected_buffer:
        rejected_buffer_path = tmp_path / "s3-rejected-edit-buffer.json"
        rejected_buffer_path.write_bytes(
            build_empty_portfolio_s3_rejected_edit_buffer().canonical_bytes()
        )
        rejected_buffer_arguments = {
            "rejected_edit_buffer_path": rejected_buffer_path,
            "expected_rejected_edit_buffer_file_sha256": _file_sha(
                rejected_buffer_path
            ),
        }
    s3 = run_portfolio_evolution_model(
        stage="s3_body_refiner",
        output_dir=tmp_path / "s3-stage",
        stage_input_path=s3_packet_path,
        expected_stage_input_file_sha256=s3_packet_sha,
        parent_bank_path=s2.candidate_bank_path,
        expected_parent_bank_file_sha256=_file_sha(s2.candidate_bank_path),
        codex_executable=executable,
        prior_stage_gate=s2_gate.report_path,
        expected_prior_stage_gate_file_sha256=(s2_gate.expected_report_file_sha256),
        process_runner=_MockCodex(
            s3_raw,
            thread_id="finalizer-s3-clean-thread",
        ),
        **rejected_buffer_arguments,
    )
    gates = (
        s1_gate,
        s2_gate,
        _gate_source(
            tmp_path,
            config="full",
            parent_bank=s2.candidate_bank,
            candidate_bank=s3.candidate_bank,
            parent_attribution=s3_packet,
        ),
    )
    inputs = FinalizationInputs(
        output_dir=tmp_path / "final-runtime",
        tool_runtime_source_root=tool_root,
        expected_tool_runtime_source_lock_file_sha256=tool_lock_sha,
        semantic_authoring_input_path=SEMANTIC_INPUT,
        expected_semantic_authoring_input_file_sha256=_file_sha(SEMANTIC_INPUT),
        codex_authoring_input_path=CODEX_INPUT,
        expected_codex_authoring_input_file_sha256=_file_sha(CODEX_INPUT),
        static_authoring_root=STATIC_ROOT,
        expected_static_invocation_receipt_file_sha256=_file_sha(
            STATIC_ROOT / "invocation-receipt.json"
        ),
        static_human_review_path=STATIC_REVIEW,
        expected_static_human_review_file_sha256=_file_sha(STATIC_REVIEW),
        stages=(
            StageSource(
                key="s1",
                root=s1.output_dir,
                expected_invocation_receipt_file_sha256=_file_sha(s1.receipt_path),
            ),
            StageSource(
                key="s2",
                root=s2.output_dir,
                expected_invocation_receipt_file_sha256=_file_sha(s2.receipt_path),
            ),
            StageSource(
                key="s3",
                root=s3.output_dir,
                expected_invocation_receipt_file_sha256=_file_sha(s3.receipt_path),
            ),
        ),
        gates=gates,
    )
    fake_queries = tuple(
        SimpleNamespace(
            query_id=f"dm-{index:03d}",
            leakage_group_id=(
                f"optimization-group-{index}"
                if index <= 25
                else f"evaluation-group-{index}"
            ),
            synthesis_batch_id=(
                OPTIMIZATION_BATCH_ID if index <= 25 else "evaluation-batch"
            ),
        )
        for index in range(1, 201)
    )
    result = finalize_portfolio_treatment_runtime(
        inputs,
        portfolio_inputs_loader=lambda: SimpleNamespace(queries=fake_queries),
    )

    verified = load_verified_portfolio_treatment_runtime(
        result.root,
        expected_runtime_lock_file_sha256=result.runtime_lock_file_sha256,
    )
    assert verified.runtime_lock["final_judge_result_schema_version"] == (
        FINAL_JUDGE_RESULT_SCHEMA_VERSION
    )
    assert verified.runtime_lock["final_judge_cache_namespace"] == (
        FINAL_JUDGE_CACHE_NAMESPACE
    )
    assert verified.runtime_lock["final_judge_max_attempts"] == 2
    assert verified.runtime_lock["final_judge_retry_policy_version"] == (
        FINAL_JUDGE_RETRY_POLICY_VERSION
    )
    assert verified.runtime_lock["final_judge_thinking_budget"] == (
        FINAL_JUDGE_THINKING_BUDGET
    )
    assert verified.runtime_lock["final_judge_transport_policy_version"] == (
        FINAL_JUDGE_TRANSPORT_POLICY_VERSION
    )
    assert verified.runtime_lock["final_judge_transport_policy_sha256"] == (
        FINAL_JUDGE_TRANSPORT_POLICY_SHA256
    )
    assert (
        verified.runtime_lock["final_judge_requested_response_format"] == "json_object"
    )
    assert verified.runtime_lock["final_judge_provider_input_token_reserve"] == (
        FINAL_JUDGE_PROVIDER_INPUT_TOKEN_RESERVE
    )
    assert verified.runtime_lock["final_judge_provider_output_token_reserve"] == (
        FINAL_JUDGE_PROVIDER_OUTPUT_TOKEN_RESERVE
    )
    assert verified.runtime_lock["final_judge_provider_pricing_status"] == (
        PORTFOLIO_GEMINI_JUDGE_PRICING_STATUS
    )
    assert verified.runtime_lock["portfolio_budget_policy_version"] == (
        PORTFOLIO_BUDGET_POLICY_VERSION
    )
    assert verified.runtime_lock["portfolio_budget_policy_sha256"] == (
        PORTFOLIO_BUDGET_POLICY_SHA256
    )
    assert verified.runtime_lock["provider_pricing_contract_sha256"] == (
        PORTFOLIO_PROVIDER_PRICING_CONTRACT_SHA256
    )
    assert verified.runtime_lock["card_requirement_guard_policy_version"] == (
        CARD_REQUIREMENT_GUARD_POLICY_VERSION
    )
    assert verified.runtime_lock["card_requirement_guard_policy_sha256"] == (
        CARD_REQUIREMENT_GUARD_POLICY_SHA256
    )
    assert verified.runtime_lock["execution_artifact_aliases"] == []
    assert (
        verified.runtime_lock["execution_artifact_alias_provider_model_call_count"] == 0
    )
    assert verified.runtime_lock["config_file_sha256"] == sha256_bytes(
        (ROOT / "src" / "skillchain" / "config.py").read_bytes()
    )
    assert verified.runtime_lock["packets_file_sha256"] == sha256_bytes(
        (ROOT / "src" / "skillchain" / "evaluation" / "packets.py").read_bytes()
    )
    assert verified.runtime_lock["evaluator_isolation_file_sha256"] == sha256_bytes(
        (
            ROOT / "src" / "skillchain" / "evaluation" / "evaluator_isolation.py"
        ).read_bytes()
    )
    assert verified.chain.manifest.status == "ready_for_matrix"
    assert len(verified.chain.manifest.optimization_query_ids) == 25
    assert len(verified.chain.manifest.evaluation_query_ids) == 175
    assert tuple(item.decision for item in verified.chain.manifest.records) == (
        "accepted",
        "accepted",
        "accepted",
        "accepted",
    )
    assert verified.chain.output_banks["s1s2"] == s2.candidate_bank
    assert verified.chain.output_banks["full"] == s3.candidate_bank
    assert (result.root / "treatments/s1-receipt.json").is_file()
    assert (result.root / "inputs/s1/packet.json").is_file()
    assert (result.root / "stages/full/codex-events.jsonl").is_file()
    assert (result.root / "stages/full/s3-text-patch.json").is_file()
    assert (result.root / "stages/full/textopt-compiler-source.py").is_file()
    assert (
        result.root / "inputs/full/rejected-edit-buffer.json"
    ).is_file() is bind_rejected_buffer

    published_patch = result.root / "stages/full/s3-text-patch.json"
    published_patch_bytes = published_patch.read_bytes()
    published_patch.write_bytes(b"{}\n")
    with pytest.raises(
        PortfolioTreatmentError,
        match="normalized TextOpt patch file digest mismatch",
    ):
        load_verified_portfolio_treatment_runtime(
            result.root,
            expected_runtime_lock_file_sha256=result.runtime_lock_file_sha256,
        )
    published_patch.write_bytes(published_patch_bytes)

    published_compiler = result.root / "stages/full/textopt-compiler-source.py"
    published_compiler_bytes = published_compiler.read_bytes()
    published_compiler.write_bytes(b"tampered TextOpt compiler snapshot\n")
    with pytest.raises(
        PortfolioTreatmentError,
        match="TextOpt compiler snapshot file digest mismatch",
    ):
        load_verified_portfolio_treatment_runtime(
            result.root,
            expected_runtime_lock_file_sha256=result.runtime_lock_file_sha256,
        )
    published_compiler.write_bytes(published_compiler_bytes)

    (result.root / "stages/full/implementation-source.py").write_bytes(
        b"tampered implementation snapshot\n"
    )
    with pytest.raises(
        PortfolioTreatmentError,
        match="implementation source snapshot file digest mismatch",
    ):
        load_verified_portfolio_treatment_runtime(
            result.root,
            expected_runtime_lock_file_sha256=result.runtime_lock_file_sha256,
        )
