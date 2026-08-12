from __future__ import annotations

from argparse import Namespace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import export_core_fast_opt_static as exporter
from skillchain.evaluation.core_fast.models import AssistantObservation
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_jsonl,
    sha256_bytes,
)


class _Dumpable(SimpleNamespace):
    def __init__(self, payload: dict[str, object], **attributes: object) -> None:
        super().__init__(**attributes)
        self._payload = payload

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self._payload


def _row(index: int, *, output_contract_pass: bool = True) -> SimpleNamespace:
    query_id = f"opt-{index:04d}"
    trace_payload = {
        "call_index": 1,
        "tool_name": "image_product_search",
        "status": "success",
        "arguments_sha256": f"{index + 1:064x}",
        "result_sha256": f"{index + 2:064x}",
        "runtime_binding_sha256": "3" * 64,
        "latency_ms": 7,
        "error_code": None,
    }
    trace = _Dumpable(
        trace_payload,
        tool_name=trace_payload["tool_name"],
        status=trace_payload["status"],
        arguments_sha256=trace_payload["arguments_sha256"],
        result_sha256=trace_payload["result_sha256"],
        error_code=trace_payload["error_code"],
    )
    response_payload = {
        "response_text": f"verified response {index}",
        "selected_capability": "product.exact_match",
        "route_trace_sha256": "4" * 64,
        "tool_trace": [trace_payload],
    }
    response = _Dumpable(
        response_payload,
        response_text=response_payload["response_text"],
        selected_capability=response_payload["selected_capability"],
        route_trace_sha256=response_payload["route_trace_sha256"],
        tool_trace=(trace,),
    )
    scorer_payload = {
        "call_index": 1,
        "tool_name": "image_product_search",
        "arguments_sha256": trace_payload["arguments_sha256"],
    }
    scorer_call = _Dumpable(scorer_payload)
    passed = int(output_contract_pass)
    return SimpleNamespace(
        query_ordinal=index,
        query=SimpleNamespace(
            query_id=query_id,
            requires_card=True,
            split="opt_pool",
        ),
        response=response,
        receipt=_Dumpable({"receipt_sha256": "5" * 64}),
        sidecar=SimpleNamespace(
            calls=(scorer_call,),
            policy_version="portfolio-gcs-scorer-evidence-v2",
        ),
        result=_Dumpable({"query_id": query_id, "response_text": f"result {index}"}),
        score=SimpleNamespace(
            answer_mode="supported" if passed else "unresolved",
            oracle_available=True,
            route_acceptable=1,
            no_hard_error=1,
            tool_contract_pass=1,
            evidence_grounded=1,
            output_contract_pass=passed,
            gcs=passed,
            hard_error=0,
            reason_codes=() if passed else ("output_section_invalid",),
        ),
        strata=SimpleNamespace(
            source_dataset="abo",
            repair_status="r3_carry_forward",
            style_submode=None,
        ),
    )


def _corpus(rows: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        rows=rows,
        core_inputs=SimpleNamespace(queries=tuple(row.query for row in rows)),
        corpus_sha256="a" * 64,
        control_file_sha256="b" * 64,
    )


def test_observation_maps_verified_evidence_and_derived_flags() -> None:
    row = _row(0, output_contract_pass=False)
    observation = exporter.observation_from_verified_row(row)

    assert observation.query_id == row.query.query_id
    assert observation.response_text == row.response.response_text
    assert observation.selected_capability == row.response.selected_capability
    assert observation.route_trace_key == row.response.route_trace_sha256
    assert observation.tool_trace_key == sha256_bytes(
        canonical_json_bytes([row.response.tool_trace[0].model_dump(mode="json")])
    )
    assert observation.tool_trace[0].arguments == {
        "arguments_sha256": row.response.tool_trace[0].arguments_sha256
    }
    assert observation.replay_context == {
        "response": row.response.model_dump(mode="json"),
        "receipt": row.receipt.model_dump(mode="json"),
        "scorer_calls": [row.sidecar.calls[0].model_dump(mode="json")],
        "scorer_capture_policy_version": row.sidecar.policy_version,
        "assistant_result": row.result.model_dump(mode="json"),
    }
    assert observation.gcs_components["output_contract_pass"] is False
    assert observation.answer_mode == "unresolved"
    assert observation.gcs_reason_codes == ("output_section_invalid",)
    assert observation.gcs_score == 0.0
    assert observation.card_violation
    assert not observation.evidence_violation
    assert not observation.tool_violation
    assert observation.source == row.strata.source_dataset
    assert observation.repair == row.strata.repair_status
    assert observation.style_submode is None


def test_export_is_deterministic_create_only_exact_opt800(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tuple(_row(index) for index in reversed(range(800)))
    corpus = _corpus(rows)
    calls: list[tuple[object, dict[str, object]]] = []

    def _load(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return corpus

    monkeypatch.setattr(exporter, "load_verified_static_gcs_corpus", _load)
    first = tmp_path / "first.jsonl"
    arguments = Namespace(
        execution_root=tmp_path / "execution",
        artifact_repository_root=tmp_path / "artifact-root",
        expected_execution_control_file_sha256="c" * 64,
        output=first,
    )
    summary = exporter.export(arguments)
    parsed = parse_canonical_jsonl(first.read_bytes(), label="test export")
    observations = tuple(
        AssistantObservation.model_validate(item, strict=True) for item in parsed
    )

    assert len(observations) == 800
    assert tuple(item.query_id for item in observations) == tuple(
        f"opt-{index:04d}" for index in range(800)
    )
    assert summary["row_count"] == 800
    assert summary["provider_calls_performed"] == 0
    assert summary["output_sha256"] == sha256_bytes(first.read_bytes())
    assert calls[0][1] == {
        "expected_control_file_sha256": "c" * 64,
        "artifact_repository_root": arguments.artifact_repository_root,
    }

    second = tmp_path / "second.jsonl"
    second_arguments = Namespace(**{**vars(arguments), "output": second})
    exporter.export(second_arguments)
    assert second.read_bytes() == first.read_bytes()
    with pytest.raises(FileExistsError, match="目标已存在"):
        exporter.export(arguments)


def test_build_observations_rejects_incomplete_opt_population() -> None:
    rows = tuple(_row(index) for index in range(799))
    with pytest.raises(
        exporter.CoreFastOptStaticExportError,
        match="ordinals 0..799",
    ):
        exporter.build_observations(_corpus(rows))


def test_cli_direct_help_bootstraps_repository_imports(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(exporter.__file__)), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "verified Static opt800 corpus" in completed.stdout
