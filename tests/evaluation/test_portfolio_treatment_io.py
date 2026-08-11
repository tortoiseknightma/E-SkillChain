from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_portfolio_evolution_model import EvolutionInvocationReceipt
from skillchain.codex_authoring import CODEX_COMMAND_SHAPE
from skillchain.evaluation.portfolio_treatment_io import (
    _verify_evolution_session_lineage,
    load_verified_portfolio_treatment_runtime,
)
from skillchain.evaluation.portfolio_treatments import PortfolioTreatmentError
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _write_runtime_lock(root: Path, payload: dict) -> str:
    lock = {
        **payload,
        "runtime_lock_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }
    content = canonical_json_bytes(lock)
    (root / "runtime-lock.json").write_bytes(content)
    return sha256_bytes(content)


def test_disk_loader_rejects_historical_scaffold_before_loading_banks(
    tmp_path: Path,
) -> None:
    digest = _write_runtime_lock(
        tmp_path,
        {
            "schema_version": 2,
            "kind": "portfolio-assistant-runtime-lock",
            "track": "portfolio",
            "formal_eligible": False,
            "bank_policy": "portfolio-post-smoke-candidate-v1",
            "official_matrix_eligible": False,
            "treatment_chain_status": "not_created",
        },
    )

    with pytest.raises(
        PortfolioTreatmentError,
        match="not a matrix-ready real treatment runtime",
    ):
        load_verified_portfolio_treatment_runtime(
            tmp_path,
            expected_runtime_lock_file_sha256=digest,
        )


def test_disk_loader_requires_external_runtime_digest(tmp_path: Path) -> None:
    (tmp_path / "runtime-lock.json").write_bytes(b"{}\n")

    with pytest.raises(PortfolioTreatmentError, match="expected SHA-256"):
        load_verified_portfolio_treatment_runtime(
            tmp_path,
            expected_runtime_lock_file_sha256="not-a-digest",
        )


def _write_detailed_receipt(
    root: Path,
    *,
    config: str,
    stage: str,
    implementation_version: str,
    policy_version: str,
    thread_id: str,
    input_roles: tuple[str, ...] = (),
    prior_gate_sha256: str | None = None,
) -> SimpleNamespace:
    stage_root = root / "stages" / config
    stage_root.mkdir(parents=True)
    implementation_bytes = f"# {config} implementation\n".encode()
    (stage_root / "implementation-source.py").write_bytes(implementation_bytes)
    implementation_sha256 = sha256_bytes(implementation_bytes)
    candidate_sha256 = sha256_bytes(f"candidate:{config}".encode())
    raw_sha256 = sha256_bytes(f"raw:{config}".encode())
    is_s2 = config == "s1s2"
    command = tuple(
        item for item in CODEX_COMMAND_SHAPE if not is_s2 or item != "--ephemeral"
    )
    effective_input_roles = input_roles or ("stage_input",)
    input_files = []
    for role in effective_input_roles:
        digest = prior_gate_sha256 if role == "prior_stage_gate" else "a" * 64
        input_files.append(
            {
                "role": role,
                "path": f"inputs/{role}.json",
                "file_sha256": digest,
                "content_sha256": digest,
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-evolution-invocation-receipt",
        "status": "completed",
        "stage": stage,
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "sandbox": "read-only",
        "thread_id": thread_id,
        "candidate_bank_sha256": candidate_sha256,
        "raw_model_output_sha256": raw_sha256,
        "implementation_id": "portfolio-evolution-codex-cli",
        "implementation_version": implementation_version,
        "policy_version": policy_version,
        "implementation_file_sha256": implementation_sha256,
        "codex_executable": "codex.exe",
        "normalized_command": list(command),
        "command_sha256": sha256_bytes(canonical_json_bytes(list(command))),
        "session_mode": "new_persistent" if is_s2 else "ephemeral",
        "ephemeral": not is_s2,
        "new_session_count": 1,
        "resume_count": 0,
        "session_turn_index": 1,
        "followup_count": 0,
        "resume_thread_id": None,
        "session_scratch_path": "s2-scratch" if is_s2 else None,
        "session_scratch_device": 1 if is_s2 else None,
        "session_scratch_inode": 2 if is_s2 else None,
        "codex_executable_sha256": "b" * 64,
        "input_files": input_files,
        "prior_invocation_receipt_file_sha256": None,
        "prior_gate_report_file_sha256": prior_gate_sha256,
        "user_config_ignored": True,
        "rules_ignored": True,
        "output_schema_requested": True,
        "invocation_count": 1,
        "retry_count": 0,
        "fallback_count": 0,
        "repair_count": 0,
        "prompt_sha256": "c" * 64,
        "output_schema_sha256": "d" * 64,
        "process_returncode": 0,
        "timed_out": False,
        "elapsed_ms": 1,
        "clean_turn": True,
        "visible_tool_activity": False,
        "input_tokens": 1,
        "cached_input_tokens": 0,
        "output_tokens": 1,
        "mutation_sha256": None if config == "s1" else "e" * 64,
        "output_files": [
            {"file": name, "file_sha256": "f" * 64}
            for name in (
                "candidate-bank.json",
                "codex-events.jsonl",
                "codex-stderr.bin",
                "prompt.txt",
                "raw-model-output.json",
            )
        ],
        "error_type": None,
        "error_message": None,
    }
    receipt = EvolutionInvocationReceipt.model_validate(
        {
            **payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )
    content = receipt.canonical_bytes()
    relative = f"stages/{config}/invocation-receipt.json"
    (root / relative).write_bytes(content)
    binding = SimpleNamespace(
        artifact_kind="detailed_invocation_receipt",
        artifact_file=relative,
        artifact_file_sha256=sha256_bytes(content),
        artifact_content_sha256=sha256_bytes(content),
    )
    record = SimpleNamespace(
        config=config,
        stage=stage,
        candidate_bank_sha256=candidate_sha256,
        raw_model_output_file_sha256=raw_sha256,
        implementation_id="portfolio-evolution-codex-cli",
        implementation_version=implementation_version,
        implementation_file_sha256=implementation_sha256,
        input_artifacts=(binding,),
        gate_report_file=f"gates/{config}/gate-report.json",
        gate_report_file_sha256=prior_gate_sha256 or "c" * 64,
    )
    return SimpleNamespace(record=record, thread_id=thread_id)


def test_runtime_lineage_keeps_self_hashed_legacy_s3_receipt_readable(
    tmp_path: Path,
) -> None:
    s1 = _write_detailed_receipt(
        tmp_path,
        config="s1",
        stage="s1_creator",
        implementation_version="1.0.0",
        policy_version="portfolio-evolution-single-clean-turn-v1",
        thread_id="s1-thread",
    )
    s2 = _write_detailed_receipt(
        tmp_path,
        config="s1s2",
        stage="s2_route_optimizer",
        implementation_version="1.3.0",
        policy_version="portfolio-evolution-actionable-scope-clean-turn-v3",
        thread_id="s2-thread",
    )
    gate_sha256 = s2.record.gate_report_file_sha256
    s3 = _write_detailed_receipt(
        tmp_path,
        config="full",
        stage="s3_body_refiner",
        implementation_version="1.4.0",
        policy_version="portfolio-evolution-gate-bound-ephemeral-turn-v4",
        thread_id="s3-thread",
        input_roles=("parent_bank", "prior_stage_gate", "stage_input"),
        prior_gate_sha256=gate_sha256,
    )
    prior_gate = SimpleNamespace(
        artifact_kind="prior_stage_gate",
        artifact_file=s2.record.gate_report_file,
        artifact_file_sha256=gate_sha256,
        artifact_content_sha256=gate_sha256,
    )
    s3.record.input_artifacts = (*s3.record.input_artifacts, prior_gate)
    manifest = SimpleNamespace(records=(s1.record, s2.record, s3.record))

    _verify_evolution_session_lineage(
        tmp_path,
        manifest,
        invocations={
            "s1": SimpleNamespace(thread_id=s1.thread_id),
            "s1s2": SimpleNamespace(thread_id=s2.thread_id),
            "full": SimpleNamespace(thread_id=s3.thread_id),
        },
    )
