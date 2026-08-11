from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from skillchain import config
from skillchain import static_authoring as authoring
from skillchain.llm import LLMResponse, LLMToolCall, LLMUsage
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

from scripts import run_formal_authoring as runner


REAL_ROOT = Path(__file__).resolve().parents[1]
FREEZE_RELATIVE = Path("specs/authoring/authoring-freeze-lock-v4.json")
CANDIDATE_RELATIVE = Path("specs/authoring/authoring-freeze-candidate-v5.json")


def _copy_freeze_tree(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    freeze = json.loads((REAL_ROOT / FREEZE_RELATIVE).read_text(encoding="utf-8"))
    variant = freeze["variants"]["primary"]
    common = freeze["common"]
    authorization = freeze["call_authorization"]
    runtime = common["runtime_lock_bundle"]
    relative_files = {
        FREEZE_RELATIVE.as_posix(),
        variant["packet_file"],
        variant["price_schedule_file"],
        common["output_channel"]["tool_definition_file"],
        authorization["protocol_deviation_file"],
        authorization["prior_adjudication_receipt_file"],
        authorization["unknown_outcome_incident_file"],
        runtime["lock_manifest_file"],
        runtime["sandbox_profile_file"],
        runtime["network_policy_file"],
        runtime["deployment_receipt_file"],
    }
    for relative in sorted(relative_files):
        source = REAL_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return tmp_path / FREEZE_RELATIVE, freeze


def _runner_argv(root: Path, freeze_path: Path, freeze: dict[str, object]) -> list[str]:
    authorization = freeze["call_authorization"]
    runtime = freeze["common"]["runtime_lock_bundle"]
    return [
        "run_formal_authoring.py",
        "--freeze-lock",
        str(freeze_path),
        "--freeze-lock-sha256",
        sha256_bytes(freeze_path.read_bytes()),
        "--variant",
        "primary",
        "--sandbox-profile",
        str(root / runtime["sandbox_profile_file"]),
        "--sandbox-profile-sha256",
        runtime["sandbox_profile_file_sha256"],
        "--run-id",
        authorization["required_run_id"],
        "--output",
        str(root / authorization["required_output_directory"]),
        "--receipt",
        str(root / authorization["required_receipt_file"]),
    ]


def _copy_v5_authority_tree(
    tmp_path: Path,
) -> tuple[
    Path,
    dict[str, object],
    object,
    object,
    Path,
    Path,
    Path,
]:
    candidate = json.loads((REAL_ROOT / CANDIDATE_RELATIVE).read_text(encoding="utf-8"))
    proposal = candidate["proposal"]
    _, legacy_freeze = _copy_freeze_tree(tmp_path)
    for relative in (
        proposal["packet_file"],
        proposal["price_schedule_file"],
    ):
        source = REAL_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    packet_path = tmp_path / proposal["packet_file"]
    price_path = tmp_path / proposal["price_schedule_file"]
    schedule = authoring.load_verified_price_schedule(
        price_path,
        expected_file_sha256=proposal["price_schedule_file_sha256"],
    )
    invocation_input = authoring.load_verified_authoring_invocation_input(
        packet_path,
        expected_file_sha256=proposal["packet_file_sha256"],
        expected_taxonomy_sha256=proposal["taxonomy_sha256"],
        expected_task_specification_sha256=proposal["task_specification_sha256"],
        expected_tool_registry_sha256=proposal["tool_registry_sha256"],
        public_sources=(),
        price_schedule=schedule,
    )
    runtime = deepcopy(legacy_freeze["common"]["runtime_lock_bundle"])
    profile_path = tmp_path / runtime["sandbox_profile_file"]
    profile = authoring.load_verified_authoring_sandbox_profile(
        profile_path,
        expected_file_sha256=runtime["sandbox_profile_file_sha256"],
    )
    value = invocation_input.value
    output_contract = authoring.build_authoring_content_output_contract(value)
    schema_path = tmp_path / "specs/authoring/v5-content-schema.json"
    schema_path.write_bytes(output_contract.json_schema_canonical_json.encode("utf-8"))
    request = authoring.build_canonical_authoring_request(value)
    request_bytes = request.canonical_bytes()
    request_path = tmp_path / "specs/authoring/v5-request.json"
    request_path.write_bytes(request_bytes)
    run_id = "llm-static-v5-authorized-runner-test-v1"
    output_reference = f"runs/formal-authoring/{run_id}"
    claim_reference = f"specs/authoring/{run_id}-attempt-claim.json"
    receipt_reference = f"specs/authoring/{run_id}-receipt.json"
    call_budget = {
        "authorized_model": value.model.model,
        "authorized_variant": "primary",
        "required_run_id": run_id,
        "max_additional_provider_attempts": 1,
        "consume_on": "atomic-create-claim-before-container-launch",
        "reissue_policy": "new-owner-approved-deviation-only",
        "retry_fallback_repair_policy": "forbidden",
    }
    deviation = {
        "schema_version": 1,
        "status": "approved_for_one_prospective_interface_repair_attempt",
        "approved_by": "project-owner",
        "call_budget": call_budget,
    }
    deviation_path = tmp_path / "specs/authoring/v5-protocol-deviation.json"
    deviation_path.write_bytes(canonical_json_bytes(deviation))
    budgets = value.budgets.model_dump(mode="json")
    authorization = {
        "authorization_id": "authoring-v5-runner-test-one-call-v1",
        "authority_issued": True,
        "provider_call_authorized": True,
        "budget_authorized": True,
        "runtime_lock_frozen": True,
        "approved_by": "project-owner",
        **call_budget,
        "authoring_input_file_sha256": invocation_input.file_sha256,
        "authoring_input_sha256": value.input_sha256,
        "authorized_budgets_sha256": sha256_bytes(canonical_json_bytes(budgets)),
        "output_contract_sha256": output_contract.contract_sha256,
        "runtime_lock_bundle_sha256": sha256_bytes(canonical_json_bytes(runtime)),
        "sandbox_profile_file_sha256": profile.file_sha256,
        "required_output_directory": output_reference,
        "required_claim_file": claim_reference,
        "required_receipt_file": receipt_reference,
        "protocol_deviation_file": deviation_path.relative_to(tmp_path).as_posix(),
        "protocol_deviation_file_sha256": sha256_bytes(deviation_path.read_bytes()),
    }
    freeze = {
        "schema_version": 3,
        "freeze_id": "authoring-v5-runner-test-freeze",
        "status": "frozen",
        "approved_by": "project-owner",
        "default_variant": "primary",
        "variants": {
            "primary": {
                "input_sha256": value.input_sha256,
                "model": value.model.model,
                "model_revision": value.model.revision,
                "packet_file": packet_path.relative_to(tmp_path).as_posix(),
                "packet_file_sha256": invocation_input.file_sha256,
                "price_schedule_file": price_path.relative_to(tmp_path).as_posix(),
                "price_schedule_file_sha256": (
                    invocation_input.price_schedule.file_sha256
                ),
            }
        },
        "common": {
            "budgets": budgets,
            "decoding": value.decoding.model_dump(mode="json"),
            "canonical_request": {
                "file": request_path.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_bytes(request_bytes),
                "byte_size": len(request_bytes),
                "semantic_request_sha256": request.request_sha256,
            },
            "taxonomy_sha256": invocation_input.expected_taxonomy_sha256,
            "task_specification_sha256": (
                invocation_input.expected_task_specification_sha256
            ),
            "tool_registry_sha256": invocation_input.expected_tool_registry_sha256,
            "output_channel": {
                "mode": output_contract.mode,
                "schema_name": output_contract.schema_name,
                "json_schema_file": schema_path.relative_to(tmp_path).as_posix(),
                "json_schema_file_sha256": output_contract.json_schema_sha256,
                "output_contract_sha256": output_contract.contract_sha256,
                "provider_response_format": {
                    "type": output_contract.provider_response_format
                },
                "provider_guarantee": output_contract.provider_guarantee,
                "json_schema_enforcement": output_contract.json_schema_enforcement,
                "runner_packet_contract_validation": True,
                "runner_validation_engine": output_contract.runner_validation_engine,
                "tools_supplied": False,
            },
            "runtime_lock_bundle": runtime,
        },
        "call_authorization": authorization,
    }
    freeze_path = tmp_path / "specs/authoring/v5-freeze.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    return (
        freeze_path,
        freeze,
        invocation_input,
        profile,
        tmp_path / output_reference,
        tmp_path / receipt_reference,
        tmp_path / claim_reference,
    )


def _consume_v5_claim(invocation_input, gateway, output, call_authorization):
    request = authoring.build_canonical_authoring_request(invocation_input.value)
    gateway._authorize_v5_call(request, call_authorization, Path(output))
    return request


def test_candidate_v5_stops_before_authority_loader_or_llm(tmp_path, monkeypatch):
    freeze_path = tmp_path / CANDIDATE_RELATIVE
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_bytes((REAL_ROOT / CANDIDATE_RELATIVE).read_bytes())
    calls = {"authority": 0, "llm": 0}

    def authority_forbidden(*_args, **_kwargs):
        calls["authority"] += 1
        raise AssertionError("candidate must stop before authority loading")

    def llm_forbidden(*_args, **_kwargs):
        calls["llm"] += 1
        raise AssertionError("candidate must stop before LLM invocation")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "load_verified_authoring_call_authorization",
        authority_forbidden,
    )
    monkeypatch.setattr(runner, "invoke_llm_static", llm_forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_formal_authoring.py",
            "--freeze-lock",
            str(freeze_path),
            "--freeze-lock-sha256",
            sha256_bytes(freeze_path.read_bytes()),
            "--variant",
            "primary",
            "--sandbox-profile",
            str(tmp_path / "never-read-profile.json"),
            "--sandbox-profile-sha256",
            "0" * 64,
            "--run-id",
            "v5-candidate-must-not-run",
            "--output",
            str(tmp_path / "must-not-exist"),
            "--receipt",
            str(tmp_path / "must-not-exist-receipt.json"),
        ],
    )

    with pytest.raises(
        PermissionError,
        match="candidate is not authorized",
    ):
        runner.main()

    assert calls == {"authority": 0, "llm": 0}


def test_frozen_v5_without_call_authority_stops_before_llm(tmp_path, monkeypatch):
    freeze_path, freeze, _, _, output, receipt, claim = _copy_v5_authority_tree(
        tmp_path
    )
    freeze.pop("call_authorization")
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    calls = 0

    def llm_forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("unauthorized freeze must stop before LLM invocation")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "invoke_llm_static", llm_forbidden)
    monkeypatch.setattr(
        "sys.argv",
        _runner_argv(
            tmp_path,
            freeze_path,
            {
                **freeze,
                "call_authorization": {
                    "required_run_id": "llm-static-v5-authorized-runner-test-v1",
                    "required_output_directory": output.relative_to(
                        tmp_path
                    ).as_posix(),
                    "required_receipt_file": receipt.relative_to(tmp_path).as_posix(),
                },
            },
        ),
    )

    with pytest.raises(
        authoring.AuthoringContractError,
        match="lacks formal call authority",
    ):
        runner.main()

    assert calls == 0
    assert not output.exists()
    assert not receipt.exists()
    assert not claim.exists()


def test_authorized_v5_success_receipt_binds_core_claim(
    tmp_path,
    monkeypatch,
) -> None:
    (
        freeze_path,
        freeze,
        invocation_input,
        _profile,
        output,
        receipt_path,
        claim_path,
    ) = _copy_v5_authority_tree(tmp_path)
    value = invocation_input.value
    response = LLMResponse(
        provider=value.model.provider,
        endpoint=value.model.endpoint,
        requested_model=value.model.model,
        response_model=value.model.model,
        request_id="provider-v5-success-fixture",
        text='{"schema_version":2,"drafts":[]}',
        tool_calls=(),
        usage=LLMUsage(input_tokens=11, output_tokens=7),
        finish_reason="stop",
        latency_ms=9,
    )
    calls = 0

    def invoke_fixture(
        verified_input,
        gateway,
        artifact_dir,
        *,
        call_authorization,
    ):
        nonlocal calls
        calls += 1
        request = _consume_v5_claim(
            verified_input,
            gateway,
            artifact_dir,
            call_authorization,
        )
        return SimpleNamespace(
            response=response,
            authoring_input_sha256=verified_input.value.input_sha256,
            request=request,
            request_file_sha256="3" * 64,
            response_file_sha256="4" * 64,
            pre_review_draft_file_sha256="5" * 64,
            isolation_attestation=SimpleNamespace(
                attestation_sha256="6" * 64,
                elapsed_ms=12,
            ),
            isolation_attestation_file_sha256="7" * 64,
            cost_microusd=10,
        )

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "invoke_llm_static", invoke_fixture)
    monkeypatch.setattr("sys.argv", _runner_argv(tmp_path, freeze_path, freeze))

    assert runner.main() == 0
    assert calls == 1
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "awaiting_human_review_and_authority_runtime_compile"
    assert receipt["authorization_consumed"] is True
    assert receipt["attempt_claim_file_sha256"] == sha256_bytes(claim_path.read_bytes())
    assert receipt["attempt_claim_sha256"] == claim["claim_sha256"]
    assert receipt["request_sha256"] == claim["request_sha256"]
    assert (
        receipt["protocol_deviation_file_sha256"]
        == claim["protocol_deviation_file_sha256"]
    )
    assert receipt["output_contract_sha256"] == claim["output_contract_sha256"]
    assert receipt["runtime_lock_bundle_sha256"] == claim["runtime_lock_bundle_sha256"]
    assert receipt["runtime_file_sha256s"] == claim["runtime_file_sha256s"]
    assert receipt["response_text_sha256"] == sha256_bytes(
        response.text.encode("utf-8")
    )
    assert receipt["submission_arguments_sha256"] is None
    assert receipt["tool_call_count"] == 0
    assert not output.exists()


def test_authorized_v5_rejected_draft_writes_consumed_failure_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    (
        freeze_path,
        freeze,
        _invocation_input,
        _profile,
        output,
        receipt_path,
        claim_path,
    ) = _copy_v5_authority_tree(tmp_path)
    value = _invocation_input.value
    response = LLMResponse(
        provider=value.model.provider,
        endpoint=value.model.endpoint,
        requested_model=value.model.model,
        response_model=value.model.model,
        request_id="provider-v5-rejected-draft-fixture",
        text="{}",
        tool_calls=(),
        usage=LLMUsage(input_tokens=13, output_tokens=2),
        finish_reason="stop",
        latency_ms=8,
    )

    def reject_after_provider_response(
        verified_input,
        gateway,
        artifact_dir,
        *,
        call_authorization,
    ):
        _consume_v5_claim(
            verified_input,
            gateway,
            artifact_dir,
            call_authorization,
        )
        output.mkdir(parents=True)
        (output / "authoring-response.json").write_bytes(
            canonical_json_bytes(response.model_dump(mode="json"))
        )
        raise authoring.AuthoringContractError("author content JSON is not valid")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "invoke_llm_static",
        reject_after_provider_response,
    )
    monkeypatch.setattr("sys.argv", _runner_argv(tmp_path, freeze_path, freeze))

    with pytest.raises(
        authoring.AuthoringContractError,
        match="author content JSON is not valid",
    ):
        runner.main()

    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "provider_call_completed_draft_rejected"
    assert receipt["authorization_consumed"] is True
    assert receipt["provider_outcome"] == "completed"
    assert receipt["formal_provider_call_eligible"] is True
    assert receipt["formal_invocation_eligible"] is False
    assert receipt["draft_formal_eligible"] is False
    assert receipt["attempt_claim_file_sha256"] == sha256_bytes(claim_path.read_bytes())
    assert receipt["request_sha256"] == claim["request_sha256"]
    assert receipt["output_contract_sha256"] == claim["output_contract_sha256"]
    assert receipt["runtime_lock_bundle_sha256"] == claim["runtime_lock_bundle_sha256"]
    assert receipt["response_text_sha256"] == sha256_bytes(b"{}")
    assert receipt["submission_arguments_sha256"] is None
    assert receipt["tool_call_count"] == 0


def test_authorized_v5_unknown_post_claim_failure_writes_consumed_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    (
        freeze_path,
        freeze,
        _invocation_input,
        _profile,
        output,
        receipt_path,
        claim_path,
    ) = _copy_v5_authority_tree(tmp_path)

    def fail_after_claim(
        verified_input,
        gateway,
        artifact_dir,
        *,
        call_authorization,
    ):
        _consume_v5_claim(
            verified_input,
            gateway,
            artifact_dir,
            call_authorization,
        )
        raise RuntimeError("post-claim provider outcome unknown")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "invoke_llm_static", fail_after_claim)
    monkeypatch.setattr("sys.argv", _runner_argv(tmp_path, freeze_path, freeze))

    with pytest.raises(RuntimeError, match="provider outcome unknown"):
        runner.main()

    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "authorization_consumed_provider_outcome_unknown"
    assert receipt["authorization_consumed"] is True
    assert receipt["provider_outcome"] == "unknown"
    assert receipt["attempt_claim_file_sha256"] == sha256_bytes(claim_path.read_bytes())
    assert receipt["request_sha256"] == claim["request_sha256"]
    assert (
        receipt["protocol_deviation_file_sha256"]
        == (claim["protocol_deviation_file_sha256"])
    )
    assert receipt["output_contract_sha256"] == claim["output_contract_sha256"]
    assert receipt["runtime_lock_bundle_sha256"] == claim["runtime_lock_bundle_sha256"]
    assert not output.exists()


def test_frozen_runtime_bundle_accepts_only_bound_profile(tmp_path, monkeypatch):
    _, freeze = _copy_freeze_tree(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runtime = freeze["common"]["runtime_lock_bundle"]
    profile_path = tmp_path / runtime["sandbox_profile_file"]

    profile, hashes = runner._load_frozen_runtime(
        freeze["common"],
        supplied_profile_path=profile_path,
        supplied_profile_sha256=runtime["sandbox_profile_file_sha256"],
    )

    assert profile.value.image_digest
    assert hashes["lock_manifest_file_sha256"] == (runtime["lock_manifest_file_sha256"])
    with pytest.raises(ValueError, match="freeze-bound profile"):
        runner._load_frozen_runtime(
            freeze["common"],
            supplied_profile_path=(tmp_path / runtime["network_policy_file"]),
            supplied_profile_sha256=runtime["network_policy_file_sha256"],
        )


def test_frozen_runtime_bundle_rejects_failed_probe(tmp_path, monkeypatch):
    _, freeze = _copy_freeze_tree(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    common = deepcopy(freeze["common"])
    runtime = common["runtime_lock_bundle"]
    receipt_path = tmp_path / runtime["deployment_receipt_file"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["probes"]["external_dns_blocked"]["exit_code"] = 1
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    receipt_bytes = canonical_json_bytes(receipt)
    receipt_path.write_bytes(receipt_bytes)
    runtime["deployment_receipt_file_sha256"] = sha256_bytes(receipt_bytes)

    manifest_path = tmp_path / runtime["lock_manifest_file"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deployment_receipt_file_sha256"] = sha256_bytes(receipt_bytes)
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    runtime["lock_manifest_file_sha256"] = sha256_bytes(manifest_bytes)

    with pytest.raises(ValueError, match="probes"):
        runner._load_frozen_runtime(
            common,
            supplied_profile_path=tmp_path / runtime["sandbox_profile_file"],
            supplied_profile_sha256=runtime["sandbox_profile_file_sha256"],
        )


def test_deviation_and_call_authorization_must_match(tmp_path, monkeypatch):
    _, freeze = _copy_freeze_tree(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    authorization = deepcopy(freeze["call_authorization"])
    authorization["required_run_id"] = "different-run"

    with pytest.raises(ValueError, match="evidence semantics"):
        runner._validate_structured_evidence(authorization)


def test_claim_survives_pre_provider_failure_and_blocks_reentry(tmp_path, monkeypatch):
    freeze_path, freeze = _copy_freeze_tree(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "invoke_llm_static",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("pre-provider fixture failure")
        ),
    )
    argv = _runner_argv(tmp_path, freeze_path, freeze)
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(RuntimeError, match="pre-provider fixture"):
        runner.main()

    authorization = freeze["call_authorization"]
    claim_path = tmp_path / authorization["required_claim_file"]
    receipt_path = tmp_path / authorization["required_receipt_file"]
    assert claim_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "authorization_consumed_provider_outcome_unknown"
    assert receipt["authorization_consumed"] is True
    assert receipt["provider_outcome"] == "unknown"
    assert receipt["attempt_claim_file_sha256"] == sha256_bytes(claim_path.read_bytes())
    assert receipt["retry_performed"] is False
    assert receipt["fallback_performed"] is False
    assert receipt["repair_performed"] is False
    assert (
        receipt["lock_manifest_file_sha256"]
        == (freeze["common"]["runtime_lock_bundle"]["lock_manifest_file_sha256"])
    )

    receipt_path.unlink()
    with pytest.raises(FileExistsError):
        runner.main()


def test_success_receipt_carries_every_attempt_binding(tmp_path, monkeypatch):
    freeze_path, freeze = _copy_freeze_tree(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    model = config.LEGACY_AUTHOR_MODEL
    response = LLMResponse(
        provider="qwen",
        endpoint=config.PROVIDER_ENDPOINTS["qwen"],
        requested_model=model,
        response_model=model,
        request_id="provider-success-fixture",
        text="",
        tool_calls=(
            LLMToolCall(
                call_id="submission-1",
                name="submit_authoring_payload",
                arguments_json='{"drafts":[],"schema_version":1}',
            ),
        ),
        usage=LLMUsage(input_tokens=10, output_tokens=5),
        finish_reason="tool_calls",
        latency_ms=7,
    )
    invocation = SimpleNamespace(
        response=response,
        authoring_input_sha256="1" * 64,
        request=SimpleNamespace(request_sha256="2" * 64),
        request_file_sha256="3" * 64,
        response_file_sha256="4" * 64,
        pre_review_draft_file_sha256="5" * 64,
        isolation_attestation=SimpleNamespace(
            attestation_sha256="6" * 64,
            elapsed_ms=11,
        ),
        isolation_attestation_file_sha256="7" * 64,
        cost_microusd=9,
    )
    monkeypatch.setattr(
        runner, "invoke_llm_static", lambda *_args, **_kwargs: invocation
    )
    monkeypatch.setattr(
        "sys.argv",
        _runner_argv(tmp_path, freeze_path, freeze),
    )

    assert runner.main() == 0

    authorization = freeze["call_authorization"]
    receipt_path = tmp_path / authorization["required_receipt_file"]
    claim_path = tmp_path / authorization["required_claim_file"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["formal_provider_call_eligible"] is True
    assert receipt["draft_formal_eligible"] is True
    assert receipt["attempt_claim_file_sha256"] == sha256_bytes(claim_path.read_bytes())
    assert (
        receipt["protocol_deviation_file_sha256"]
        == (authorization["protocol_deviation_file_sha256"])
    )
    assert (
        receipt["unknown_outcome_incident_file_sha256"]
        == (authorization["unknown_outcome_incident_file_sha256"])
    )
    assert (
        receipt["tool_definition_file_sha256"]
        == (freeze["common"]["output_channel"]["tool_definition_file_sha256"])
    )
    assert receipt["retry_performed"] is False
    assert receipt["fallback_performed"] is False
    assert receipt["repair_performed"] is False
