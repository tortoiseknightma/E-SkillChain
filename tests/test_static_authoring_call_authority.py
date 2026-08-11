from __future__ import annotations

import sys
from pathlib import Path

import pytest

from skillchain import config
import skillchain.static_authoring as authoring
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)


_REPOSITORY = Path(__file__).resolve().parents[1]


def _copy_verified_v5_invocation(root: Path):
    authoring_root = root / "specs" / "authoring"
    authoring_root.mkdir(parents=True)
    packet_path = authoring_root / "packet-v5.json"
    price_path = authoring_root / "price-v5.json"
    packet_path.write_bytes(
        (
            _REPOSITORY / "specs/authoring/authoring-packet-primary-v5-candidate.json"
        ).read_bytes()
    )
    price_path.write_bytes(
        (
            _REPOSITORY / "specs/authoring/price-qwen3-vl-flash-2026-01-22-v1.json"
        ).read_bytes()
    )
    packet_raw = parse_canonical_json(packet_path.read_bytes(), label="v5 packet")
    assert isinstance(packet_raw, dict)
    schedule = authoring.load_verified_price_schedule(
        price_path,
        expected_file_sha256=sha256_bytes(price_path.read_bytes()),
    )
    invocation_input = authoring.load_verified_authoring_invocation_input(
        packet_path,
        expected_file_sha256=sha256_bytes(packet_path.read_bytes()),
        expected_taxonomy_sha256=packet_raw["taxonomy"]["identity_sha256"],
        expected_task_specification_sha256=packet_raw["task_specification"][
            "identity_sha256"
        ],
        expected_tool_registry_sha256=packet_raw["tool_registry"]["identity_sha256"],
        public_sources=(),
        price_schedule=schedule,
    )
    return invocation_input, packet_path, price_path


def _verified_profile_and_runtime(root: Path):
    runtime_root = root / "deploy" / "authoring" / "locks-v5"
    runtime_root.mkdir(parents=True)
    engine = runtime_root / "docker.exe"
    engine.write_bytes(Path(sys.executable).read_bytes())
    image_digest = "1" * 64
    network_id = "3" * 64
    network_payload = {
        "schema_version": 1,
        "allowed_hostname": "dashscope.aliyuncs.com",
        "allowed_port": 443,
        "authoring_network_id": network_id,
        "authoring_network_internal": True,
        "authoring_network_name": "provider-egress-qwen-v5",
        "denies_direct_egress": True,
        "denies_ip_literal_proxy_targets": True,
        "denies_non_allowlisted_proxy_targets": True,
        "proxy_container_name": "skillchain-egress-proxy",
        "proxy_external_network_name": "provider-egress-external-v5",
        "proxy_image_digest": image_digest,
        "proxy_url": "http://skillchain-egress-proxy:3128",
    }
    network_payload["policy_sha256"] = sha256_bytes(
        canonical_json_bytes(network_payload)
    )
    network_path = runtime_root / "network-policy.json"
    network_path.write_bytes(canonical_json_bytes(network_payload))
    network_file_sha256 = sha256_bytes(network_path.read_bytes())
    profile_payload = {
        "schema_version": 2,
        "engine": "docker",
        "engine_path": str(engine.resolve()),
        "engine_binary_sha256": sha256_bytes(engine.read_bytes()),
        "image_reference": f"example/skillchain-author@sha256:{image_digest}",
        "image_digest": image_digest,
        "network_name": "provider-egress-qwen-v5",
        "network_id": network_id,
        "network_policy_sha256": network_file_sha256,
        "proxy_url": "http://skillchain-egress-proxy:3128",
        "proxy_container_name": "skillchain-egress-proxy",
        "proxy_image_digest": image_digest,
        "proxy_external_network_name": "provider-egress-external-v5",
        "allowed_provider_endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "credential_env_name": config.PROVIDER_API_KEY_ENV["qwen"],
        "memory_megabytes": 512,
        "cpu_count_milli": 1000,
        "pids_limit": 64,
        "timeout_seconds": 60,
    }
    profile_payload["profile_sha256"] = sha256_bytes(
        canonical_json_bytes(profile_payload)
    )
    profile_path = runtime_root / "sandbox-profile.json"
    profile_path.write_bytes(canonical_json_bytes(profile_payload))
    profile_file_sha256 = sha256_bytes(profile_path.read_bytes())
    profile = authoring.load_verified_authoring_sandbox_profile(
        profile_path,
        expected_file_sha256=profile_file_sha256,
    )
    deployment_payload = {
        "schema_version": 1,
        "runtime_version": "formal-v5-test",
        "engine_binary_sha256": profile.value.engine_binary_sha256,
        "image_digest": profile.value.image_digest,
        "network_id": profile.value.network_id,
        "network_policy_file_sha256": network_file_sha256,
        "proxy_image_digest": profile.value.proxy_image_digest,
        "sandbox_profile_file_sha256": profile_file_sha256,
        "probes": {
            name: {"exit_code": 0}
            for name in (
                "allowlisted_connect_succeeds",
                "non_allowlisted_connect_denied",
                "direct_ip_egress_blocked",
                "external_dns_blocked",
            )
        },
    }
    deployment_payload["receipt_sha256"] = sha256_bytes(
        canonical_json_bytes(deployment_payload)
    )
    deployment_path = runtime_root / "deployment-receipt.json"
    deployment_path.write_bytes(canonical_json_bytes(deployment_payload))
    deployment_sha256 = sha256_bytes(deployment_path.read_bytes())
    manifest_payload = {
        "schema_version": 1,
        "deployment_receipt_file_sha256": deployment_sha256,
        "network_policy_file_sha256": network_file_sha256,
        "sandbox_profile_file_sha256": profile_file_sha256,
    }
    manifest_path = runtime_root / "lock-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest_payload))
    runtime_bundle = {
        "runtime_version": "formal-v5-test",
        "deployment_receipt_file": deployment_path.relative_to(root).as_posix(),
        "deployment_receipt_file_sha256": deployment_sha256,
        "lock_manifest_file": manifest_path.relative_to(root).as_posix(),
        "lock_manifest_file_sha256": sha256_bytes(manifest_path.read_bytes()),
        "network_policy_file": network_path.relative_to(root).as_posix(),
        "network_policy_file_sha256": network_file_sha256,
        "sandbox_profile_file": profile_path.relative_to(root).as_posix(),
        "sandbox_profile_file_sha256": profile_file_sha256,
    }
    return profile, runtime_bundle


def _write_formal_v5_authority(
    root: Path,
    invocation_input,
    packet_path: Path,
    price_path: Path,
    profile,
    runtime_bundle: dict[str, object],
):
    value = invocation_input.value
    output_contract = authoring.build_authoring_content_output_contract(value)
    schema_path = root / "specs" / "authoring" / "content-schema-v2.json"
    schema_path.write_bytes(output_contract.json_schema_canonical_json.encode("utf-8"))
    request = authoring.build_canonical_authoring_request(value)
    request_bytes = request.canonical_bytes()
    request_path = root / "specs" / "authoring" / "request-v5.json"
    request_path.write_bytes(request_bytes)
    run_id = "llm-static-v5-authorized-test-v1"
    output_reference = f"runs/formal-authoring/{run_id}"
    claim_reference = f"specs/authoring/{run_id}-attempt-claim.json"
    receipt_reference = f"specs/authoring/{run_id}-invocation-receipt.json"
    call_budget = {
        "authorized_model": value.model.model,
        "authorized_variant": "primary",
        "required_run_id": run_id,
        "max_additional_provider_attempts": 1,
        "consume_on": "atomic-create-claim-before-container-launch",
        "reissue_policy": "new-owner-approved-deviation-only",
        "retry_fallback_repair_policy": "forbidden",
    }
    deviation_payload = {
        "schema_version": 1,
        "status": "approved_for_one_prospective_interface_repair_attempt",
        "approved_by": "project-owner",
        "call_budget": call_budget,
    }
    deviation_path = root / "specs" / "authoring" / "deviation-v5.json"
    deviation_path.write_bytes(canonical_json_bytes(deviation_payload))
    budgets = value.budgets.model_dump(mode="json")
    runtime_sha256 = sha256_bytes(canonical_json_bytes(runtime_bundle))
    authorization = {
        "authorization_id": "authoring-v5-test-one-call-v1",
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
        "runtime_lock_bundle_sha256": runtime_sha256,
        "sandbox_profile_file_sha256": profile.file_sha256,
        "required_output_directory": output_reference,
        "required_claim_file": claim_reference,
        "required_receipt_file": receipt_reference,
        "protocol_deviation_file": deviation_path.relative_to(root).as_posix(),
        "protocol_deviation_file_sha256": sha256_bytes(deviation_path.read_bytes()),
    }
    freeze_payload = {
        "schema_version": 3,
        "freeze_id": "authoring-freeze-v5-test",
        "status": "frozen",
        "approved_by": "project-owner",
        "default_variant": "primary",
        "variants": {
            "primary": {
                "input_sha256": value.input_sha256,
                "model": value.model.model,
                "model_revision": value.model.revision,
                "packet_file": packet_path.relative_to(root).as_posix(),
                "packet_file_sha256": invocation_input.file_sha256,
                "price_schedule_file": price_path.relative_to(root).as_posix(),
                "price_schedule_file_sha256": (
                    invocation_input.price_schedule.file_sha256
                ),
            }
        },
        "common": {
            "budgets": budgets,
            "decoding": value.decoding.model_dump(mode="json"),
            "canonical_request": {
                "file": request_path.relative_to(root).as_posix(),
                "file_sha256": sha256_bytes(request_bytes),
                "byte_size": len(request_bytes),
                "semantic_request_sha256": request.request_sha256,
            },
            "taxonomy_sha256": invocation_input.expected_taxonomy_sha256,
            "task_specification_sha256": (
                invocation_input.expected_task_specification_sha256
            ),
            "tool_registry_sha256": (invocation_input.expected_tool_registry_sha256),
            "output_channel": {
                "mode": output_contract.mode,
                "schema_name": output_contract.schema_name,
                "json_schema_file": schema_path.relative_to(root).as_posix(),
                "json_schema_file_sha256": output_contract.json_schema_sha256,
                "output_contract_sha256": output_contract.contract_sha256,
                "provider_response_format": {
                    "type": output_contract.provider_response_format
                },
                "provider_guarantee": output_contract.provider_guarantee,
                "json_schema_enforcement": (output_contract.json_schema_enforcement),
                "runner_packet_contract_validation": True,
                "runner_validation_engine": output_contract.runner_validation_engine,
                "tools_supplied": False,
            },
            "runtime_lock_bundle": runtime_bundle,
        },
        "call_authorization": authorization,
    }
    freeze_path = root / "specs" / "authoring" / "freeze-v5.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze_payload))
    output_path = root / output_reference
    receipt_path = root / receipt_reference
    claim_path = root / claim_reference
    return (
        freeze_path,
        sha256_bytes(freeze_path.read_bytes()),
        run_id,
        output_path,
        receipt_path,
        claim_path,
    )


def test_v5_core_and_direct_gateway_reject_without_authority_before_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    invocation_input, _, _ = _copy_verified_v5_invocation(tmp_path)
    profile, _ = _verified_profile_and_runtime(tmp_path)
    request = authoring.build_canonical_authoring_request(invocation_input.value)
    subprocess_calls: list[object] = []

    def forbidden_subprocess(*args, **kwargs):
        subprocess_calls.append((args, kwargs))
        raise AssertionError("runtime boundary must not be reached")

    monkeypatch.setattr(authoring.subprocess, "run", forbidden_subprocess)
    with pytest.raises(
        authoring.AuthoringContractError,
        match="without a consumed owner-issued call authorization",
    ):
        authoring.FormalContainerAuthoringGateway(profile).complete_once(
            request.canonical_bytes()
        )
    with pytest.raises(
        authoring.AuthoringContractError,
        match="requires a verified owner-issued call authorization",
    ):
        authoring.invoke_llm_static(
            invocation_input,
            authoring.FormalContainerAuthoringGateway(profile),
            tmp_path / "runs/formal-authoring/unauthorized-v5",
        )
    assert subprocess_calls == []


def test_candidate_freeze_cannot_grant_a_call_handle(tmp_path: Path) -> None:
    invocation_input, _, _ = _copy_verified_v5_invocation(tmp_path)
    profile, _ = _verified_profile_and_runtime(tmp_path)
    freeze_path = tmp_path / "specs" / "authoring" / "candidate-v5.json"
    freeze_path.write_bytes(
        (
            _REPOSITORY / "specs/authoring/authoring-freeze-candidate-v5.json"
        ).read_bytes()
    )
    with pytest.raises(
        authoring.AuthoringContractError,
        match="candidate authority is not executable",
    ):
        authoring.load_verified_authoring_call_authorization(
            freeze_path,
            expected_freeze_lock_file_sha256=sha256_bytes(freeze_path.read_bytes()),
            repository_root=tmp_path,
            invocation_input=invocation_input,
            sandbox_profile=profile,
            variant="primary",
            run_id="candidate-has-no-run-id",
            output_directory=tmp_path / "runs/candidate",
            receipt_path=tmp_path / "specs/authoring/candidate-receipt.json",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("file_sha256", "0" * 64),
        ("byte_size", 1),
        ("semantic_request_sha256", "f" * 64),
    ),
)
def test_authority_rejects_canonical_request_binding_tamper(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    invocation_input, packet_path, price_path = _copy_verified_v5_invocation(tmp_path)
    profile, runtime_bundle = _verified_profile_and_runtime(tmp_path)
    (
        freeze_path,
        _,
        run_id,
        output_path,
        receipt_path,
        claim_path,
    ) = _write_formal_v5_authority(
        tmp_path,
        invocation_input,
        packet_path,
        price_path,
        profile,
        runtime_bundle,
    )
    freeze = parse_canonical_json(freeze_path.read_bytes(), label="v5 freeze")
    freeze["common"]["canonical_request"][field] = replacement
    freeze_path.write_bytes(canonical_json_bytes(freeze))

    with pytest.raises(
        authoring.AuthoringContractError,
        match="canonical authoring request authority drifted",
    ):
        authoring.load_verified_authoring_call_authorization(
            freeze_path,
            expected_freeze_lock_file_sha256=sha256_bytes(freeze_path.read_bytes()),
            repository_root=tmp_path,
            invocation_input=invocation_input,
            sandbox_profile=profile,
            variant="primary",
            run_id=run_id,
            output_directory=output_path,
            receipt_path=receipt_path,
        )

    assert not claim_path.exists()


def test_authority_rejects_canonical_request_file_tamper(
    tmp_path: Path,
) -> None:
    invocation_input, packet_path, price_path = _copy_verified_v5_invocation(tmp_path)
    profile, runtime_bundle = _verified_profile_and_runtime(tmp_path)
    (
        freeze_path,
        freeze_sha256,
        run_id,
        output_path,
        receipt_path,
        claim_path,
    ) = _write_formal_v5_authority(
        tmp_path,
        invocation_input,
        packet_path,
        price_path,
        profile,
        runtime_bundle,
    )
    freeze = parse_canonical_json(freeze_path.read_bytes(), label="v5 freeze")
    request_path = tmp_path / freeze["common"]["canonical_request"]["file"]
    request_path.write_bytes(request_path.read_bytes() + b"\n")

    with pytest.raises(
        authoring.AuthoringContractError,
        match="canonical authoring request authority drifted",
    ):
        authoring.load_verified_authoring_call_authorization(
            freeze_path,
            expected_freeze_lock_file_sha256=freeze_sha256,
            repository_root=tmp_path,
            invocation_input=invocation_input,
            sandbox_profile=profile,
            variant="primary",
            run_id=run_id,
            output_directory=output_path,
            receipt_path=receipt_path,
        )

    assert not claim_path.exists()


def test_claim_is_create_only_across_reloaded_handles_and_new_gateways(
    tmp_path: Path,
) -> None:
    invocation_input, packet_path, price_path = _copy_verified_v5_invocation(tmp_path)
    profile, runtime_bundle = _verified_profile_and_runtime(tmp_path)
    (
        freeze_path,
        freeze_sha256,
        run_id,
        output_path,
        receipt_path,
        claim_path,
    ) = _write_formal_v5_authority(
        tmp_path,
        invocation_input,
        packet_path,
        price_path,
        profile,
        runtime_bundle,
    )

    def load_handle():
        return authoring.load_verified_authoring_call_authorization(
            freeze_path,
            expected_freeze_lock_file_sha256=freeze_sha256,
            repository_root=tmp_path,
            invocation_input=invocation_input,
            sandbox_profile=profile,
            variant="primary",
            run_id=run_id,
            output_directory=output_path,
            receipt_path=receipt_path,
        )

    first = load_handle()
    independently_loaded = load_handle()
    request = authoring.build_canonical_authoring_request(invocation_input.value)
    authoring.FormalContainerAuthoringGateway(profile)._authorize_v5_call(
        request,
        first,
        output_path,
    )
    claim = parse_canonical_json(claim_path.read_bytes(), label="attempt claim")
    assert claim["authorization_id"] == first.authorization_id
    assert claim["request_sha256"] == request.request_sha256
    assert claim["run_id"] == run_id
    assert claim["runtime_lock_bundle_sha256"] == (
        sha256_bytes(canonical_json_bytes(runtime_bundle))
    )

    with pytest.raises(
        authoring.AuthoringContractError,
        match="authorized attempt claim already exists",
    ):
        authoring.FormalContainerAuthoringGateway(profile)._authorize_v5_call(
            request,
            independently_loaded,
            output_path,
        )
    with pytest.raises(
        authoring.AuthoringContractError,
        match="authorized attempt claim already exists",
    ):
        load_handle()


@pytest.mark.parametrize(
    ("nested_field", "nested_reference", "expected_missing_reference"),
    (
        (
            "required_claim_file",
            "runs/formal-authoring",
            "runs/formal-authoring",
        ),
        (
            "required_receipt_file",
            "specs/authoring/claims/invocation-receipt.json",
            "specs/authoring/claims",
        ),
    ),
)
def test_authority_rejects_reverse_and_cross_path_nesting_before_claim(
    tmp_path: Path,
    nested_field: str,
    nested_reference: str,
    expected_missing_reference: str,
) -> None:
    invocation_input, packet_path, price_path = _copy_verified_v5_invocation(tmp_path)
    profile, runtime_bundle = _verified_profile_and_runtime(tmp_path)
    (
        freeze_path,
        _,
        run_id,
        output_path,
        receipt_path,
        claim_path,
    ) = _write_formal_v5_authority(
        tmp_path,
        invocation_input,
        packet_path,
        price_path,
        profile,
        runtime_bundle,
    )
    freeze = parse_canonical_json(freeze_path.read_bytes(), label="v5 freeze")
    freeze["call_authorization"][nested_field] = nested_reference
    if nested_field == "required_receipt_file":
        freeze["call_authorization"]["required_claim_file"] = "specs/authoring/claims"
        receipt_path = tmp_path / nested_reference
        claim_path = tmp_path / "specs/authoring/claims"
    freeze_path.write_bytes(canonical_json_bytes(freeze))

    with pytest.raises(
        authoring.AuthoringContractError,
        match="must be pairwise distinct and non-nested",
    ):
        authoring.load_verified_authoring_call_authorization(
            freeze_path,
            expected_freeze_lock_file_sha256=sha256_bytes(freeze_path.read_bytes()),
            repository_root=tmp_path,
            invocation_input=invocation_input,
            sandbox_profile=profile,
            variant="primary",
            run_id=run_id,
            output_directory=output_path,
            receipt_path=receipt_path,
        )

    assert not (tmp_path / expected_missing_reference).exists()
    assert not claim_path.exists()


def test_runtime_authority_rejects_coordinated_rehash_of_failed_probe(
    tmp_path: Path,
) -> None:
    invocation_input, packet_path, price_path = _copy_verified_v5_invocation(tmp_path)
    profile, runtime_bundle = _verified_profile_and_runtime(tmp_path)
    deployment_path = tmp_path / str(runtime_bundle["deployment_receipt_file"])
    deployment = parse_canonical_json(
        deployment_path.read_bytes(),
        label="deployment receipt",
    )
    deployment.pop("receipt_sha256")
    deployment["probes"]["direct_ip_egress_blocked"]["exit_code"] = 9
    deployment["receipt_sha256"] = sha256_bytes(canonical_json_bytes(deployment))
    deployment_path.write_bytes(canonical_json_bytes(deployment))
    runtime_bundle["deployment_receipt_file_sha256"] = sha256_bytes(
        deployment_path.read_bytes()
    )
    manifest_path = tmp_path / str(runtime_bundle["lock_manifest_file"])
    manifest = parse_canonical_json(
        manifest_path.read_bytes(),
        label="runtime manifest",
    )
    manifest["deployment_receipt_file_sha256"] = runtime_bundle[
        "deployment_receipt_file_sha256"
    ]
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    runtime_bundle["lock_manifest_file_sha256"] = sha256_bytes(
        manifest_path.read_bytes()
    )
    (
        freeze_path,
        freeze_sha256,
        run_id,
        output_path,
        receipt_path,
        _,
    ) = _write_formal_v5_authority(
        tmp_path,
        invocation_input,
        packet_path,
        price_path,
        profile,
        runtime_bundle,
    )
    with pytest.raises(
        authoring.AuthoringContractError,
        match="runtime lock bundle cross-check failed",
    ):
        authoring.load_verified_authoring_call_authorization(
            freeze_path,
            expected_freeze_lock_file_sha256=freeze_sha256,
            repository_root=tmp_path,
            invocation_input=invocation_input,
            sandbox_profile=profile,
            variant="primary",
            run_id=run_id,
            output_directory=output_path,
            receipt_path=receipt_path,
        )
