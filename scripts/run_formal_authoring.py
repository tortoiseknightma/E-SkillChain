"""Execute one create-only formal LLMStatic authoring invocation.

This command intentionally stops before human review and Bank compilation.
The invocation packet freezes ToolSpec but defers the machine-specific
authority runtime until ``finalize_llm_static``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from skillchain.llm import LLMResponse
from skillchain.static_authoring import (
    AUTHORING_SUBMISSION_TOOL_NAME,
    FORCED_SUBMISSION_RESPONSE_FORMAT,
    PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    FormalContainerAuthoringGateway,
    VerifiedAuthoringCallAuthorization,
    authoring_response_envelope_is_complete,
    authoring_submission_tool_definition,
    load_verified_authoring_call_authorization,
    load_verified_authoring_invocation_input,
    load_verified_authoring_sandbox_profile,
    load_verified_price_schedule,
    invoke_llm_static,
)
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[1]


def _submission_tool_name(response_format: str) -> str | None:
    """Return the exact submission envelope name for a structured format."""

    if response_format == FORCED_SUBMISSION_RESPONSE_FORMAT:
        return AUTHORING_SUBMISSION_TOOL_NAME
    return None


def _sha(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _locked_object(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    content = read_stable_regular_file(path, label=label)
    if sha256_bytes(content) != expected_sha256:
        raise ValueError(f"{label} file digest mismatch")
    raw = parse_canonical_json(content, label=label)
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain an object")
    return raw


def _repo_file(relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    candidate = (ROOT / relative).resolve(strict=True)
    try:
        candidate.relative_to(ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} must be a regular non-symbolic file")
    return candidate


def _repo_creation_path(relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    candidate = (ROOT / relative).resolve(strict=False)
    try:
        candidate.relative_to(ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return candidate


def _verify_self_hash(value: dict[str, object], field: str, label: str) -> str:
    actual = _sha(str(value.get(field)), f"{label} self digest")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if actual != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError(f"{label} self digest mismatch")
    return actual


def _load_frozen_runtime(
    common: dict[str, object],
    *,
    supplied_profile_path: Path,
    supplied_profile_sha256: str,
):
    bundle = common.get("runtime_lock_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("freeze lacks a runtime lock bundle")
    manifest_path = _repo_file(
        bundle.get("lock_manifest_file"),
        "runtime lock manifest",
    )
    manifest_sha = _sha(
        str(bundle.get("lock_manifest_file_sha256")),
        "runtime lock manifest digest",
    )
    manifest = _locked_object(
        manifest_path,
        manifest_sha,
        "runtime lock manifest",
    )
    expected_files = {
        "deployment_receipt": "deployment-receipt.json",
        "network_policy": "network-policy.json",
        "sandbox_profile": "sandbox-profile.json",
    }
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    objects: dict[str, dict[str, object]] = {}
    for key, filename in expected_files.items():
        path = _repo_file(bundle.get(f"{key}_file"), f"runtime {key}")
        expected_path = (manifest_path.parent / filename).resolve(strict=True)
        if path != expected_path:
            raise ValueError(f"runtime {key} is outside the manifest bundle")
        file_sha = _sha(
            str(bundle.get(f"{key}_file_sha256")),
            f"runtime {key} digest",
        )
        raw = _locked_object(path, file_sha, f"runtime {key}")
        paths[key] = path
        hashes[key] = file_sha
        objects[key] = raw
    if (
        manifest.get("deployment_receipt_file_sha256") != hashes["deployment_receipt"]
        or manifest.get("network_policy_file_sha256") != hashes["network_policy"]
        or manifest.get("sandbox_profile_file_sha256") != hashes["sandbox_profile"]
    ):
        raise ValueError("runtime lock manifest relationships drifted")
    supplied_resolved = supplied_profile_path.resolve(strict=True)
    if (
        supplied_resolved != paths["sandbox_profile"]
        or supplied_profile_sha256 != hashes["sandbox_profile"]
    ):
        raise ValueError("CLI sandbox profile is not the freeze-bound profile")
    profile = load_verified_authoring_sandbox_profile(
        paths["sandbox_profile"],
        expected_file_sha256=hashes["sandbox_profile"],
    )
    policy = objects["network_policy"]
    receipt = objects["deployment_receipt"]
    _verify_self_hash(policy, "policy_sha256", "runtime network policy")
    _verify_self_hash(receipt, "receipt_sha256", "runtime deployment receipt")
    probes = receipt.get("probes")
    required_probes = {
        "allowlisted_connect_succeeds",
        "non_allowlisted_connect_denied",
        "direct_ip_egress_blocked",
        "external_dns_blocked",
    }
    if (
        not isinstance(probes, dict)
        or set(probes) != required_probes
        or any(
            not isinstance(item, dict) or item.get("exit_code") != 0
            for item in probes.values()
        )
    ):
        raise ValueError("runtime deployment probes are incomplete or failed")
    value = profile.value
    if (
        value.network_policy_sha256 != hashes["network_policy"]
        or value.image_digest != policy.get("proxy_image_digest")
        or value.proxy_image_digest != policy.get("proxy_image_digest")
        or value.network_id != policy.get("authoring_network_id")
        or value.network_name != policy.get("authoring_network_name")
        or value.proxy_container_name != policy.get("proxy_container_name")
        or value.proxy_external_network_name
        != policy.get("proxy_external_network_name")
        or value.proxy_url != policy.get("proxy_url")
        or value.allowed_provider_endpoint
        != f"https://{policy.get('allowed_hostname')}/compatible-mode/v1"
        or value.engine_binary_sha256 != receipt.get("engine_binary_sha256")
        or value.image_digest != receipt.get("image_digest")
        or value.proxy_image_digest != receipt.get("proxy_image_digest")
        or value.network_id != receipt.get("network_id")
        or hashes["network_policy"] != receipt.get("network_policy_file_sha256")
        or hashes["sandbox_profile"] != receipt.get("sandbox_profile_file_sha256")
        or receipt.get("runtime_version") != bundle.get("runtime_version")
    ):
        raise ValueError("runtime lock bundle cross-check failed")
    return profile, {
        "deployment_receipt_file_sha256": hashes["deployment_receipt"],
        "lock_manifest_file_sha256": manifest_sha,
        "network_policy_file_sha256": hashes["network_policy"],
        "sandbox_profile_file_sha256": hashes["sandbox_profile"],
    }


def _validate_structured_evidence(
    authorization: dict[str, object],
) -> dict[str, str]:
    deviation_path = _repo_file(
        authorization.get("protocol_deviation_file"),
        "protocol deviation",
    )
    deviation_sha = _sha(
        str(authorization.get("protocol_deviation_file_sha256")),
        "protocol deviation digest",
    )
    deviation = _locked_object(deviation_path, deviation_sha, "protocol deviation")
    _verify_self_hash(deviation, "deviation_sha256", "protocol deviation")
    prior_path = _repo_file(
        authorization.get("prior_adjudication_receipt_file"),
        "prior adjudication receipt",
    )
    prior_sha = _sha(
        str(authorization.get("prior_adjudication_receipt_file_sha256")),
        "prior adjudication receipt digest",
    )
    prior = _locked_object(
        prior_path,
        prior_sha,
        "prior adjudication receipt",
    )
    _verify_self_hash(prior, "receipt_sha256", "prior adjudication receipt")
    incident_path = _repo_file(
        authorization.get("unknown_outcome_incident_file"),
        "unknown-outcome incident",
    )
    incident_sha = _sha(
        str(authorization.get("unknown_outcome_incident_file_sha256")),
        "unknown-outcome incident digest",
    )
    incident = _locked_object(
        incident_path,
        incident_sha,
        "unknown-outcome incident",
    )
    call_budget = deviation.get("call_budget")
    trigger = deviation.get("trigger")
    incident_failure = incident.get("failure")
    if (
        deviation.get("status")
        != "approved_for_one_prospective_interface_repair_attempt"
        or not isinstance(call_budget, dict)
        or call_budget.get("max_additional_provider_attempts") != 1
        or call_budget.get("retry_fallback_repair_policy") != "forbidden"
        or any(
            call_budget.get(key) != authorization.get(key)
            for key in (
                "authorized_model",
                "authorized_variant",
                "required_run_id",
                "known_auditable_formal_provider_responses_counted",
                "unknown_provider_outcome_incidents",
                "max_additional_provider_attempts",
                "consume_on",
                "reissue_policy",
                "retry_fallback_repair_policy",
            )
        )
        or not isinstance(trigger, dict)
        or trigger.get("adjudication_receipt_file_sha256") != prior_sha
        or trigger.get("unknown_outcome_incident_file_sha256") != incident_sha
        or trigger.get("unknown_provider_generation_or_billing_outcomes") != 1
        or prior.get("run_id") != "llm-static-primary-20260724-replacement-v1"
        or prior.get("status") != "provider_call_completed_draft_rejected"
        or prior.get("formal_provider_call_eligible") is not True
        or prior.get("draft_formal_eligible") is not False
        or prior.get("retry_performed") is not False
        or prior.get("fallback_performed") is not False
        or prior.get("freeze_lock_file_sha256")
        != authorization.get("prior_freeze_lock_file_sha256")
        or incident.get("status") != "invalid_not_c1_evidence"
        or incident.get("classification")
        != "infrastructure-invalidated-authoring-attempt"
        or not isinstance(incident_failure, dict)
        or incident_failure.get("provider_generation_outcome") != "unknown"
        or incident_failure.get("provider_billing_outcome") != "unknown"
    ):
        raise ValueError("structured-submission evidence semantics drifted")
    return {
        "protocol_deviation_file_sha256": deviation_sha,
        "prior_adjudication_receipt_file_sha256": prior_sha,
        "unknown_outcome_incident_file_sha256": incident_sha,
    }


def _attempt_claim_bytes(
    *,
    authorization: dict[str, object],
    freeze_sha256: str,
    runtime_hashes: dict[str, str],
    evidence_hashes: dict[str, str],
    tool_definition_sha256: str,
) -> bytes:
    payload = {
        "schema_version": 1,
        "authorization_id": authorization["authorization_id"],
        "status": "consumed_before_container_launch",
        "consume_on": "atomic-create-claim-before-container-launch",
        "attempt_ordinal_under_deviation": 1,
        "freeze_lock_file_sha256": freeze_sha256,
        "run_id": authorization["required_run_id"],
        "variant": authorization["authorized_variant"],
        "model": authorization["authorized_model"],
        "required_claim_file": authorization["required_claim_file"],
        "required_output_directory": authorization["required_output_directory"],
        "required_receipt_file": authorization["required_receipt_file"],
        "tool_definition_file_sha256": tool_definition_sha256,
        **evidence_hashes,
        **runtime_hashes,
    }
    return canonical_json_bytes(
        {
            **payload,
            "claim_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
    )


def _call_artifact_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in (
        "authoring-input.json",
        "authoring-request.json",
        "authoring-response.json",
        "isolation-attestation.json",
        "pre-review-draft.json",
        "container-stdout.bin",
        "container-stderr.bin",
        "job-failure.json",
    ):
        path = root / name
        if path.is_file() and not path.is_symlink():
            hashes[name] = sha256_bytes(
                read_stable_regular_file(path, label=f"formal call {name}")
            )
    return hashes


def _v5_claim_receipt_bindings(
    authorization: VerifiedAuthoringCallAuthorization,
    *,
    freeze_sha256: str,
    run_id: str,
) -> dict[str, object] | None:
    """Bind a receipt to the core-created v5 claim, if it was consumed."""

    claim_path = authorization.required_claim_file
    if not os.path.lexists(claim_path):
        return None
    content = read_stable_regular_file(
        claim_path,
        label="formal authoring v5 attempt claim",
    )
    raw = parse_canonical_json(content, label="formal authoring v5 attempt claim")
    if not isinstance(raw, dict):
        raise ValueError("formal authoring v5 attempt claim must contain an object")
    claim_sha256 = _verify_self_hash(
        raw,
        "claim_sha256",
        "formal authoring v5 attempt claim",
    )
    sha_fields = (
        "protocol_deviation_file_sha256",
        "authoring_input_file_sha256",
        "authoring_input_sha256",
        "authorized_budgets_sha256",
        "output_contract_sha256",
        "runtime_lock_bundle_sha256",
        "request_sha256",
    )
    bound_sha256s = {
        field: _sha(str(raw.get(field)), f"v5 attempt claim {field}")
        for field in sha_fields
    }
    runtime_files = raw.get("runtime_file_sha256s")
    if (
        not isinstance(runtime_files, dict)
        or not runtime_files
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for name, digest in runtime_files.items()
        )
    ):
        raise ValueError("formal authoring v5 runtime claim bindings are invalid")
    if (
        raw.get("schema_version") != 1
        or raw.get("authorization_id") != authorization.authorization_id
        or raw.get("status") != "consumed_before_container_launch"
        or raw.get("consume_on") != "atomic-create-claim-before-container-launch"
        or raw.get("freeze_lock_file_sha256") != freeze_sha256
        or raw.get("run_id") != run_id
    ):
        raise ValueError("formal authoring v5 attempt claim semantics drifted")
    return {
        "authorization_id": authorization.authorization_id,
        "attempt_claim_file_sha256": sha256_bytes(content),
        "attempt_claim_sha256": claim_sha256,
        **bound_sha256s,
        "runtime_file_sha256s": dict(sorted(runtime_files.items())),
        "retry_performed": False,
        "fallback_performed": False,
        "repair_performed": False,
    }


def _persisted_response_facts(root: Path, invocation_input) -> dict[str, object]:
    """Recover provider facts after draft validation rejects a paid response."""

    response_path = root / "authoring-response.json"
    raw = parse_canonical_json(
        read_stable_regular_file(
            response_path,
            label="formal call authoring-response.json",
        ),
        label="formal call authoring-response.json",
    )
    if not isinstance(raw, dict):
        raise TypeError("formal call response must contain an object")
    response = LLMResponse.model_validate(raw)
    value = invocation_input.value
    submission_tool_name = _submission_tool_name(value.decoding.response_format)
    usage = response.usage
    schedule = value.price_schedule
    cost_microusd = (
        usage.input_tokens * schedule.input_microusd_per_million_tokens
        + usage.output_tokens * schedule.output_microusd_per_million_tokens
        + 999_999
    ) // 1_000_000
    formal_provider_call_eligible = (
        response.provider == value.model.provider
        and response.endpoint == value.model.endpoint
        and response.requested_model == value.model.model
        and response.response_model == value.model.model
        and authoring_response_envelope_is_complete(value.decoding, response)
        and usage.input_tokens > 0
        and usage.output_tokens > 0
        and usage.input_tokens <= value.budgets.max_input_tokens
        and usage.output_tokens <= value.budgets.max_output_tokens
        and usage.total_tokens <= value.budgets.max_total_tokens
        and cost_microusd <= value.budgets.max_cost_microusd
    )
    return {
        "cost_microusd": cost_microusd,
        "endpoint": response.endpoint,
        "finish_reason": response.finish_reason,
        "formal_provider_call_eligible": formal_provider_call_eligible,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "provider": response.provider,
        "provider_latency_ms": response.latency_ms,
        "provider_request_id_sha256": sha256_bytes(response.request_id.encode("utf-8")),
        "requested_model": response.requested_model,
        "response_model": response.response_model,
        "response_format": value.decoding.response_format,
        "response_text_sha256": sha256_bytes(response.text.encode("utf-8")),
        "submission_arguments_sha256": (
            sha256_bytes(response.tool_calls[0].arguments_json.encode("utf-8"))
            if len(response.tool_calls) == 1
            and response.tool_calls[0].name == submission_tool_name
            else None
        ),
        "tool_call_count": len(response.tool_calls),
        "total_tokens": usage.total_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-lock", type=Path, required=True)
    parser.add_argument("--freeze-lock-sha256", required=True)
    parser.add_argument("--variant", choices=("primary", "fallback"), default="primary")
    parser.add_argument("--sandbox-profile", type=Path, required=True)
    parser.add_argument("--sandbox-profile-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", args.run_id):
        raise ValueError("run-id must be a stable lowercase identifier")
    freeze_sha = _sha(args.freeze_lock_sha256, "freeze-lock-sha256")
    profile_sha = _sha(args.sandbox_profile_sha256, "sandbox-profile-sha256")
    freeze = _locked_object(args.freeze_lock.resolve(strict=True), freeze_sha, "freeze")
    if freeze.get("status") == "candidate_not_authorized":
        raise PermissionError(
            "authoring v5 candidate is not authorized for provider invocation"
        )
    if freeze.get("status") != "frozen":
        raise ValueError("authoring freeze lock is not frozen")
    variants = freeze.get("variants")
    common = freeze.get("common")
    if not isinstance(variants, dict) or not isinstance(common, dict):
        raise ValueError("authoring freeze lock is incomplete")
    variant = variants.get(args.variant)
    if not isinstance(variant, dict):
        raise ValueError("authoring variant is missing")
    if args.variant != freeze.get("default_variant"):
        raise ValueError(
            "fallback requires a separately approved pre-run selection record"
        )

    packet_sha = _sha(str(variant.get("packet_file_sha256")), "packet digest")
    price_sha = _sha(
        str(variant.get("price_schedule_file_sha256")), "price-schedule digest"
    )
    taxonomy_sha = _sha(str(common.get("taxonomy_sha256")), "taxonomy digest")
    tasks_sha = _sha(
        str(common.get("task_specification_sha256")), "task-specification digest"
    )
    registry_sha = _sha(str(common.get("tool_registry_sha256")), "tool-registry digest")
    packet_path = _repo_file(variant.get("packet_file"), "packet file")
    price_path = _repo_file(variant.get("price_schedule_file"), "price file")
    schedule = load_verified_price_schedule(
        price_path,
        expected_file_sha256=price_sha,
    )
    invocation_input = load_verified_authoring_invocation_input(
        packet_path,
        expected_file_sha256=packet_sha,
        expected_taxonomy_sha256=taxonomy_sha,
        expected_task_specification_sha256=tasks_sha,
        expected_tool_registry_sha256=registry_sha,
        public_sources=(),
        price_schedule=schedule,
    )
    is_v5 = (
        invocation_input.value.decoding.response_format
        == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
    )
    output = args.output.resolve(strict=False)
    receipt_path = args.receipt.resolve(strict=False)
    receipt_bindings: dict[str, object] = {}
    call_authorization: VerifiedAuthoringCallAuthorization | None = None
    if (
        invocation_input.value.decoding.response_format
        == FORCED_SUBMISSION_RESPONSE_FORMAT
    ):
        output_channel = common.get("output_channel")
        authorization = freeze.get("call_authorization")
        if not isinstance(output_channel, dict) or not isinstance(authorization, dict):
            raise ValueError(
                "forced-submission freeze lacks output-channel authorization"
            )
        tool_path = _repo_file(
            output_channel.get("tool_definition_file"),
            "submission tool definition",
        )
        tool_sha = _sha(
            str(output_channel.get("tool_definition_file_sha256")),
            "submission tool definition digest",
        )
        tool_bytes = read_stable_regular_file(
            tool_path,
            label="submission tool definition",
        )
        if sha256_bytes(tool_bytes) != tool_sha or tool_bytes != canonical_json_bytes(
            authoring_submission_tool_definition()
        ):
            raise ValueError("submission tool definition does not match runner schema")
        evidence_hashes = _validate_structured_evidence(authorization)
        required_output = _repo_creation_path(
            authorization.get("required_output_directory"),
            "required output directory",
        )
        required_receipt = _repo_creation_path(
            authorization.get("required_receipt_file"),
            "required receipt file",
        )
        required_claim = _repo_creation_path(
            authorization.get("required_claim_file"),
            "required attempt claim",
        )
        if (
            output_channel.get("mode") != "forced_single_function_submission"
            or output_channel.get("tool_name") != AUTHORING_SUBMISSION_TOOL_NAME
            or output_channel.get("tool_execution") is not False
            or output_channel.get("parallel_tool_calls") is not False
            or authorization.get("authorized_variant") != args.variant
            or authorization.get("authorized_model")
            != invocation_input.value.model.model
            or authorization.get("required_run_id") != args.run_id
            or authorization.get("max_additional_provider_attempts") != 1
            or authorization.get("known_auditable_formal_provider_responses_counted")
            != 1
            or authorization.get("unknown_provider_outcome_incidents") != 1
            or authorization.get("consume_on")
            != "atomic-create-claim-before-container-launch"
            or authorization.get("reissue_policy")
            != "new-owner-approved-deviation-only"
            or authorization.get("retry_fallback_repair_policy") != "forbidden"
            or output != required_output
            or receipt_path != required_receipt
        ):
            raise ValueError("forced-submission call authorization mismatch")
        if os.path.lexists(required_output) or os.path.lexists(required_receipt):
            raise FileExistsError(
                "authorized output or receipt already exists; refusing another call"
            )
        profile, runtime_hashes = _load_frozen_runtime(
            common,
            supplied_profile_path=args.sandbox_profile,
            supplied_profile_sha256=profile_sha,
        )
        claim_bytes = _attempt_claim_bytes(
            authorization=authorization,
            freeze_sha256=freeze_sha,
            runtime_hashes=runtime_hashes,
            evidence_hashes=evidence_hashes,
            tool_definition_sha256=tool_sha,
        )
        atomic_create_file(required_claim, claim_bytes)
        receipt_bindings = {
            "attempt_claim_file_sha256": sha256_bytes(claim_bytes),
            "tool_definition_file_sha256": tool_sha,
            **evidence_hashes,
            **runtime_hashes,
            "retry_performed": False,
            "fallback_performed": False,
            "repair_performed": False,
        }
    elif is_v5:
        profile = load_verified_authoring_sandbox_profile(
            args.sandbox_profile.resolve(strict=True),
            expected_file_sha256=profile_sha,
        )
        call_authorization = load_verified_authoring_call_authorization(
            args.freeze_lock,
            expected_freeze_lock_file_sha256=freeze_sha,
            repository_root=ROOT,
            invocation_input=invocation_input,
            sandbox_profile=profile,
            variant=args.variant,
            run_id=args.run_id,
            output_directory=output,
            receipt_path=receipt_path,
        )
    else:
        profile = load_verified_authoring_sandbox_profile(
            args.sandbox_profile.resolve(strict=True),
            expected_file_sha256=profile_sha,
        )
    try:
        gateway = FormalContainerAuthoringGateway(profile)
        if call_authorization is None:
            invocation = invoke_llm_static(
                invocation_input,
                gateway,
                output,
            )
        else:
            invocation = invoke_llm_static(
                invocation_input,
                gateway,
                output,
                call_authorization=call_authorization,
            )
            consumed_bindings = _v5_claim_receipt_bindings(
                call_authorization,
                freeze_sha256=freeze_sha,
                run_id=args.run_id,
            )
            if consumed_bindings is None:
                raise RuntimeError(
                    "v5 formal invocation returned without its create-only claim"
                )
            receipt_bindings = consumed_bindings
    except Exception as error:
        if call_authorization is not None and not receipt_bindings:
            consumed_bindings = _v5_claim_receipt_bindings(
                call_authorization,
                freeze_sha256=freeze_sha,
                run_id=args.run_id,
            )
            if consumed_bindings is None:
                raise
            receipt_bindings = consumed_bindings
        artifact_hashes = _call_artifact_hashes(output) if output.is_dir() else {}
        response_facts: dict[str, object] = {}
        response_completed = "authoring-response.json" in artifact_hashes
        isolated_job_failed = "job-failure.json" in artifact_hashes
        if response_completed:
            try:
                response_facts = _persisted_response_facts(
                    output,
                    invocation_input,
                )
            except (TypeError, ValueError) as fact_error:
                response_facts = {
                    "provider_fact_extraction_error_type": type(fact_error).__name__,
                }
        failure_payload = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": (
                "provider_call_completed_draft_rejected"
                if response_completed
                else (
                    "isolated_job_failed_without_provider_receipt"
                    if isolated_job_failed
                    else "authorization_consumed_provider_outcome_unknown"
                )
            ),
            "authorization_consumed": bool(receipt_bindings),
            "provider_outcome": "completed" if response_completed else "unknown",
            "formal_invocation_eligible": False,
            "draft_formal_eligible": False,
            "variant": args.variant,
            "freeze_lock_file_sha256": freeze_sha,
            "sandbox_profile_file_sha256": profile.file_sha256,
            "artifact_file_sha256s": artifact_hashes,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **receipt_bindings,
            **response_facts,
        }
        atomic_create_file(
            receipt_path,
            canonical_json_bytes(
                {
                    **failure_payload,
                    "receipt_sha256": sha256_bytes(
                        canonical_json_bytes(failure_payload)
                    ),
                }
            ),
        )
        raise
    response = invocation.response
    submission_tool_name = _submission_tool_name(
        invocation_input.value.decoding.response_format
    )
    receipt_payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "awaiting_human_review_and_authority_runtime_compile",
        "authorization_consumed": bool(receipt_bindings),
        "formal_invocation_eligible": True,
        "formal_provider_call_eligible": True,
        "draft_formal_eligible": True,
        "variant": args.variant,
        "freeze_lock_file_sha256": freeze_sha,
        "authoring_input_sha256": invocation.authoring_input_sha256,
        "artifact_file_sha256s": _call_artifact_hashes(output),
        "request_sha256": invocation.request.request_sha256,
        "request_file_sha256": invocation.request_file_sha256,
        "response_file_sha256": invocation.response_file_sha256,
        "pre_review_draft_file_sha256": invocation.pre_review_draft_file_sha256,
        "isolation_attestation_sha256": (
            invocation.isolation_attestation.attestation_sha256
        ),
        "isolation_attestation_file_sha256": (
            invocation.isolation_attestation_file_sha256
        ),
        "sandbox_profile_file_sha256": profile.file_sha256,
        "provider": response.provider,
        "endpoint": response.endpoint,
        "requested_model": response.requested_model,
        "response_model": response.response_model,
        "provider_request_id_sha256": sha256_bytes(response.request_id.encode("utf-8")),
        "finish_reason": response.finish_reason,
        "response_format": invocation_input.value.decoding.response_format,
        "submission_arguments_sha256": (
            sha256_bytes(response.tool_calls[0].arguments_json.encode("utf-8"))
            if len(response.tool_calls) == 1
            and response.tool_calls[0].name == submission_tool_name
            else None
        ),
        "tool_call_count": len(response.tool_calls),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost_microusd": invocation.cost_microusd,
        "provider_latency_ms": response.latency_ms,
        "container_elapsed_ms": invocation.isolation_attestation.elapsed_ms,
        **receipt_bindings,
    }
    if is_v5:
        receipt_payload["response_text_sha256"] = sha256_bytes(
            response.text.encode("utf-8")
        )
    receipt_bytes = canonical_json_bytes(
        {
            **receipt_payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
        }
    )
    atomic_create_file(receipt_path, receipt_bytes)
    print(
        json.dumps(
            {
                "cost_microusd": invocation.cost_microusd,
                "formal_invocation_eligible": True,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "receipt": str(receipt_path),
                "receipt_file_sha256": sha256_bytes(receipt_bytes),
                "run_id": args.run_id,
                "status": "awaiting_human_review_and_authority_runtime_compile",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
