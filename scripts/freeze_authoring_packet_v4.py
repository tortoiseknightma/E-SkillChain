"""Build the approved v4 forced-submission authoring freeze deterministically."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from skillchain import config
from skillchain.static_authoring import (
    FORCED_SUBMISSION_RESPONSE_FORMAT,
    AuthoringBudgets,
    FixedDecoding,
    ModelIdentity,
    PromptIdentity,
    authoring_submission_tool_definition,
    build_authoring_packet,
    load_verified_price_schedule,
)
from skillchain.task_spec import load_default_task_specification
from skillchain.taxonomy import load_default_taxonomy_registry
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.registry import build_mvp_registry_spec
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
AUTHORING_ROOT = ROOT / "specs" / "authoring"
PROMPT_PATH = AUTHORING_ROOT / "static-author-v4.txt"
DEVIATION_PATH = AUTHORING_ROOT / "authoring-structured-submission-deviation-v1.json"
TOOL_DEFINITION_PATH = AUTHORING_ROOT / "authoring-submission-tool-v1.json"
FREEZE_PATH = AUTHORING_ROOT / "authoring-freeze-lock-v4.json"
LOCK_MANIFEST_PATH = (
    ROOT / "deploy" / "authoring" / "locks-v4" / "lock-manifest.json"
)
SANDBOX_PROFILE_PATH = LOCK_MANIFEST_PATH.parent / "sandbox-profile.json"
NETWORK_POLICY_PATH = LOCK_MANIFEST_PATH.parent / "network-policy.json"
DEPLOYMENT_RECEIPT_PATH = LOCK_MANIFEST_PATH.parent / "deployment-receipt.json"
ACCESS_PATH = AUTHORING_ROOT / "model-access-evidence-v2.json"
UNKNOWN_OUTCOME_INCIDENT = (
    AUTHORING_ROOT / "incidents" / "llm-static-primary-20260723-v1.json"
)
PRIOR_INVOCATION_RECEIPT = (
    AUTHORING_ROOT
    / "llm-static-primary-20260724-replacement-v1-invocation-receipt.json"
)
PRIOR_ADJUDICATION_RECEIPT = (
    AUTHORING_ROOT
    / "llm-static-primary-20260724-replacement-v1-adjudication-receipt.json"
)
PRICING_SOURCE_URL = "https://help.aliyun.com/zh/model-studio/model-pricing"
FROZEN_ON = "2026-07-24"
CONVERSION_MICROUSD_PER_CNY = 150_000
AUTHORIZED_RUN_ID = "llm-static-primary-20260724-structured-v1"


VARIANTS = {
    "primary": {
        "model": config.LEGACY_AUTHOR_MODEL,
        "revision": config.LEGACY_AUTHOR_MODEL_REVISION,
        "source_input": 150_000,
        "source_output": 1_500_000,
    },
}


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _converted(source_microunits: int) -> int:
    return (
        source_microunits * CONVERSION_MICROUSD_PER_CNY + 999_999
    ) // 1_000_000


def _deviation_bytes() -> bytes:
    payload = {
        "schema_version": 1,
        "deviation_id": "authoring-structured-submission-2026-07-24-v1",
        "status": "approved_for_one_prospective_interface_repair_attempt",
        "approved_by": "project-owner",
        "approved_on": FROZEN_ON,
        "approval_basis": (
            "Project owner instructed the implementation to solve the identified "
            "schema-output problem using the recommended constrained transport."
        ),
        "classification": (
            "interface-contract repair after a preserved negative result; "
            "not result-directed model selection"
        ),
        "trigger": {
            "run_id": "llm-static-primary-20260724-replacement-v1",
            "outcome": "provider_call_completed_draft_rejected",
            "formal_provider_call_eligible": True,
            "draft_formal_eligible": False,
            "schema_error_count": 40,
            "invocation_receipt_file": PRIOR_INVOCATION_RECEIPT.relative_to(
                ROOT
            ).as_posix(),
            "invocation_receipt_file_sha256": _sha(PRIOR_INVOCATION_RECEIPT),
            "adjudication_receipt_file": PRIOR_ADJUDICATION_RECEIPT.relative_to(
                ROOT
            ).as_posix(),
            "adjudication_receipt_file_sha256": _sha(
                PRIOR_ADJUDICATION_RECEIPT
            ),
            "unknown_outcome_incident_file": UNKNOWN_OUTCOME_INCIDENT.relative_to(
                ROOT
            ).as_posix(),
            "unknown_outcome_incident_file_sha256": _sha(
                UNKNOWN_OUTCOME_INCIDENT
            ),
            "unknown_provider_generation_or_billing_outcomes": 1,
        },
        "provider_capability_basis": {
            "retrieved_on": FROZEN_ON,
            "structured_output_url": (
                "https://help.aliyun.com/zh/model-studio/qwen-structured-output"
            ),
            "function_calling_url": (
                "https://help.aliyun.com/zh/model-studio/qwen-function-calling"
            ),
            "findings": [
                (
                    "DashScope JSON mode exposes json_object rather than a "
                    "response_format JSON-Schema contract."
                ),
                (
                    "DashScope function calling accepts JSON-Schema parameters "
                    "and supports a forced named tool choice for Qwen3-VL-Flash."
                ),
                (
                    "Provider framing does not replace strict downstream schema "
                    "validation."
                ),
            ],
        },
        "authorized_changes": [
            "replace json_object response framing with one forced submission function",
            "supply the exact hash-free AuthoringDraftPayload schema as parameters",
            "move canonicalization of provider JSON arguments into the trusted runner",
            "update the prompt and rebuild a separately versioned formal runtime",
            "send enable_thinking=false explicitly on the Qwen provider request",
            "disable parallel tool calls explicitly",
        ],
        "frozen_unchanged": [
            "taxonomy",
            "task specification",
            "tool registry specification",
            "public_sources empty allowlist",
            "reference skill bundle absence",
            "primary model and immutable revision",
            "price schedule and cost ceiling",
            "temperature, top_p, seed, token ceilings, and human-review ceiling",
        ],
        "forbidden_actions": [
            "modify or reinterpret the prior rejected response",
            "repair missing prior fields from Task Specification",
            "retry the prior run ID",
            "automatic or post-outcome Plus fallback",
            "retry or repair the v4 provider response",
            "execute the submission function as a project tool",
        ],
        "call_budget": {
            "known_auditable_formal_provider_responses_counted": 1,
            "unknown_provider_outcome_incidents": 1,
            "max_additional_provider_attempts": 1,
            "authorized_variant": "primary",
            "authorized_model": config.LEGACY_AUTHOR_MODEL,
            "required_run_id": AUTHORIZED_RUN_ID,
            "consume_on": "atomic-create-claim-before-container-launch",
            "reissue_policy": "new-owner-approved-deviation-only",
            "retry_fallback_repair_policy": "forbidden",
        },
        "acceptance": {
            "assistant_text_must_be_empty": True,
            "finish_reason": "tool_calls",
            "submission_count": 1,
            "submission_name": "submit_authoring_payload",
            "parallel_tool_calls": False,
            "arguments_must_pass_strict_json_and_AuthoringDraftPayload": True,
            "runner_must_validate_all_rules_and_tool_permissions": True,
            "human_review_required_before_bank_compile": True,
        },
    }
    return canonical_json_bytes(
        {
            **payload,
            "deviation_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
    )


def _expected_files() -> dict[Path, bytes]:
    prompt_bytes = PROMPT_PATH.read_bytes()
    if not prompt_bytes.endswith(b"\n") or prompt_bytes.endswith(b"\n\n"):
        raise ValueError("prompt source must end in exactly one LF")
    prompt_text = prompt_bytes[:-1].decode("utf-8", errors="strict")
    prompt = PromptIdentity(
        prompt_id="static-author-v4",
        prompt_version="4.0.0",
        template=prompt_text,
        prompt_sha256=sha256_bytes(prompt_text.encode("utf-8")),
    )
    taxonomy = load_default_taxonomy_registry()
    tasks = load_default_task_specification()
    registry = build_mvp_registry_spec()
    decoding = FixedDecoding(
        seed=20260722,
        max_output_tokens=8_000,
        response_format=FORCED_SUBMISSION_RESPONSE_FORMAT,
    )
    budgets = AuthoringBudgets(
        max_input_tokens=32_000,
        max_output_tokens=8_000,
        max_total_tokens=40_000,
        max_cost_microusd=50_000,
        max_human_review_minutes=30,
    )
    tool_definition_bytes = canonical_json_bytes(
        authoring_submission_tool_definition()
    )
    deviation_bytes = _deviation_bytes()
    generated: dict[Path, bytes] = {
        TOOL_DEFINITION_PATH: tool_definition_bytes,
        DEVIATION_PATH: deviation_bytes,
    }
    variant_locks: dict[str, object] = {}
    for name, settings in sorted(VARIANTS.items()):
        model = str(settings["model"])
        schedule_path = AUTHORING_ROOT / f"price-{model}-v1.json"
        schedule = load_verified_price_schedule(
            schedule_path,
            expected_file_sha256=_sha(schedule_path),
        )
        if (
            schedule.value.model != model
            or schedule.value.input_microusd_per_million_tokens
            != _converted(int(settings["source_input"]))
            or schedule.value.output_microusd_per_million_tokens
            != _converted(int(settings["source_output"]))
        ):
            raise ValueError(f"frozen price schedule drifted for {name}")
        packet = build_authoring_packet(
            taxonomy=taxonomy,
            task_specification=tasks,
            tool_registry=registry,
            prompt=prompt,
            model=ModelIdentity(
                provider="qwen",
                endpoint=config.PROVIDER_ENDPOINTS["qwen"],
                model=model,
                revision=str(settings["revision"]),
            ),
            decoding=decoding,
            price_schedule=schedule,
            budgets=budgets,
            public_sources=(),
            reference_skill_bundle=None,
            defer_tool_registry_runtime=True,
        )
        packet_path = AUTHORING_ROOT / f"authoring-packet-{name}-v4.json"
        packet_bytes = packet.canonical_bytes()
        generated[packet_path] = packet_bytes
        worst_cost = (
            budgets.max_input_tokens
            * schedule.value.input_microusd_per_million_tokens
            + budgets.max_output_tokens
            * schedule.value.output_microusd_per_million_tokens
            + 999_999
        ) // 1_000_000
        variant_locks[name] = {
            "input_sha256": packet.input_sha256,
            "model": model,
            "model_revision": settings["revision"],
            "packet_file": packet_path.relative_to(ROOT).as_posix(),
            "packet_file_sha256": sha256_bytes(packet_bytes),
            "price_schedule_file": schedule_path.relative_to(ROOT).as_posix(),
            "price_schedule_file_sha256": _sha(schedule_path),
            "worst_case_cost_microusd": worst_cost,
        }
    freeze = {
        "schema_version": 2,
        "status": "frozen",
        "freeze_id": "authoring-freeze-2026-07-24-v4",
        "approved_on": FROZEN_ON,
        "approved_by": "project-owner",
        "default_variant": "primary",
        "fallback_policy": (
            "unavailable-under-this-deviation; prior output exists; "
            "no automatic or post-outcome manual fallback"
        ),
        "variants": variant_locks,
        "common": {
            "budgets": budgets.model_dump(mode="json"),
            "decoding": decoding.model_dump(mode="json"),
            "exchange_rate_policy": {
                "conversion_microusd_per_cny": CONVERSION_MICROUSD_PER_CNY,
                "meaning": "conservative budget conversion; 1 CNY = 0.15 USD",
            },
            "model_access_evidence_file": ACCESS_PATH.relative_to(ROOT).as_posix(),
            "model_access_evidence_file_sha256": _sha(ACCESS_PATH),
            "pricing_source_url": PRICING_SOURCE_URL,
            "prompt_file": PROMPT_PATH.relative_to(ROOT).as_posix(),
            "prompt_file_sha256": _sha(PROMPT_PATH),
            "prompt_text_sha256": prompt.prompt_sha256,
            "prompt_transform": "utf8-strip-one-final-lf-v1",
            "public_source_ids": [],
            "reference_skill_bundle": None,
            "runtime_lock_bundle": {
                "runtime_version": "formal-v3",
                "lock_manifest_file": LOCK_MANIFEST_PATH.relative_to(
                    ROOT
                ).as_posix(),
                "lock_manifest_file_sha256": _sha(LOCK_MANIFEST_PATH),
                "sandbox_profile_file": SANDBOX_PROFILE_PATH.relative_to(
                    ROOT
                ).as_posix(),
                "sandbox_profile_file_sha256": _sha(SANDBOX_PROFILE_PATH),
                "network_policy_file": NETWORK_POLICY_PATH.relative_to(
                    ROOT
                ).as_posix(),
                "network_policy_file_sha256": _sha(NETWORK_POLICY_PATH),
                "deployment_receipt_file": DEPLOYMENT_RECEIPT_PATH.relative_to(
                    ROOT
                ).as_posix(),
                "deployment_receipt_file_sha256": _sha(
                    DEPLOYMENT_RECEIPT_PATH
                ),
            },
            "task_specification_sha256": tasks.task_spec_sha256,
            "taxonomy_sha256": taxonomy.taxonomy_sha256,
            "tool_registry_runtime_policy": "deferred_until_bank_compile",
            "tool_registry_sha256": registry.registry_sha256,
            "output_channel": {
                "mode": "forced_single_function_submission",
                "tool_name": "submit_authoring_payload",
                "tool_execution": False,
                "parallel_tool_calls": False,
                "assistant_text_policy": "forbidden",
                "arguments_policy": (
                    "strict-json-parse-then-pydantic-validate-and-runner-canonicalize"
                ),
                "tool_definition_file": TOOL_DEFINITION_PATH.relative_to(
                    ROOT
                ).as_posix(),
                "tool_definition_file_sha256": sha256_bytes(
                    tool_definition_bytes
                ),
            },
        },
        "call_authorization": {
            "authorization_id": (
                "authoring-structured-submission-primary-attempt-2026-07-24-v1"
            ),
            "protocol_deviation_file": DEVIATION_PATH.relative_to(ROOT).as_posix(),
            "protocol_deviation_file_sha256": sha256_bytes(deviation_bytes),
            "prior_adjudication_receipt_file": (
                PRIOR_ADJUDICATION_RECEIPT.relative_to(ROOT).as_posix()
            ),
            "prior_adjudication_receipt_file_sha256": _sha(
                PRIOR_ADJUDICATION_RECEIPT
            ),
            "prior_freeze_lock_file_sha256": (
                "42b4ebe49dcb4df529aed8c4e553179c8d34fbd7ee848cf8c883c1046a8234a1"
            ),
            "unknown_outcome_incident_file": UNKNOWN_OUTCOME_INCIDENT.relative_to(
                ROOT
            ).as_posix(),
            "unknown_outcome_incident_file_sha256": _sha(
                UNKNOWN_OUTCOME_INCIDENT
            ),
            "known_auditable_formal_provider_responses_counted": 1,
            "unknown_provider_outcome_incidents": 1,
            "max_additional_provider_attempts": 1,
            "authorized_variant": "primary",
            "authorized_model": config.LEGACY_AUTHOR_MODEL,
            "required_run_id": AUTHORIZED_RUN_ID,
            "required_output_directory": (
                f"runs/formal-authoring/{AUTHORIZED_RUN_ID}"
            ),
            "required_receipt_file": (
                "specs/authoring/"
                f"{AUTHORIZED_RUN_ID}-invocation-receipt.json"
            ),
            "required_claim_file": (
                "specs/authoring/"
                f"{AUTHORIZED_RUN_ID}-attempt-claim.json"
            ),
            "consume_on": "atomic-create-claim-before-container-launch",
            "reissue_policy": "new-owner-approved-deviation-only",
            "retry_fallback_repair_policy": "forbidden",
        },
    }
    generated[FREEZE_PATH] = canonical_json_bytes(freeze)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = _expected_files()
    if args.check:
        mismatches = [
            path.relative_to(ROOT).as_posix()
            for path, content in expected.items()
            if not path.is_file() or path.read_bytes() != content
        ]
        if mismatches:
            raise SystemExit(
                "authoring v4 freeze drift: " + ", ".join(mismatches)
            )
        return 0
    existing = [
        path.relative_to(ROOT).as_posix()
        for path in expected
        if os.path.lexists(path)
    ]
    if existing:
        raise SystemExit(
            "authoring v4 freeze is create-only; existing: " + ", ".join(existing)
        )
    for path, content in expected.items():
        atomic_create_file(path, content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
