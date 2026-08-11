"""Build the deterministic v5 author-content call candidate.

This script does not issue authority, approve a budget, allocate a run ID,
create an attempt claim, freeze a runtime, or contact a provider.  Its output
is suitable only for offline review followed by a separately owner-approved
protocol deviation and formal freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from skillchain import config
from skillchain.static_authoring import (
    AUTHORING_CONTENT_SCHEMA_NAME,
    PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    AuthoringBudgets,
    FixedDecoding,
    ModelIdentity,
    PromptIdentity,
    authoring_content_json_schema,
    build_authoring_content_output_contract,
    build_authoring_packet,
    load_verified_price_schedule,
)
from skillchain.synthesis.store import atomic_create_file
from skillchain.task_spec import load_mvp_task_specification_v1
from skillchain.taxonomy import load_default_taxonomy_registry
from skillchain.tools.registry import build_mvp_registry_spec
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
AUTHORING_ROOT = ROOT / "specs" / "authoring"
PROMPT_PATH = AUTHORING_ROOT / "static-author-v5.txt"
PRICE_PATH = AUTHORING_ROOT / f"price-{config.LEGACY_AUTHOR_MODEL}-v1.json"
PACKET_PATH = AUTHORING_ROOT / "authoring-packet-primary-v5-candidate.json"
CONTENT_SCHEMA_PATH = AUTHORING_ROOT / "authoring-content-schema-v2-candidate.json"
CANDIDATE_PATH = AUTHORING_ROOT / "authoring-freeze-candidate-v5.json"

# These are the already-adjudicated v4 bytes.  The candidate builder refuses
# to operate if any are rewritten, so a forward fix cannot silently mutate the
# preserved negative result or its frozen submission contract.
PRIOR_V4_LOCKS = {
    AUTHORING_ROOT / "authoring-freeze-lock-v4.json": (
        "d9f5f646554fc2f6be8929db702792daf04c540ea65323957a507fb159481462"
    ),
    AUTHORING_ROOT / "authoring-packet-primary-v4.json": (
        "76c15605862b32a26730d2cd5bf6af77f0ee1ebdcc622891387665a564eab729"
    ),
    AUTHORING_ROOT / "authoring-submission-tool-v1.json": (
        "9b262ccbab0dbc53b3af9acd0c52e64b37e433fb4674070738ad1954cb2d85f9"
    ),
    AUTHORING_ROOT
    / "llm-static-primary-20260724-structured-v1-adjudication-receipt.json": (
        "27898254058216a31ea6e54306ce4ab7ab094179baeff0f8e879adae3ac05c54"
    ),
}


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_prior_v4_immutable() -> None:
    mismatches = [
        path.relative_to(ROOT).as_posix()
        for path, expected_sha256 in PRIOR_V4_LOCKS.items()
        if _sha(path) != expected_sha256
    ]
    if mismatches:
        raise ValueError("preserved v4 artifact drift: " + ", ".join(mismatches))


def _expected_files() -> dict[Path, bytes]:
    _assert_prior_v4_immutable()
    prompt_bytes = PROMPT_PATH.read_bytes()
    if not prompt_bytes.endswith(b"\n") or prompt_bytes.endswith(b"\n\n"):
        raise ValueError("prompt source must end in exactly one LF")
    prompt_text = prompt_bytes[:-1].decode("utf-8", errors="strict")
    prompt = PromptIdentity(
        prompt_id="static-author-v5-candidate",
        prompt_version="5.0.0-candidate",
        template=prompt_text,
        prompt_sha256=sha256_bytes(prompt_text.encode("utf-8")),
    )
    taxonomy = load_default_taxonomy_registry()
    tasks = load_mvp_task_specification_v1()
    registry = build_mvp_registry_spec(include_multi_product=True)
    decoding = FixedDecoding(
        seed=20260722,
        max_output_tokens=6_000,
        response_format=PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    )
    proposed_budgets = AuthoringBudgets(
        max_input_tokens=30_000,
        max_output_tokens=6_000,
        max_total_tokens=36_000,
        max_cost_microusd=3_000,
        max_human_review_minutes=30,
    )
    price_sha256 = _sha(PRICE_PATH)
    schedule = load_verified_price_schedule(
        PRICE_PATH,
        expected_file_sha256=price_sha256,
    )
    packet = build_authoring_packet(
        taxonomy=taxonomy,
        task_specification=tasks,
        tool_registry=registry,
        prompt=prompt,
        model=ModelIdentity(
            provider="qwen",
            endpoint=config.PROVIDER_ENDPOINTS["qwen"],
            model=config.LEGACY_AUTHOR_MODEL,
            revision=config.LEGACY_AUTHOR_MODEL_REVISION,
        ),
        decoding=decoding,
        price_schedule=schedule,
        budgets=proposed_budgets,
        public_sources=(),
        reference_skill_bundle=None,
        defer_tool_registry_runtime=True,
    )
    packet_bytes = packet.canonical_bytes()
    content_schema_bytes = canonical_json_bytes(authoring_content_json_schema(packet))
    output_contract = build_authoring_content_output_contract(packet)
    worst_case_cost_microusd = (
        proposed_budgets.max_input_tokens
        * schedule.value.input_microusd_per_million_tokens
        + proposed_budgets.max_output_tokens
        * schedule.value.output_microusd_per_million_tokens
        + 999_999
    ) // 1_000_000
    candidate = {
        "schema_version": 1,
        "artifact_kind": "authoring_freeze_candidate",
        "status": "candidate_not_authorized",
        "candidate_id": "authoring-contract-v5-offline-candidate-2026-07-24",
        "formal_use": (
            "forbidden_until_owner_approves_a_new_protocol_deviation_budget_"
            "run_id_and_independently_frozen_runtime"
        ),
        "authorization": {
            "provider_call_authorized": False,
            "budget_authorized": False,
            "run_id": None,
            "attempt_claim_file": None,
            "authority_issued": False,
            "runtime_lock_frozen": False,
            "required_owner_actions": [
                "approve a prospective protocol deviation",
                "approve a new call budget",
                "allocate a new run ID and create-only output paths",
                "freeze and verify a new independent runtime lock",
            ],
        },
        "preserved_prior_result": {
            "status": "provider_call_completed_draft_rejected",
            "adjudication_receipt_file": (
                "specs/authoring/"
                "llm-static-primary-20260724-structured-v1-"
                "adjudication-receipt.json"
            ),
            "adjudication_receipt_file_sha256": PRIOR_V4_LOCKS[
                AUTHORING_ROOT / "llm-static-primary-20260724-structured-v1-"
                "adjudication-receipt.json"
            ],
            "prior_run_id_reusable": False,
            "prior_claim_reusable": False,
            "prior_budget_authorization_reusable": False,
        },
        "proposal": {
            "variant": "primary",
            "model": config.LEGACY_AUTHOR_MODEL,
            "model_revision": config.LEGACY_AUTHOR_MODEL_REVISION,
            "packet_file": PACKET_PATH.relative_to(ROOT).as_posix(),
            "packet_file_sha256": sha256_bytes(packet_bytes),
            "authoring_input_sha256": packet.input_sha256,
            "prompt_file": PROMPT_PATH.relative_to(ROOT).as_posix(),
            "prompt_file_sha256": _sha(PROMPT_PATH),
            "prompt_text_sha256": prompt.prompt_sha256,
            "prompt_transform": "utf8-strip-one-final-lf-v1",
            "price_schedule_file": PRICE_PATH.relative_to(ROOT).as_posix(),
            "price_schedule_file_sha256": price_sha256,
            "budget_proposal_id": ("authoring-v5-budget-proposal-2026-07-24-v1"),
            "proposed_budgets_not_authorized": proposed_budgets.model_dump(mode="json"),
            "proposed_worst_case_cost_microusd_not_authorized": (
                worst_case_cost_microusd
            ),
            "decoding": decoding.model_dump(mode="json"),
            "task_specification_sha256": tasks.task_spec_sha256,
            "taxonomy_sha256": taxonomy.taxonomy_sha256,
            "tool_registry_sha256": registry.registry_sha256,
            "tool_registry_runtime_policy": "not_frozen_candidate_only",
            "public_source_ids": [],
            "reference_skill_bundle": None,
            "output_channel": {
                "mode": "provider_json_object_content_submission",
                "provider_response_format": {"type": "json_object"},
                "provider_guarantee": "json_syntax_only",
                "tools_supplied": False,
                "response_policy": "exactly_one_json_object_without_commentary",
                "json_schema_enforcement": "prompt_and_audit_only",
                "runner_packet_contract_validation": True,
                "runner_validation_engine": "pydantic_and_trusted_compiler",
                "runner_validation_policy": (
                    "strict-json-parse-then-pydantic-shape-validation-and-"
                    "packet-bound-trusted-compiler-validation-and-injection"
                ),
                "schema_name": AUTHORING_CONTENT_SCHEMA_NAME,
                "json_schema_file": (CONTENT_SCHEMA_PATH.relative_to(ROOT).as_posix()),
                "json_schema_file_sha256": sha256_bytes(content_schema_bytes),
                "output_contract_sha256": output_contract.contract_sha256,
            },
        },
    }
    return {
        PACKET_PATH: packet_bytes,
        CONTENT_SCHEMA_PATH: content_schema_bytes,
        CANDIDATE_PATH: canonical_json_bytes(candidate),
    }


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
            raise SystemExit("authoring v5 candidate drift: " + ", ".join(mismatches))
    else:
        existing = [
            path.relative_to(ROOT).as_posix()
            for path in expected
            if os.path.lexists(path)
        ]
        if existing:
            raise SystemExit(
                "authoring v5 candidate is create-only; existing: "
                + ", ".join(existing)
            )
        for path, content in expected.items():
            atomic_create_file(path, content)
    print(
        json.dumps(
            {
                "status": "candidate_not_authorized",
                "provider_call_authorized": False,
                "authority_issued": False,
                "candidate": str(CANDIDATE_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
