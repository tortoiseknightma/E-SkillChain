from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from skillchain import config
from skillchain.static_authoring import (
    FORCED_SUBMISSION_RESPONSE_FORMAT,
    PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    authoring_content_json_schema,
    authoring_submission_tool_definition,
    build_canonical_authoring_request,
    load_authoring_packet,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]


def _script_env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
    }


def test_authoring_freeze_artifacts_are_reproducible_and_role_locked() -> None:
    subprocess.run(
        [sys.executable, "scripts/freeze_authoring_packet.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    lock_path = ROOT / "specs/authoring/authoring-freeze-lock-v3.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == "frozen"
    assert lock["default_variant"] == "primary"
    assert lock["fallback_policy"].startswith("manual-selection-before-run-only")
    assert lock["common"]["public_source_ids"] == []
    assert lock["common"]["reference_skill_bundle"] is None
    assert lock["common"]["budgets"] == {
        "max_cost_microusd": 50_000,
        "max_human_review_minutes": 30,
        "max_input_tokens": 32_000,
        "max_output_tokens": 8_000,
        "max_successful_calls": 1,
        "max_total_tokens": 40_000,
    }
    expected_models = {
        "primary": config.LEGACY_AUTHOR_MODEL,
        "fallback": config.LEGACY_AUTHOR_FALLBACK_MODEL,
    }
    for name, expected_model in expected_models.items():
        variant = lock["variants"][name]
        packet_path = ROOT / variant["packet_file"]
        packet_bytes = packet_path.read_bytes()
        assert sha256_bytes(packet_bytes) == variant["packet_file_sha256"]
        packet = load_authoring_packet(
            packet_path,
            expected_file_sha256=variant["packet_file_sha256"],
        )
        assert packet.model.model == expected_model
        assert packet.external_verification == "externally_verified"
        assert packet.public_sources == ()
        assert packet.reference_skill_bundle is None
        assert packet.tool_registry_runtime_sha256 is None
        assert (
            packet.price_schedule_file_sha256 == (variant["price_schedule_file_sha256"])
        )


def test_v4_forced_submission_freeze_is_reproducible_and_single_attempt() -> None:
    subprocess.run(
        [sys.executable, "scripts/freeze_authoring_packet_v4.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    lock_path = ROOT / "specs/authoring/authoring-freeze-lock-v4.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == "frozen"
    assert set(lock["variants"]) == {"primary"}
    assert lock["fallback_policy"].startswith("unavailable-under-this-deviation")
    assert lock["common"]["decoding"]["response_format"] == (
        FORCED_SUBMISSION_RESPONSE_FORMAT
    )
    output = lock["common"]["output_channel"]
    tool_path = ROOT / output["tool_definition_file"]
    tool_bytes = tool_path.read_bytes()
    assert sha256_bytes(tool_bytes) == output["tool_definition_file_sha256"]
    assert tool_bytes == canonical_json_bytes(authoring_submission_tool_definition())
    legacy_tool_names = (
        tool_bytes
        and json.loads(tool_bytes)["function"]["parameters"]["properties"]["drafts"][
            "items"
        ]["properties"]["steps"]["items"]["properties"]["tool_name"]["enum"]
    )
    assert legacy_tool_names == [
        "image_product_search",
        "text_product_search",
        "style_similar_search",
        "encyclopedia_lookup",
        "recipe_lookup",
        "object_detect",
        "document_ocr",
    ]
    assert output["parallel_tool_calls"] is False
    assert output["tool_execution"] is False

    runtime = lock["common"]["runtime_lock_bundle"]
    manifest = json.loads(
        (ROOT / runtime["lock_manifest_file"]).read_text(encoding="utf-8")
    )
    assert runtime["runtime_version"] == "formal-v3"
    assert (
        manifest["sandbox_profile_file_sha256"]
        == runtime["sandbox_profile_file_sha256"]
    )
    assert (
        manifest["network_policy_file_sha256"] == runtime["network_policy_file_sha256"]
    )
    assert (
        manifest["deployment_receipt_file_sha256"]
        == runtime["deployment_receipt_file_sha256"]
    )

    authorization = lock["call_authorization"]
    assert authorization["max_additional_provider_attempts"] == 1
    assert authorization["known_auditable_formal_provider_responses_counted"] == 1
    assert authorization["unknown_provider_outcome_incidents"] == 1
    assert authorization["consume_on"] == (
        "atomic-create-claim-before-container-launch"
    )
    assert authorization["reissue_policy"] == ("new-owner-approved-deviation-only")
    assert authorization["retry_fallback_repair_policy"] == "forbidden"
    assert authorization["required_claim_file"].endswith("-attempt-claim.json")

    variant = lock["variants"]["primary"]
    packet_path = ROOT / variant["packet_file"]
    packet = load_authoring_packet(
        packet_path,
        expected_file_sha256=variant["packet_file_sha256"],
    )
    assert packet.model.model == config.LEGACY_AUTHOR_MODEL
    assert packet.decoding.response_format == FORCED_SUBMISSION_RESPONSE_FORMAT
    assert packet.public_sources == ()
    assert packet.reference_skill_bundle is None


def test_v5_candidate_is_reproducible_and_cannot_authorize_a_call(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_authoring_packet_v5_candidate.py",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "candidate_not_authorized"
    assert report["provider_call_authorized"] is False
    assert report["authority_issued"] is False

    candidate_path = ROOT / "specs/authoring/authoring-freeze-candidate-v5.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["status"] == "candidate_not_authorized"
    assert candidate["formal_use"].startswith("forbidden_until_owner_approves")
    assert candidate["authorization"] == {
        "attempt_claim_file": None,
        "authority_issued": False,
        "budget_authorized": False,
        "provider_call_authorized": False,
        "required_owner_actions": [
            "approve a prospective protocol deviation",
            "approve a new call budget",
            "allocate a new run ID and create-only output paths",
            "freeze and verify a new independent runtime lock",
        ],
        "run_id": None,
        "runtime_lock_frozen": False,
    }
    assert candidate["preserved_prior_result"]["prior_run_id_reusable"] is False
    assert candidate["preserved_prior_result"]["prior_claim_reusable"] is False
    assert (
        candidate["preserved_prior_result"]["prior_budget_authorization_reusable"]
        is False
    )

    proposal = candidate["proposal"]
    v4_lock = json.loads(
        (ROOT / "specs/authoring/authoring-freeze-lock-v4.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        proposal["budget_proposal_id"] == "authoring-v5-budget-proposal-2026-07-24-v1"
    )
    assert proposal["proposed_budgets_not_authorized"] != v4_lock["common"]["budgets"]
    packet_path = ROOT / proposal["packet_file"]
    packet = load_authoring_packet(
        packet_path,
        expected_file_sha256=proposal["packet_file_sha256"],
    )
    request = build_canonical_authoring_request(packet)
    request_bytes = request.canonical_bytes()
    assert (
        request.request_sha256
        == "064a317323384c210300766baa2429908e0ca04d7bf67c7798e05e1e8f3e327e"
    )
    assert (
        sha256_bytes(request_bytes)
        == "29f923c20fca190fd6e2a5e8d928a44845e581c14db55c0542bfbe421ba90d14"
    )
    assert len(request_bytes) == 80_961
    assert packet.decoding.response_format == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
    assert packet.task_specification.version == "ecommerce-task-spec-v1"
    assert packet.tool_registry.version == "mvp-tool-registry-v2"
    assert packet.tool_registry_runtime_sha256 is None

    output = proposal["output_channel"]
    assert output["mode"] == "provider_json_object_content_submission"
    assert output["provider_response_format"] == {"type": "json_object"}
    assert output["provider_guarantee"] == "json_syntax_only"
    assert output["tools_supplied"] is False
    assert output["json_schema_enforcement"] == "prompt_and_audit_only"
    assert output["runner_packet_contract_validation"] is True
    assert output["runner_validation_engine"] == "pydantic_and_trusted_compiler"
    assert "tool_name" not in output
    assert "parallel_tool_calls" not in output
    schema_path = ROOT / output["json_schema_file"]
    schema_bytes = schema_path.read_bytes()
    assert sha256_bytes(schema_bytes) == output["json_schema_file_sha256"]
    assert schema_bytes == canonical_json_bytes(authoring_content_json_schema(packet))
    schema = json.loads(schema_bytes)
    assert schema["title"] == "skillchain_authoring_content_v2"
    multi_product = next(
        item
        for item in schema["properties"]["drafts"]["items"]["oneOf"]
        if item["properties"]["capability_id"].get("const") == "product.multi_search"
    )
    assert multi_product["properties"]["steps"]["items"]["properties"]["tool_name"][
        "enum"
    ] == ["multi_product_search"]

    blocked = subprocess.run(
        [
            sys.executable,
            "scripts/run_formal_authoring.py",
            "--freeze-lock",
            str(candidate_path),
            "--freeze-lock-sha256",
            sha256_bytes(candidate_path.read_bytes()),
            "--variant",
            "primary",
            "--sandbox-profile",
            str(tmp_path / "never-read-sandbox-profile.json"),
            "--sandbox-profile-sha256",
            "0" * 64,
            "--run-id",
            "v5-candidate-must-not-run",
            "--output",
            str(tmp_path / "must-not-exist"),
            "--receipt",
            str(tmp_path / "must-not-exist-receipt.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert blocked.returncode != 0
    assert "candidate is not authorized for provider invocation" in blocked.stderr
    assert not (tmp_path / "must-not-exist").exists()
    assert not (tmp_path / "must-not-exist-receipt.json").exists()

    subprocess.run(
        [sys.executable, "scripts/freeze_authoring_packet_v4.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
