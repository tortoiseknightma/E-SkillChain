from __future__ import annotations

import json
import re

import pytest

import scripts.build_codex_authoring_approval_package_v5 as builder_v5
import scripts.run_codex_authoring_v5 as runner_v5
from skillchain.codex_authoring_v5 import (
    CodexAuthoringContractError,
    V5_PROMPT_ID,
    V5_PROMPT_VERSION,
    normalize_codex_authoring_output,
    parse_codex_authoring_request,
)
from skillchain.static_authoring import load_authoring_packet
from skillchain.tools.serialization import sha256_bytes


def test_v5_candidate_changes_only_the_visible_prompt_contract() -> None:
    expected = builder_v5._expected_files()
    packet = json.loads(expected[builder_v5.PACKET_PATH])
    v4_packet = json.loads(
        builder_v5.v4_builder.PACKET_PATH.read_text(encoding="utf-8")
    )

    assert packet["prompt"]["prompt_id"] == V5_PROMPT_ID
    assert packet["prompt"]["prompt_version"] == V5_PROMPT_VERSION
    assert "predicted class name" in packet["prompt"]["template"]
    assert packet["prompt"] != v4_packet["prompt"]
    for field in (
        "taxonomy",
        "task_specification",
        "tool_registry",
        "compiler",
        "model",
        "session_budget",
        "public_sources",
        "reference_skill_bundle",
    ):
        assert packet[field] == v4_packet[field]

    freeze = json.loads(expected[builder_v5.FREEZE_PATH])
    receipt_binding = freeze["bindings"]["v4_canonical_receipt"]
    receipt_bytes = builder_v5.V4_RECEIPT_PATH.read_bytes()
    assert receipt_binding["file_sha256"] == sha256_bytes(receipt_bytes)
    assert freeze["prior_attempt_history"]["v4_result_rewritten_or_reclassified"] is False


def test_v4_rejection_is_explained_by_the_v5_visible_wording_rule() -> None:
    expected = builder_v5._expected_files()
    packet = parse_codex_authoring_request(
        expected[builder_v5.REQUEST_PATH]
    ).authoring_input
    semantic_source = load_authoring_packet(
        builder_v5.SEMANTIC_SOURCE_PATH,
        expected_file_sha256=builder_v5.SEMANTIC_SOURCE_SHA256,
    )
    raw = builder_v5.V4_RAW_OUTPUT_PATH.read_bytes()

    with pytest.raises(CodexAuthoringContractError):
        normalize_codex_authoring_output(
            raw,
            authoring_input=packet,
            semantic_source=semantic_source,
        )

    rewritten = re.sub(
        rb"\blabels?\b",
        b"predicted class name",
        raw,
        flags=re.IGNORECASE,
    )
    compiled = normalize_codex_authoring_output(
        rewritten,
        authoring_input=packet,
        semantic_source=semantic_source,
    )
    assert len(compiled.drafts) == 6


def test_published_v5_bundle_independently_replays_as_review_ready() -> None:
    guard_bytes = builder_v5.TERMINAL_GUARD_PATH.read_bytes()
    receipt = json.loads(builder_v5.RECEIPT_PATH.read_bytes())

    assert receipt["status"] == "codex_session_completed_draft_ready_for_review"
    assert receipt["formal_codex_session_eligible"] is True
    assert receipt["authorization_consumed"] is True
    assert receipt["repository_retry_performed"] is False
    assert receipt["followup_performed"] is False
    assert receipt["repair_performed"] is False
    assert receipt["fallback_performed"] is False
    assert runner_v5.validate_canonical_bundle_from_disk(
        builder_v5.OUTPUT_DIR,
        claim_path=builder_v5.CLAIM_PATH,
        guard_path=builder_v5.TERMINAL_GUARD_PATH,
        guard_file_sha256=sha256_bytes(guard_bytes),
    )
