from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import scripts.record_codex_authoring_review as review_script
from skillchain.codex_authoring_review import CodexAuthoringHumanReviewReceipt
from skillchain.static_authoring import ReviewChecklist
from skillchain.tools.serialization import (
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


def _source_hashes() -> tuple[str, str]:
    receipt = read_stable_regular_file(
        review_script.RUN_ROOT / "invocation-receipt.json",
        label="test Codex invocation receipt",
    )
    draft = read_stable_regular_file(
        review_script.RUN_ROOT / "pre-review-draft.json",
        label="test Codex pre-review draft",
    )
    return sha256_bytes(receipt), sha256_bytes(draft)


def _record(output: Path, **overrides):
    receipt_sha256, draft_sha256 = _source_hashes()
    arguments = {
        "expected_invocation_receipt_file_sha256": receipt_sha256,
        "expected_pre_review_file_sha256": draft_sha256,
        "reviewer_id": "wenxi_0726",
        "review_minutes": 5,
        "change_reason": "未修改，接受原稿",
        "checklist": ReviewChecklist(
            no_private_inputs=True,
            safety_checked=True,
            schema_valid=True,
            source_citations_checked=True,
            tool_permissions_checked=True,
            edit_scope_checked=True,
        ),
        "output_path": output,
    }
    arguments.update(overrides)
    return review_script.record_codex_authoring_review(**arguments)


def test_record_codex_review_is_external_create_only_and_replay_safe(
    tmp_path, monkeypatch
):
    replay_calls = []

    def validated_replay(*args, **kwargs):
        replay_calls.append((args, kwargs))
        return True

    monkeypatch.setattr(
        review_script.v5_runner,
        "validate_canonical_bundle_from_disk",
        validated_replay,
    )
    original_run_files = {path.name for path in review_script.RUN_ROOT.iterdir()}
    output = tmp_path / "human-review.json"
    receipt, created = _record(output)

    assert created is True
    assert receipt.decision == "accepted_unchanged"
    assert receipt.human_review.reviewer_id == "wenxi_0726"
    assert receipt.human_review.review_minutes == 5
    assert receipt.human_review.change_reason == "未修改，接受原稿"
    assert receipt.human_review.changed is False
    assert all(receipt.human_review.checklist.model_dump(mode="json").values())
    raw = parse_canonical_json(
        read_stable_regular_file(output, label="test human-review receipt"),
        label="test human-review receipt",
    )
    assert CodexAuthoringHumanReviewReceipt.model_validate(raw, strict=True) == receipt
    assert replay_calls
    assert {
        path.name for path in review_script.RUN_ROOT.iterdir()
    } == original_run_files

    repeated, repeated_created = _record(output)
    assert repeated == receipt
    assert repeated_created is False

    with pytest.raises(FileExistsError, match="different bytes"):
        _record(output, reviewer_id="another-human")


def test_record_codex_review_rejects_hash_and_budget_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_script.v5_runner,
        "validate_canonical_bundle_from_disk",
        lambda *args, **kwargs: True,
    )
    with pytest.raises(ValueError, match="pre-review draft SHA-256"):
        _record(
            tmp_path / "bad-sha.json",
            expected_pre_review_file_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="budget or input binding"):
        _record(tmp_path / "over-budget.json", review_minutes=31)


@pytest.mark.parametrize("use_parent_alias", [False, True])
def test_record_codex_review_rejects_relative_run_output_alias(
    tmp_path, monkeypatch, use_parent_alias
):
    monkeypatch.chdir(review_script.ROOT)
    run_relative = review_script.RUN_ROOT.relative_to(review_script.ROOT)
    filename = f"{tmp_path.name}-forbidden-review.json"
    if use_parent_alias:
        output = run_relative / "unused-alias" / ".." / filename
    else:
        output = run_relative / filename

    with pytest.raises(ValueError, match="outside the canonical run"):
        _record(output)
    assert not (review_script.RUN_ROOT / filename).exists()


def test_review_receipt_model_rejects_nonstrict_type_coercion(tmp_path, monkeypatch):
    monkeypatch.setattr(
        review_script.v5_runner,
        "validate_canonical_bundle_from_disk",
        lambda *args, **kwargs: True,
    )
    receipt, _ = _record(tmp_path / "strict-review.json")
    raw = receipt.model_dump(mode="json")
    raw["max_human_review_minutes"] = str(raw["max_human_review_minutes"])

    with pytest.raises(ValidationError):
        CodexAuthoringHumanReviewReceipt.model_validate(raw)
