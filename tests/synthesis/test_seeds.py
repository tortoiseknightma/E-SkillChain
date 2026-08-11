import hashlib
import json

import pytest

from skillchain.synthesis.batches import corpus_status
from skillchain.synthesis.seeds import (
    accept_seed_batch,
    reject_seed_batch,
    stage_seed_batch,
)


def _write_draft(path, draft):
    path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")


def test_seed_stage_requires_exact_model_confirmation(tmp_path, mechanical_seed_draft):
    mechanical_seed_draft["model_display_name"] = "GPT-5"
    draft = tmp_path / "draft.json"
    _write_draft(draft, mechanical_seed_draft)

    with pytest.raises(ValueError, match="5.6 Sol Ultra"):
        stage_seed_batch(draft, tmp_path / "queries", "seed-r1")


def test_seed_stage_requires_five_intents_and_three_unique_examples(
    tmp_path, mechanical_seed_draft
):
    del mechanical_seed_draft["examples"]["utility"]
    draft = tmp_path / "draft.json"
    _write_draft(draft, mechanical_seed_draft)

    with pytest.raises(ValueError, match="五个意图"):
        stage_seed_batch(draft, tmp_path / "queries", "seed-r1")


def test_seed_stage_rejects_drifted_query_relations_as_canonical_intents(
    tmp_path, mechanical_seed_draft
):
    drifted_intents = (
        "exact_match",
        "substitute",
        "complement",
        "attribute_query",
        "scenario_query",
    )
    mechanical_seed_draft["examples"] = {
        intent: [f"机械占位-{intent}-{index}" for index in range(1, 4)]
        for intent in drifted_intents
    }
    draft = tmp_path / "draft.json"
    _write_draft(draft, mechanical_seed_draft)
    root = tmp_path / "queries"

    with pytest.raises(ValueError, match="substitute"):
        stage_seed_batch(draft, root, "seed-r1")

    assert not (root / "seeds/staging").exists()


def test_seed_batch_stays_staging_until_explicit_accept(
    tmp_path, mechanical_seed_draft
):
    draft = tmp_path / "draft.json"
    _write_draft(draft, mechanical_seed_draft)
    root = tmp_path / "queries"

    staged = stage_seed_batch(draft, root, "seed-r1")

    assert staged == root / "seeds/staging/seed-r1"
    assert (staged / "seed_examples.json").is_file()
    assert (staged / "manifest.json").is_file()
    assert not (root / "seeds/accepted/seed_examples.json").exists()
    assert corpus_status(root)["seed_status"] == "staging"
    assert corpus_status(root)["staged_seed_batch_ids"] == ["seed-r1"]
    with pytest.raises(ValueError, match="ACCEPT"):
        accept_seed_batch(root, "seed-r1", confirmation="yes")


def test_accept_seed_batch_publishes_hash_verified_seed_set(
    tmp_path, mechanical_seed_draft
):
    draft = tmp_path / "draft.json"
    _write_draft(draft, mechanical_seed_draft)
    root = tmp_path / "queries"
    staged = stage_seed_batch(draft, root, "seed-r1")
    staged_seed_bytes = (staged / "seed_examples.json").read_bytes()

    accepted = accept_seed_batch(root, "seed-r1", confirmation="ACCEPT")

    assert accepted == root / "seeds/accepted"
    assert not staged.exists()
    assert (accepted / "seed_examples.json").read_bytes() == staged_seed_bytes
    manifest = json.loads((accepted / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed_batch_id"] == "seed-r1"
    assert manifest["provider"] == "codex"
    assert manifest["model_display_name"] == "5.6 Sol Ultra"
    assert manifest["generated_at"] == "2026-07-11T00:00:00Z"
    assert manifest["seed_set_sha256"] == hashlib.sha256(staged_seed_bytes).hexdigest()


def test_seed_stage_refuses_overwrite(tmp_path, mechanical_seed_draft):
    draft = tmp_path / "draft.json"
    _write_draft(draft, mechanical_seed_draft)
    root = tmp_path / "queries"
    stage_seed_batch(draft, root, "seed-r1")

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        stage_seed_batch(draft, root, "seed-r1")


def test_reject_seed_batch_preserves_content_and_reason(
    tmp_path, mechanical_seed_draft
):
    draft = tmp_path / "draft.json"
    _write_draft(draft, mechanical_seed_draft)
    root = tmp_path / "queries"
    staged = stage_seed_batch(draft, root, "seed-r1")
    before = (staged / "seed_examples.json").read_bytes()

    rejected = reject_seed_batch(root, "seed-r1", reason="机械拒绝原因")

    assert rejected == root / "seeds/rejected/seed-r1"
    assert (rejected / "seed_examples.json").read_bytes() == before
    assert json.loads((rejected / "reason.json").read_text(encoding="utf-8")) == {
        "reason": "机械拒绝原因"
    }
