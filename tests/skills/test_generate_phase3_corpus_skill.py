import json
from pathlib import Path

from skillchain.synthesis.cli import main
from skillchain.synthesis.seeds import stage_seed_batch
from skillchain.taxonomy import INTENT_IDS

SKILL = Path(".agents/skills/generate-phase3-corpus")


def _load_seed_contract_example() -> dict:
    contract = (SKILL / "references/corpus-contract.md").read_text(encoding="utf-8")
    seed_section = contract.split("## Seed draft", 1)[1].split("## Query draft", 1)[0]
    payload = seed_section.split("```json", 1)[1].split("```", 1)[0]
    return json.loads(payload)


def test_skill_requires_model_confirmation_before_images_or_writes():
    body = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "5.6 Sol Ultra" in body
    assert "guard-model" in body
    assert "不得生成任何正式语料" in body
    assert "禁止派发子代理" in body
    assert body.index("guard-model") < body.index("view_image")
    assert body.index("guard-model") < body.index("inbox")
    assert "data/clean" in body


def test_skill_enforces_staging_and_explicit_acceptance():
    body = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "只写 staging" in body
    assert "明确接受" in body
    assert "不得自动 accept" in body
    assert "每批恰好 25 条" in body


def test_all_approved_roll_forward_preserves_the_next_human_gate():
    body = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    contract = (SKILL / "references/corpus-contract.md").read_text(encoding="utf-8")

    assert "roll-forward" in body
    assert "通过=25，待修改=0，拒绝=0" in body
    assert "positive integer `review_minutes`" in body
    assert "current user turn" in body
    assert "Never accept the newly generated batch" in body
    assert "Never roll back or mutate accepted" in body
    assert "requires its own human review" in body
    assert "next_batch_id=null" in body
    assert "never invent a successor" in body
    assert "25/0/0" in contract
    assert "successor is written only to staging" in contract
    assert "active plan is exhausted" in contract
    assert "do not fabricate another batch" in contract


def test_generate_requires_accepted_seed_set_before_composition():
    body = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "seed_status" in body
    assert "data/queries/seeds/accepted/seed_examples.json" in body
    assert body.index("guard-model") < body.index("seed_status")


def test_skill_frontmatter_and_ui_metadata_are_discoverable():
    body = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = body.split("---", 2)[1]

    assert "\nname: generate-phase3-corpus\n" in frontmatter
    assert "\ndescription: Use when " in frontmatter
    assert "Phase 3" in frontmatter
    assert (SKILL / "agents/openai.yaml").is_file()
    assert (SKILL / "references/corpus-contract.md").is_file()


def test_unconfirmed_model_dry_run_stops_before_any_corpus_write(tmp_path):
    root = tmp_path / "queries"

    result = main(["--queries-root", str(root), "guard-model"])

    assert result == 2
    assert not (root / "inbox").exists()
    assert not (root / "staging").exists()
    assert not (root / "accepted-ledger.jsonl").exists()


def test_contract_uses_only_mechanical_nonpublishable_examples():
    contract = (SKILL / "references/corpus-contract.md").read_text(encoding="utf-8")

    assert "机械占位-不可发布" in contract
    assert "测试专用" in contract
    assert '"plan_id"' in contract
    assert '"turns"' in contract
    assert "generated_at" in contract


def test_contract_seed_example_matches_frozen_intents_and_stages(tmp_path):
    example = _load_seed_contract_example()

    assert tuple(example["examples"]) == INTENT_IDS

    draft = tmp_path / "seed-contract-example.json"
    draft.write_text(json.dumps(example, ensure_ascii=False), encoding="utf-8")
    staged = stage_seed_batch(draft, tmp_path / "queries", "contract-example")
    staged_examples = json.loads(
        (staged / "seed_examples.json").read_text(encoding="utf-8")
    )

    assert set(staged_examples["examples"]) == set(INTENT_IDS)
