from datetime import datetime, timezone
import hashlib
import json
import os
import zipfile

import pytest

import skillchain.data.recipes as recipes
from skillchain.data._kb_source_lock import (
    KBSourceLock,
    KBSourceLockError,
    canonical_json_bytes,
)
from skillchain.data.kb_catalog import KBEntryV2, build_kb_catalog
from skillchain.data.recipes import (
    clean_recipes,
    clean_recipes_formal,
    iter_records,
    recipe_to_kb_entry,
)
from skillchain.data.wiki_zh import clean_wikipedia_formal


def _record(name="酸甜鲜美，草莓虾仁", dish="草莓虾仁"):
    return {
        "name": name,
        "dish": dish,
        "description": "酸甜鲜美。",
        "recipeIngredient": ["100克草莓", "300克虾仁"],
        "recipeInstructions": ["草莓切块", "虾仁炒熟后拌匀"],
        "author": "author_1",
        "keywords": ["草莓虾仁的做法"],
    }


def _write_zip(path, records):
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("recipe_corpus_full.json", content.encode("utf-8"))


def _source_lock_values(
    source,
    *,
    adapter_id="xiachufang-recipe-zip-v1",
    source_kind="recipe",
    source_dataset="fixture/xiachufang-recipes",
    **updates,
):
    revision = updates.get("source_revision", "fixture-revision-0123456789abcdef")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    review_record_sha256 = "c" * 64
    values = {
        "adapter_id": adapter_id,
        "source_kind": source_kind,
        "source_dataset": source_dataset,
        "source_revision": revision,
        "source_sha256": source_sha256,
        "source_uri": (
            f"https://example.test/kb/{revision}/{source.name}"
            f"#source-sha256={source_sha256}"
        ),
        "license_id": "CC-BY-4.0",
        "license_uri": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Fixture corpus contributors",
        "local_research_allowed": True,
        "cloud_processing_allowed": False,
        "redistribution_allowed": True,
        "public_demo_allowed": False,
        "reviewer_kind": "human",
        "reviewer_id": "fixture-reviewer-2",
        "reviewed_at": datetime(2026, 7, 20, tzinfo=timezone.utc),
        "review_record_uri": f"urn:sha256:{review_record_sha256}",
        "review_record_sha256": review_record_sha256,
    }
    values.update(updates)
    return values


def _write_source_lock(source, lock_path, **updates):
    lock = KBSourceLock(**_source_lock_values(source, **updates))
    content = canonical_json_bytes(lock)
    lock_path.write_bytes(content)
    return lock, hashlib.sha256(content).hexdigest()


def test_iter_records_streams_jsonl_member_without_extracting(tmp_path):
    archive = tmp_path / "recipes.zip"
    _write_zip(archive, [_record(), _record("红烧羊肉", "Unknown")])

    rows = list(iter_records(archive))

    assert [row["name"] for row in rows] == ["酸甜鲜美，草莓虾仁", "红烧羊肉"]


def test_recipe_to_kb_entry_uses_canonical_dish_and_structured_text():
    row = recipe_to_kb_entry(_record())

    assert row["title"] == "草莓虾仁"
    assert row["kind"] == "recipe"
    assert row["origin"] == "dump"
    entry = KBEntryV2.model_validate(row)
    assert entry.verification_status == "unverified"
    assert entry.license_id == "unknown-unverified"
    assert entry.attribution == "author_1"
    assert "食材：\n- 100克草莓\n- 300克虾仁" in row["text"]
    assert "步骤：\n1. 草莓切块\n2. 虾仁炒熟后拌匀" in row["text"]


def test_clean_recipes_filters_incomplete_and_deduplicates_titles(tmp_path):
    archive = tmp_path / "recipes.zip"
    incomplete = _record("没步骤", "没步骤")
    incomplete["recipeInstructions"] = []
    _write_zip(
        archive,
        [
            _record(),
            _record("同菜另一做法", "草莓虾仁"),
            _record("红烧羊肉", "Unknown"),
            incomplete,
        ],
    )
    output = tmp_path / "recipes.jsonl"

    report = clean_recipes(archive, output, max_entries=100_000)

    assert report.kept == 2
    assert report.duplicates == 1
    assert report.incomplete == 1
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["title"] for row in rows] == ["草莓虾仁", "红烧羊肉"]


def test_clean_recipes_escapes_unicode_line_separators_for_jsonl(tmp_path):
    archive = tmp_path / "recipes.zip"
    record = _record()
    record["description"] = "第一行\u2028第二行"
    _write_zip(archive, [record])
    output = tmp_path / "recipes.jsonl"

    clean_recipes(archive, output)

    physical_lines = output.read_text(encoding="utf-8").splitlines()
    assert len(physical_lines) == 1
    assert "第一行\n第二行" in json.loads(physical_lines[0])["text"]


def test_clean_recipes_rejects_whitespace_only_ingredients_and_steps(tmp_path):
    archive = tmp_path / "recipes.zip"
    blank = _record("空菜谱", "空菜谱")
    blank["recipeIngredient"] = [" ", "\t"]
    blank["recipeInstructions"] = ["\n"]
    _write_zip(archive, [blank])
    output = tmp_path / "recipes.jsonl"

    report = clean_recipes(archive, output)

    assert report.kept == 0
    assert report.incomplete == 1


def test_formal_recipe_clean_preserves_locked_record_citation_and_permissions(
    tmp_path,
):
    archive = tmp_path / "recipes.zip"
    _write_zip(archive, [_record()])
    lock_path = tmp_path / "recipe-source-lock.json"
    lock, lock_sha256 = _write_source_lock(archive, lock_path)
    output = tmp_path / "recipes-formal.jsonl"

    report = clean_recipes_formal(
        archive,
        output,
        source_lock_path=lock_path,
        expected_source_lock_sha256=lock_sha256,
    )

    assert report.kept == 1
    entry = KBEntryV2.model_validate_json(output.read_text(encoding="utf-8"))
    assert entry.formally_verified is True
    assert entry.source_dataset == lock.source_dataset
    assert entry.source_revision == lock.source_revision
    assert entry.source_record_id.startswith(
        "recipe_corpus_full.json:line:1:fields:recipe-core-v1:sha256:"
    )
    assert f"sha256={lock.source_sha256}" in entry.source_uri
    assert f"source-lock-sha256={lock_sha256}" in entry.source_uri
    assert (
        "recipe_corpus_full.json-line-1-fields-recipe-core-v1-sha256-"
        in entry.source_uri
    )
    assert entry.license_id == lock.license_id
    assert lock.attribution in (entry.attribution or "")
    assert "record author: author_1" in (entry.attribution or "")
    assert lock.local_research_allowed is True
    assert lock.cloud_processing_allowed is False
    assert lock.redistribution_allowed is True
    assert lock.public_demo_allowed is False
    assert (
        entry.content_sha256 == hashlib.sha256(entry.text.encode("utf-8")).hexdigest()
    )


def test_formal_recipe_rejects_noncanonical_or_symlinked_lock(tmp_path):
    archive = tmp_path / "recipes.zip"
    _write_zip(archive, [_record()])
    lock = KBSourceLock(**_source_lock_values(archive))
    lock_path = tmp_path / "noncanonical-lock.json"
    lock_path.write_text(
        json.dumps(lock.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    expected = hashlib.sha256(lock_path.read_bytes()).hexdigest()

    with pytest.raises(KBSourceLockError, match="canonical JSON"):
        clean_recipes_formal(
            archive,
            tmp_path / "noncanonical.jsonl",
            source_lock_path=lock_path,
            expected_source_lock_sha256=expected,
        )

    canonical_lock = tmp_path / "canonical-lock.json"
    _, canonical_sha256 = _write_source_lock(archive, canonical_lock)
    link = tmp_path / "lock-link.json"
    try:
        os.symlink(canonical_lock, link)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(KBSourceLockError, match="symlink"):
        clean_recipes_formal(
            archive,
            tmp_path / "symlinked-lock.jsonl",
            source_lock_path=link,
            expected_source_lock_sha256=canonical_sha256,
        )


def test_formal_recipe_rechecks_lock_before_publish(tmp_path, monkeypatch):
    archive = tmp_path / "recipes.zip"
    _write_zip(archive, [_record()])
    lock_path = tmp_path / "recipe-source-lock.json"
    _, lock_sha256 = _write_source_lock(archive, lock_path)
    output = tmp_path / "recipes-formal.jsonl"
    original_verify = recipes._verify_formal_inputs

    def mutate_then_verify(verified_lock, source_snapshot):
        lock_path.write_bytes(lock_path.read_bytes() + b" ")
        original_verify(verified_lock, source_snapshot)

    monkeypatch.setattr(recipes, "_verify_formal_inputs", mutate_then_verify)
    with pytest.raises(KBSourceLockError, match="changed during"):
        clean_recipes_formal(
            archive,
            output,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
        )
    assert not output.exists()


def test_recipe_archive_requires_one_regular_locked_member(tmp_path):
    archive = tmp_path / "ambiguous.zip"
    content = (json.dumps(_record(), ensure_ascii=False) + "\n").encode("utf-8")
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("recipe_corpus_full.json", content)
        with pytest.warns(UserWarning, match="Duplicate name"):
            target.writestr("recipe_corpus_full.json", content)
    lock_path = tmp_path / "recipe-source-lock.json"
    _, lock_sha256 = _write_source_lock(archive, lock_path)

    with pytest.raises(ValueError, match="exactly once"):
        clean_recipes_formal(
            archive,
            tmp_path / "recipes-formal.jsonl",
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
        )


def test_locked_adapter_outputs_can_feed_independently_locked_verified_catalog(
    tmp_path,
):
    wiki_source = tmp_path / "wiki.json"
    wiki_source.write_text(
        json.dumps(
            [
                {
                    "completion": "棉花\n棉花是锦葵科植物，可用于纺织棉布。",
                    "source": "upstream-wikipedia",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    wiki_lock_path = tmp_path / "wiki-source-lock.json"
    _, wiki_lock_sha256 = _write_source_lock(
        wiki_source,
        wiki_lock_path,
        adapter_id="wiki-zh-filtered-v1",
        source_kind="encyclopedia",
        source_dataset="fixture/wiki-zh",
        attribution="Fixture Wikipedia contributors",
    )
    wiki_output = tmp_path / "encyclopedia.jsonl"
    clean_wikipedia_formal(
        wiki_source,
        wiki_output,
        source_lock_path=wiki_lock_path,
        expected_source_lock_sha256=wiki_lock_sha256,
        keywords=("棉",),
        min_chars=10,
    )

    recipe_source = tmp_path / "recipes.zip"
    _write_zip(recipe_source, [_record()])
    recipe_lock_path = tmp_path / "recipe-source-lock.json"
    _, recipe_lock_sha256 = _write_source_lock(recipe_source, recipe_lock_path)
    recipe_output = tmp_path / "recipes.jsonl"
    clean_recipes_formal(
        recipe_source,
        recipe_output,
        source_lock_path=recipe_lock_path,
        expected_source_lock_sha256=recipe_lock_sha256,
    )

    catalog = build_kb_catalog(
        wiki_output,
        recipe_output,
        tmp_path / "kb-catalog",
        expected_encyclopedia_sha256=hashlib.sha256(
            wiki_output.read_bytes()
        ).hexdigest(),
        expected_recipe_sha256=hashlib.sha256(recipe_output.read_bytes()).hexdigest(),
    )

    assert catalog.manifest.mode == "verified"
    assert catalog.external_sha256_verified is True
    assert all(entry.formally_verified for entry in catalog.entries)
