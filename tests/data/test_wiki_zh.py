from datetime import datetime, timezone
import hashlib
import json
import os

import pytest
from pydantic import ValidationError

import skillchain.data.wiki_zh as wiki_zh
from skillchain.data._kb_source_lock import (
    KBSourceLock,
    KBSourceLockError,
    canonical_json_bytes,
)
from skillchain.data.kb_catalog import KBEntryV2
from skillchain.data.wiki_zh import (
    clean_wikipedia,
    clean_wikipedia_formal,
    console_safe,
    derive_title,
    iter_records,
)


def _source_lock_values(source, **updates):
    revision = updates.get("source_revision", "fixture-revision-0123456789abcdef")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    review_record_sha256 = "b" * 64
    values = {
        "adapter_id": "wiki-zh-filtered-v1",
        "source_kind": "encyclopedia",
        "source_dataset": "fixture/wiki-zh",
        "source_revision": revision,
        "source_sha256": source_sha256,
        "source_uri": (
            f"https://example.test/wiki/{revision}/{source.name}"
            f"#source-sha256={source_sha256}"
        ),
        "license_id": "CC-BY-SA-4.0",
        "license_uri": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Fixture Wikipedia contributors",
        "local_research_allowed": True,
        "cloud_processing_allowed": False,
        "redistribution_allowed": True,
        "public_demo_allowed": True,
        "reviewer_kind": "human",
        "reviewer_id": "fixture-reviewer-1",
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


def test_iter_records_streams_json_array(tmp_path):
    source = tmp_path / "wiki.json"
    source.write_text(
        json.dumps(
            [
                {"completion": "棉花是锦葵科植物。", "source": "wiki"},
                {"completion": "涤纶是一种合成纤维。", "source": "wiki"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert list(iter_records(source)) == [
        {"completion": "棉花是锦葵科植物。", "source": "wiki"},
        {"completion": "涤纶是一种合成纤维。", "source": "wiki"},
    ]


def test_derive_title_handles_heading_and_lead_sentence():
    assert derive_title("我的英雄学院\n这是一部动画电影。") == "我的英雄学院"
    assert derive_title("昭通机场（ZPZT）是位于云南的民用机场。") == "昭通机场"


def test_console_safe_escapes_characters_not_supported_by_windows_gbk():
    assert console_safe("乌克兰字母 і", encoding="gbk") == "乌克兰字母 \\u0456"


def test_clean_wikipedia_writes_relevant_deduplicated_kb_entries(tmp_path):
    source = tmp_path / "wiki.json"
    source.write_text(
        json.dumps(
            [
                {
                    "completion": "棉花\n棉花是锦葵科植物，可用于纺织棉布。",
                    "source": "wiki",
                },
                {"completion": "棉花\n棉花是重复的锦葵科植物条目。", "source": "wiki"},
                {"completion": "某机场是一个与商品百科无关的地点。", "source": "wiki"},
                {"completion": "棉", "source": "wiki"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "encyclopedia.jsonl"

    report = clean_wikipedia(
        source,
        output,
        keywords=("棉", "植物", "纺织"),
        min_chars=10,
    )

    assert report.kept == 1
    assert report.duplicates == 1
    assert report.irrelevant == 1
    assert report.too_short == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    entry = KBEntryV2.model_validate(row)
    assert entry.title == "棉花"
    assert entry.text == "棉花\n棉花是锦葵科植物，可用于纺织棉布。"
    assert entry.kind == "encyclopedia"
    assert entry.origin == "dump"
    assert entry.verification_status == "unverified"
    assert entry.license_id == "unknown-unverified"
    assert entry.attribution == "wiki"
    assert row["entry_id"].startswith("wiki-")


def test_clean_wikipedia_escapes_unicode_line_separators_for_jsonl(tmp_path):
    source = tmp_path / "wiki.json"
    source.write_text(
        json.dumps(
            [
                {
                    "completion": "棉花\u2028棉花是可用于纺织的植物材料。",
                    "source": "wiki",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "encyclopedia.jsonl"

    clean_wikipedia(source, output, keywords=("棉",), min_chars=10)

    physical_lines = output.read_text(encoding="utf-8").splitlines()
    assert len(physical_lines) == 1
    assert json.loads(physical_lines[0])["text"] == "棉花\n棉花是可用于纺织的植物材料。"


def test_formal_wikipedia_clean_requires_external_lock_and_preserves_citation(
    tmp_path,
):
    source = tmp_path / "wiki.json"
    source.write_text(
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
    lock_path = tmp_path / "wiki-source-lock.json"
    lock, lock_sha256 = _write_source_lock(source, lock_path)
    output = tmp_path / "encyclopedia-formal.jsonl"

    report = clean_wikipedia_formal(
        source,
        output,
        source_lock_path=lock_path,
        expected_source_lock_sha256=lock_sha256,
        keywords=("棉",),
        min_chars=10,
    )

    assert report.kept == 1
    entry = KBEntryV2.model_validate_json(output.read_text(encoding="utf-8"))
    assert entry.formally_verified is True
    assert entry.verification_status == "source_verified"
    assert entry.source_dataset == lock.source_dataset
    assert entry.source_revision == lock.source_revision
    assert entry.license_id == lock.license_id
    assert entry.attribution == lock.attribution
    assert entry.source_record_id.startswith(
        "item:1:field:completion:canonical-sha256:"
    )
    assert f"sha256={lock.source_sha256}" in entry.source_uri
    assert f"source-lock-sha256={lock_sha256}" in entry.source_uri
    assert "record=item-1-field-completion-canonical-sha256-" in entry.source_uri
    assert (
        entry.content_sha256 == hashlib.sha256(entry.text.encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "source_revision": "main",
                "source_uri": "https://example.test/wiki/main/wiki.json",
            },
            "immutable revision",
        ),
        (
            {
                "source_revision": "unknown",
                "source_uri": "https://example.test/wiki/unknown/wiki.json",
            },
            "immutable revision",
        ),
        ({"license_id": "unknown-unverified"}, "externally reviewed"),
        (
            {"source_uri": ("https://example.test/wiki/resolve/main/wiki.json")},
            "mutable revision",
        ),
        (
            {
                "source_uri": (
                    "https://example.test/wiki/"
                    "fixture-revision-0123456789abcdef/wiki.json"
                )
            },
            "content-addressed",
        ),
        (
            {"review_record_uri": "urn:fixture-review:not-content-addressed"},
            "review_record_sha256",
        ),
        ({"reviewer_kind": "llm"}, "literal_error"),
        ({"reviewer_id": "gpt-5.6-sol"}, "LLM self-attestation"),
        ({"local_research_allowed": False}, "literal_error"),
    ],
)
def test_source_lock_rejects_mutable_unknown_or_llm_attested_claims(
    tmp_path, updates, message
):
    source = tmp_path / "wiki.json"
    source.write_text("[]", encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        KBSourceLock(**_source_lock_values(source, **updates))


def test_formal_wikipedia_rejects_missing_or_wrong_external_lock_digest(tmp_path):
    source = tmp_path / "wiki.json"
    source.write_text(
        json.dumps([{"completion": "棉花\n棉花是用于纺织的植物材料。"}]),
        encoding="utf-8",
    )
    lock_path = tmp_path / "source-lock.json"
    _, lock_sha256 = _write_source_lock(source, lock_path)

    for index, expected in enumerate(("", "0" * 64)):
        output = tmp_path / f"rejected-{index}.jsonl"
        with pytest.raises(KBSourceLockError, match="external expected|digest"):
            clean_wikipedia_formal(
                source,
                output,
                source_lock_path=lock_path,
                expected_source_lock_sha256=expected,
                keywords=("棉",),
                min_chars=5,
            )
        assert not output.exists()
    assert len(lock_sha256) == 64


def test_formal_wikipedia_rejects_self_consistent_but_wrong_source_hash(tmp_path):
    source = tmp_path / "wiki.json"
    source.write_text(
        json.dumps([{"completion": "棉花\n棉花是用于纺织的植物材料。"}]),
        encoding="utf-8",
    )
    lock_path = tmp_path / "source-lock.json"
    _, lock_sha256 = _write_source_lock(
        source,
        lock_path,
        source_sha256="0" * 64,
        source_uri=(
            "https://example.test/wiki/fixture-revision-0123456789abcdef/"
            f"wiki.json#source-sha256={'0' * 64}"
        ),
    )
    output = tmp_path / "formal.jsonl"

    with pytest.raises(KBSourceLockError, match="external source lock"):
        clean_wikipedia_formal(
            source,
            output,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
            keywords=("棉",),
            min_chars=5,
        )
    assert not output.exists()


def test_formal_wikipedia_rechecks_source_before_create_only_publish(
    tmp_path, monkeypatch
):
    source = tmp_path / "wiki.json"
    source.write_text(
        json.dumps([{"completion": "棉花\n棉花是用于纺织的植物材料。"}]),
        encoding="utf-8",
    )
    lock_path = tmp_path / "source-lock.json"
    _, lock_sha256 = _write_source_lock(source, lock_path)
    output = tmp_path / "formal.jsonl"
    original_verify = wiki_zh._verify_formal_inputs

    def mutate_then_verify(verified_lock, source_snapshot):
        source.write_bytes(source.read_bytes() + b" ")
        original_verify(verified_lock, source_snapshot)

    monkeypatch.setattr(wiki_zh, "_verify_formal_inputs", mutate_then_verify)
    with pytest.raises(KBSourceLockError, match="changed during"):
        clean_wikipedia_formal(
            source,
            output,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
            keywords=("棉",),
            min_chars=5,
        )
    assert not output.exists()


def test_formal_wikipedia_rejects_symlink_source_and_existing_output(tmp_path):
    source = tmp_path / "wiki.json"
    source.write_text(
        json.dumps([{"completion": "棉花\n棉花是用于纺织的植物材料。"}]),
        encoding="utf-8",
    )
    lock_path = tmp_path / "source-lock.json"
    _, lock_sha256 = _write_source_lock(source, lock_path)
    output = tmp_path / "formal.jsonl"
    output.write_bytes(b"do-not-replace")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        clean_wikipedia_formal(
            source,
            output,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
            keywords=("棉",),
            min_chars=5,
        )
    assert output.read_bytes() == b"do-not-replace"

    link = tmp_path / "wiki-link.json"
    try:
        os.symlink(source, link)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    output.unlink()
    with pytest.raises(KBSourceLockError, match="symlink"):
        clean_wikipedia_formal(
            link,
            output,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
            keywords=("棉",),
            min_chars=5,
        )
    assert not output.exists()
