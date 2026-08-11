import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skillchain.data.kb_catalog import build_kb_catalog
from skillchain.tools.kb_index import build_kb_bundle, load_kb_bundle
from skillchain.tools.kb_lookup import (
    KBLookupService,
    KB_QUERY_MAX_CHARS,
    configure_kb_lookup_service_factory,
    encyclopedia_lookup,
    recipe_lookup,
)


def _entry(entry_id: str, kind: str, title: str, text: str) -> dict:
    return {
        "schema_version": 2,
        "entry_id": entry_id,
        "title": title,
        "text": text,
        "kind": kind,
        "origin": "dump",
        "source_dataset": "fixture-kb",
        "source_revision": "revision-1",
        "source_record_id": entry_id,
        "source_uri": f"https://example.test/kb/{entry_id}",
        "license_id": "CC-BY-4.0",
        "attribution": "Fixture Author",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "entity_group_id": f"entity-{entry_id}",
        "near_duplicate_cluster_id": None,
        "derivation_parent_entry_ids": [],
        "synth_provider": None,
        "synth_model": None,
        "synthesis_prompt_sha256": None,
        "verification_status": "source_verified",
    }


def _service(
    tmp_path: Path,
    *,
    encyclopedia: list[dict] | None = None,
    recipes: list[dict] | None = None,
    provisional_limit: int | None = None,
) -> KBLookupService:
    encyclopedia = encyclopedia or [
        _entry(
            "encyclopedia-panda",
            "encyclopedia",
            "大熊猫",
            "大熊猫主要以竹子为食。",
        ),
        _entry(
            "encyclopedia-granite",
            "encyclopedia",
            "花岗岩",
            "花岗岩是一种火成岩。",
        ),
    ]
    recipes = recipes or [
        _entry(
            "recipe-kung-pao",
            "recipe",
            "宫保鸡丁",
            "宫保鸡丁需要鸡肉、花生和调味汁。",
        ),
        _entry(
            "recipe-fried-rice",
            "recipe",
            "蛋炒饭",
            "蛋炒饭需要米饭、鸡蛋和食用油。",
        ),
    ]
    encyclopedia_path = tmp_path / "encyclopedia.jsonl"
    recipe_path = tmp_path / "recipes.jsonl"
    for path, rows in ((encyclopedia_path, encyclopedia), (recipe_path, recipes)):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    catalog = build_kb_catalog(
        encyclopedia_path,
        recipe_path,
        tmp_path / "catalog",
        expected_encyclopedia_sha256=hashlib.sha256(
            encyclopedia_path.read_bytes()
        ).hexdigest(),
        expected_recipe_sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
    )
    if provisional_limit is None:
        published = build_kb_bundle(
            catalog.root,
            tmp_path / "index",
            expected_catalog_sha256=catalog.manifest.catalog_sha256,
        )
        bundle = load_kb_bundle(
            published.root,
            catalog=catalog,
            expected_bundle_sha256=published.manifest.bundle_sha256,
        )
        return KBLookupService(bundle)
    bundle = build_kb_bundle(
        catalog.root,
        tmp_path / "index-smoke",
        expected_catalog_sha256=catalog.manifest.catalog_sha256,
        limit_per_kind=provisional_limit,
        allow_provisional=True,
    )
    return KBLookupService(bundle, allow_provisional=True)


def test_lookup_keeps_libraries_isolated_and_returns_exact_citation_and_binding(
    tmp_path: Path,
):
    service = _service(tmp_path)

    encyclopedia_hits = service.encyclopedia_lookup("大熊猫")
    recipe_hits = service.recipe_lookup("宫保鸡丁")

    assert encyclopedia_hits[0]["citation"]["entry_id"] == "encyclopedia-panda"
    assert recipe_hits[0]["citation"]["entry_id"] == "recipe-kung-pao"
    assert all(hit["kind"] == "encyclopedia" for hit in encyclopedia_hits)
    assert all(hit["kind"] == "recipe" for hit in recipe_hits)
    hit = encyclopedia_hits[0]
    assert hit["citation"]["char_start"] == 0
    assert hit["citation"]["char_end"] == len(hit["text"])
    assert (
        hit["citation"]["excerpt_sha256"]
        == hashlib.sha256(hit["text"].encode("utf-8")).hexdigest()
    )
    assert hit["verification_status"] == "source_verified"
    binding = hit["artifact_binding"]
    assert binding["mode"] == "verified"
    assert binding["kind"] == "encyclopedia"
    assert len(binding["bundle_sha256"]) == 64
    assert len(binding["index_integrity_sha256"]) == 64
    assert binding["catalog_sha256"] == service.bundle.manifest.catalog_sha256


def test_verified_lookup_service_rejects_bundle_without_external_lock(tmp_path: Path):
    service = _service(tmp_path)
    forged = replace(service.bundle, external_sha256_verified=False)
    with pytest.raises(ValueError, match="externally locked"):
        KBLookupService(forged)


def test_lookup_returns_empty_for_out_of_vocabulary_instead_of_zero_score_noise(
    tmp_path: Path,
):
    service = _service(tmp_path)
    assert service.encyclopedia_lookup("zzzzzzzzzzzzzzzzzzzzzzzz") == []
    assert service.recipe_lookup("qqqqqqqqqqqqqqqqqqqqqqqq") == []


def test_lookup_rejects_blank_wrong_type_and_oversize_queries(tmp_path: Path):
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="blank"):
        service.encyclopedia_lookup(" \t\n")
    with pytest.raises(TypeError, match="string"):
        service.recipe_lookup(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not exceed"):
        service.encyclopedia_lookup("熊" * (KB_QUERY_MAX_CHARS + 1))


def test_full_score_ties_are_globally_sorted_by_entry_id_not_candidate_cutoff(
    tmp_path: Path,
):
    encyclopedia = [
        _entry(
            f"encyclopedia-{index:03d}",
            "encyclopedia",
            f"Title {index:03d}",
            "sharedtoken",
        )
        for index in reversed(range(30))
    ]
    recipes = [
        _entry("recipe-1", "recipe", "Recipe One", "recipetoken"),
        _entry("recipe-2", "recipe", "Recipe Two", "othertoken"),
    ]
    service = _service(tmp_path, encyclopedia=encyclopedia, recipes=recipes)

    hits = service.encyclopedia_lookup("sharedtoken")

    assert [hit["citation"]["entry_id"] for hit in hits] == [
        f"encyclopedia-{index:03d}" for index in range(5)
    ]
    assert all(hit["score"] == hits[0]["score"] for hit in hits)
    assert [hit["rank"] for hit in hits] == [1, 2, 3, 4, 5]


def test_lookup_scores_are_finite_rounded_and_repeatable(tmp_path: Path):
    service = _service(tmp_path)
    first = service.recipe_lookup("鸡肉")
    second = service.recipe_lookup("鸡肉")
    assert first == second
    assert first
    assert all(np.isfinite(hit["score"]) and hit["score"] > 0 for hit in first)
    assert all(hit["score"] == round(hit["score"], 8) for hit in first)


def test_lookup_rejects_invalid_score_shape_and_non_finite_values(tmp_path: Path):
    service = _service(tmp_path)
    retriever = service.bundle.encyclopedia.retriever
    retriever.get_scores = lambda tokens: np.asarray([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="score shape"):
        service.encyclopedia_lookup("大熊猫")

    retriever.get_scores = lambda tokens: np.asarray(
        [float("nan")] * len(service.bundle.encyclopedia.entries), dtype=np.float32
    )
    with pytest.raises(ValueError, match="non-finite"):
        service.encyclopedia_lookup("大熊猫")


def test_provisional_hits_carry_only_explicit_provisional_identity(tmp_path: Path):
    service = _service(tmp_path, provisional_limit=1)
    hit = service.encyclopedia_lookup("大熊猫")[0]
    assert hit["artifact_binding"] == {
        "mode": "provisional",
        "kind": "encyclopedia",
        "bundle_sha256": None,
        "index_integrity_sha256": None,
        "catalog_sha256": None,
        "catalog_entries_sha256": None,
        "indexed_entries_sha256": None,
        "tokenizer_policy_version": None,
        "ranking_policy_version": None,
    }


def test_module_functions_use_resettable_explicit_factory(tmp_path: Path):
    service = _service(tmp_path)
    configure_kb_lookup_service_factory(lambda: service)
    try:
        assert encyclopedia_lookup("大熊猫")[0]["kind"] == "encyclopedia"
        assert recipe_lookup("宫保鸡丁")[0]["kind"] == "recipe"
    finally:
        configure_kb_lookup_service_factory(None)
