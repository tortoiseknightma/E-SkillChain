from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillchain.taxonomy import (
    CAPABILITIES_BY_INTENT,
    DEFAULT_TAXONOMY_PATH,
    DEFAULT_TAXONOMY_SHA256,
    INTENT_IDS,
    TAXONOMY_VERSION,
    TaxonomyError,
    capability_for_intent,
    capabilities_for_intent,
    load_default_taxonomy_registry,
    load_taxonomy_registry,
    requires_card_for_intent,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rehash(raw: dict) -> None:
    payload = dict(raw)
    payload.pop("taxonomy_sha256", None)
    raw["taxonomy_sha256"] = sha256_bytes(canonical_json_bytes(payload))


def _copy_raw() -> dict:
    return json.loads(DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"))


def test_default_taxonomy_is_frozen_complete_and_runtime_compatible():
    registry = load_default_taxonomy_registry()

    assert registry.taxonomy_version == TAXONOMY_VERSION
    assert registry.taxonomy_sha256 == DEFAULT_TAXONOMY_SHA256
    assert tuple(item.intent_id for item in registry.intents) == INTENT_IDS
    assert len(registry.capabilities) == 6
    assert all(
        8 <= len(item.candidate_capability_ids) <= 12 for item in registry.intents
    )
    assert {
        item.intent_id: item.primary_capability_id for item in registry.intents
    } == {intent: capability_for_intent(intent) for intent in INTENT_IDS}
    assert {item.intent_id: item.requires_card for item in registry.capabilities} == {
        intent: requires_card_for_intent(intent) for intent in INTENT_IDS
    }
    assert set(CAPABILITIES_BY_INTENT) == set(INTENT_IDS)
    assert {item.capability_id for item in capabilities_for_intent("utility")} == {
        "utility.document_reading",
        "utility.recipe_guidance",
    }
    assert capability_for_intent("utility") == "utility.document_reading"


def test_taxonomy_expected_hash_rejects_coordinated_content_change(tmp_path: Path):
    raw = _copy_raw()
    raw["intents"][0]["definition"] = "A benign but unreviewed replacement definition."
    _rehash(raw)
    changed = tmp_path / "taxonomy.json"
    changed.write_bytes(_canonical_pretty(raw))

    assert load_taxonomy_registry(changed).taxonomy_sha256 != DEFAULT_TAXONOMY_SHA256
    with pytest.raises(TaxonomyError, match="expected SHA-256"):
        load_taxonomy_registry(changed, expected_sha256=DEFAULT_TAXONOMY_SHA256)


@pytest.mark.parametrize(
    "contamination",
    [
        "Use a candidate Skill to define this boundary.",
        "Copy the rubric and gold labels.",
        "Tune this definition from evaluation results.",
        "Read the training trajectory first.",
        "从技能库与评测结果反推该规则。",
    ],
)
def test_taxonomy_rejects_experiment_derived_references(
    tmp_path: Path, contamination: str
):
    raw = _copy_raw()
    raw["intents"][0]["definition"] = contamination
    _rehash(raw)
    path = tmp_path / "contaminated.json"
    path.write_bytes(_canonical_pretty(raw))

    with pytest.raises(TaxonomyError, match="violates schema"):
        load_taxonomy_registry(path)


def test_taxonomy_rejects_invalid_matrix_and_extra_fields(tmp_path: Path):
    raw = _copy_raw()
    raw["intents"][0]["candidate_capability_ids"] = ["product.exact_match"]
    raw["unexpected"] = True
    _rehash(raw)
    path = tmp_path / "invalid.json"
    path.write_bytes(_canonical_pretty(raw))

    with pytest.raises(TaxonomyError, match="violates schema"):
        load_taxonomy_registry(path)


def test_taxonomy_rejects_noncanonical_duplicate_and_symlink(tmp_path: Path):
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(_copy_raw(), ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(TaxonomyError, match="canonical pretty JSON"):
        load_taxonomy_registry(noncanonical)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(TaxonomyError, match="duplicate key"):
        load_taxonomy_registry(duplicate)

    if not hasattr(os, "symlink"):
        return
    link = tmp_path / "link.json"
    try:
        link.symlink_to(DEFAULT_TAXONOMY_PATH)
    except OSError:
        pytest.skip("symlink creation is not permitted")
    with pytest.raises(TaxonomyError, match="safely"):
        load_taxonomy_registry(link)


def test_taxonomy_rejects_invalid_expected_digest():
    with pytest.raises(TaxonomyError, match="expected taxonomy SHA-256"):
        load_taxonomy_registry(DEFAULT_TAXONOMY_PATH, expected_sha256="not-a-hash")
