import hashlib
import json
import os
from pathlib import Path

import pytest

from skillchain.data.kb_catalog import (
    build_kb_catalog,
    canonical_json_bytes,
    load_kb_catalog,
)
import skillchain.tools.kb_index as kb_index
from skillchain.tools.kb_index import (
    KBIndexError,
    build_kb_bundle,
    load_kb_bundle,
)


def _entry(entry_id: str, kind: str, text: str) -> dict:
    return {
        "schema_version": 2,
        "entry_id": entry_id,
        "title": f"标题-{entry_id}",
        "text": text,
        "kind": kind,
        "origin": "dump",
        "source_dataset": "fixture-kb",
        "source_revision": "revision-1",
        "source_record_id": entry_id,
        "source_uri": f"https://example.test/kb/{entry_id}",
        "license_id": "CC-BY-4.0",
        "attribution": None,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "entity_group_id": f"entity-{entry_id}",
        "near_duplicate_cluster_id": None,
        "derivation_parent_entry_ids": [],
        "synth_provider": None,
        "synth_model": None,
        "synthesis_prompt_sha256": None,
        "verification_status": "source_verified",
    }


def _catalog(tmp_path: Path, count: int = 3):
    encyclopedia = [
        _entry(f"encyclopedia-{index}", "encyclopedia", f"百科共同词 熊猫 {index}")
        for index in range(count)
    ]
    recipes = [
        _entry(f"recipe-{index}", "recipe", f"菜谱共同词 鸡肉 {index}")
        for index in range(count)
    ]
    encyclopedia_path = tmp_path / "encyclopedia.jsonl"
    recipe_path = tmp_path / "recipes.jsonl"
    for path, rows in ((encyclopedia_path, encyclopedia), (recipe_path, recipes)):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    output = tmp_path / "catalog"
    return build_kb_catalog(
        encyclopedia_path,
        recipe_path,
        output,
        expected_encyclopedia_sha256=hashlib.sha256(
            encyclopedia_path.read_bytes()
        ).hexdigest(),
        expected_recipe_sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
    )


def _build_bundle(catalog, output: Path, **kwargs):
    return build_kb_bundle(
        catalog.root,
        output,
        expected_catalog_sha256=catalog.manifest.catalog_sha256,
        **kwargs,
    )


def _load_bundle(bundle, catalog, **kwargs):
    return load_kb_bundle(
        bundle.root,
        catalog=catalog,
        expected_bundle_sha256=bundle.manifest.bundle_sha256,
        **kwargs,
    )


def test_builds_and_loads_two_indices_as_one_verified_bundle(tmp_path: Path):
    catalog = _catalog(tmp_path)
    output = tmp_path / "kb-index"

    bundle = _build_bundle(catalog, output)

    assert bundle.manifest.mode == "verified"
    assert bundle.external_sha256_verified is False
    assert bundle.manifest.complete is True
    assert set(bundle.manifest.indices) == {"encyclopedia", "recipe"}
    assert len(bundle.encyclopedia.entries) == 3
    assert len(bundle.recipe.entries) == 3
    for index in (bundle.encyclopedia, bundle.recipe):
        assert index.manifest.complete is True
        assert index.manifest.entry_count == 3
        assert index.manifest.tokenizer.hmm is False
        assert index.manifest.bm25.method == "lucene"
        assert index.retriever.scores["num_docs"] == 3
    locked_catalog = load_kb_catalog(
        catalog.root,
        expected_catalog_sha256=catalog.manifest.catalog_sha256,
    )
    locked_bundle = _load_bundle(bundle, locked_catalog)
    assert locked_bundle.manifest == bundle.manifest
    assert locked_bundle.external_sha256_verified is True
    with pytest.raises(KBIndexError, match="external expected bundle"):
        load_kb_bundle(output, catalog=locked_catalog)
    with pytest.raises(KBIndexError, match="external expected lock"):
        load_kb_bundle(
            output,
            catalog=locked_catalog,
            expected_bundle_sha256="0" * 64,
        )
    with pytest.raises(KBIndexError, match="requires its verified catalog"):
        load_kb_bundle(
            output,
            expected_bundle_sha256=bundle.manifest.bundle_sha256,
        )
    with pytest.raises(FileExistsError):
        _build_bundle(catalog, output)


def test_provisional_index_is_limited_and_requires_two_explicit_opt_ins(
    tmp_path: Path,
):
    catalog = _catalog(tmp_path, count=3)
    output = tmp_path / "kb-smoke"
    with pytest.raises(KBIndexError, match="positive limit_per_kind"):
        _build_bundle(catalog, output, allow_provisional=True)

    bundle = _build_bundle(
        catalog,
        output,
        limit_per_kind=1,
        allow_provisional=True,
    )
    assert bundle.manifest.mode == "provisional"
    assert bundle.manifest.complete is False
    assert len(bundle.encyclopedia.entries) == 1
    with pytest.raises(KBIndexError, match="explicit opt-in"):
        load_kb_bundle(output)
    assert load_kb_bundle(output, allow_provisional=True).manifest.mode == "provisional"


def test_bundle_publication_is_joint_and_failure_leaves_no_partial_destination(
    tmp_path: Path, monkeypatch
):
    catalog = _catalog(tmp_path)
    output = tmp_path / "kb-index"
    original = kb_index._build_child_index

    def fail_second(destination, **kwargs):
        if kwargs["kind"] == "recipe":
            raise RuntimeError("recipe build failed")
        return original(destination, **kwargs)

    monkeypatch.setattr(kb_index, "_build_child_index", fail_second)
    with pytest.raises(RuntimeError, match="recipe build failed"):
        _build_bundle(catalog, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".kb-index.staging-*"))


def test_index_rechecks_catalog_bytes_after_both_child_builds(
    tmp_path: Path, monkeypatch
):
    catalog = _catalog(tmp_path)
    output = tmp_path / "kb-index"
    original = kb_index._build_child_index
    calls = 0

    def mutate_catalog_after_first_child(destination, **kwargs):
        nonlocal calls
        result = original(destination, **kwargs)
        calls += 1
        if calls == 1:
            path = catalog.root / "entries.jsonl"
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(
        kb_index, "_build_child_index", mutate_catalog_after_first_child
    )
    with pytest.raises(Exception, match="catalog artifact changed"):
        _build_bundle(catalog, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "encyclopedia/data.csc.index.npy",
        "encyclopedia/entries.jsonl",
        "recipe/vocab.index.json",
        "recipe/manifest.json",
        "manifest.json",
    ],
)
def test_load_rejects_each_tampered_index_artifact(tmp_path: Path, relative_path: str):
    catalog = _catalog(tmp_path)
    output = tmp_path / "kb-index"
    bundle = _build_bundle(catalog, output)
    target = output / relative_path
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(Exception):
        _load_bundle(bundle, catalog)


def test_load_rejects_extra_files_and_symlink_artifacts(tmp_path: Path):
    catalog = _catalog(tmp_path)
    output = tmp_path / "kb-index"
    bundle = _build_bundle(catalog, output)
    (output / "recipe" / "extra.bin").write_bytes(b"extra")
    with pytest.raises(KBIndexError, match="artifact set"):
        _load_bundle(bundle, catalog)

    output2 = tmp_path / "kb-index-2"
    bundle2 = _build_bundle(catalog, output2)
    target = output2 / "encyclopedia" / "entries.jsonl"
    backup = tmp_path / "saved-entries.jsonl"
    target.replace(backup)
    try:
        os.symlink(backup, target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(Exception, match="regular non-symlink"):
        _load_bundle(bundle2, catalog)


def test_load_rejects_token_row_reordering_even_with_rehashed_manifest_chain(
    tmp_path: Path,
):
    catalog = _catalog(tmp_path)
    output = tmp_path / "kb-index"
    _build_bundle(catalog, output)
    token_path = output / "encyclopedia" / "tokens.jsonl"
    lines = token_path.read_bytes().splitlines(keepends=True)
    changed_tokens = b"".join(reversed(lines))
    token_path.write_bytes(changed_tokens)

    child_manifest_path = output / "encyclopedia" / "manifest.json"
    child = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    child["artifacts"]["tokens.jsonl"].update(
        {
            "bytes": len(changed_tokens),
            "sha256": hashlib.sha256(changed_tokens).hexdigest(),
        }
    )
    child_unsigned = dict(child)
    child_unsigned.pop("integrity_sha256")
    child["integrity_sha256"] = hashlib.sha256(
        canonical_json_bytes(child_unsigned)
    ).hexdigest()
    child_bytes = canonical_json_bytes(child) + b"\n"
    child_manifest_path.write_bytes(child_bytes)

    bundle_manifest_path = output / "manifest.json"
    bundle = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    bundle["indices"]["encyclopedia"].update(
        {
            "manifest_sha256": hashlib.sha256(child_bytes).hexdigest(),
            "integrity_sha256": child["integrity_sha256"],
        }
    )
    bundle_unsigned = dict(bundle)
    bundle_unsigned.pop("bundle_sha256")
    bundle["bundle_sha256"] = hashlib.sha256(
        canonical_json_bytes(bundle_unsigned)
    ).hexdigest()
    bundle_manifest_path.write_bytes(canonical_json_bytes(bundle) + b"\n")

    with pytest.raises(KBIndexError, match="row indices"):
        load_kb_bundle(
            output,
            catalog=catalog,
            expected_bundle_sha256=bundle["bundle_sha256"],
        )
