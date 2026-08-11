import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import skillchain.data.kb_catalog as kb_catalog
from skillchain.data.kb_catalog import (
    KBCatalogError,
    KBEntryV2,
    build_kb_catalog,
    canonical_json_bytes,
    load_kb_catalog,
)


def _entry(
    entry_id: str,
    kind: str,
    text: str,
    *,
    entity: str | None = None,
    parents: list[str] | None = None,
    verification_status: str = "source_verified",
) -> dict:
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
        "attribution": "Fixture Author",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "entity_group_id": entity or f"entity-{entry_id}",
        "near_duplicate_cluster_id": None,
        "derivation_parent_entry_ids": parents or [],
        "synth_provider": None,
        "synth_model": None,
        "synthesis_prompt_sha256": None,
        "verification_status": verification_status,
    }


def _write_sources(
    root: Path,
    *,
    encyclopedia: list[dict] | None = None,
    recipes: list[dict] | None = None,
) -> tuple[Path, Path]:
    encyclopedia = encyclopedia or [
        _entry("encyclopedia-1", "encyclopedia", "大熊猫以竹子为主要食物。"),
        _entry("encyclopedia-2", "encyclopedia", "花岗岩是一种岩石。"),
    ]
    recipes = recipes or [
        _entry("recipe-1", "recipe", "宫保鸡丁需要鸡肉、花生与调味料。"),
        _entry("recipe-2", "recipe", "蛋炒饭需要米饭和鸡蛋。"),
    ]
    encyclopedia_path = root / "encyclopedia-source.jsonl"
    recipe_path = root / "recipe-source.jsonl"
    for path, rows in ((encyclopedia_path, encyclopedia), (recipe_path, recipes)):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return encyclopedia_path, recipe_path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_verified(encyclopedia_path: Path, recipe_path: Path, output: Path):
    return build_kb_catalog(
        encyclopedia_path,
        recipe_path,
        output,
        expected_encyclopedia_sha256=_sha256_file(encyclopedia_path),
        expected_recipe_sha256=_sha256_file(recipe_path),
    )


def test_kb_entry_v2_is_strict_and_binds_content_bytes():
    row = _entry("encyclopedia-1", "encyclopedia", "熊猫正文")
    assert KBEntryV2.model_validate(row).content_sha256 == row["content_sha256"]

    with pytest.raises(ValidationError, match="extra_forbidden"):
        KBEntryV2.model_validate({**row, "unexpected": "discard me"})
    with pytest.raises(ValidationError, match="must not be blank"):
        KBEntryV2.model_validate({**row, "entry_id": "  "})
    with pytest.raises(ValidationError, match="content_sha256"):
        KBEntryV2.model_validate({**row, "content_sha256": "0" * 64})


def test_verified_catalog_is_canonical_create_only_and_recomputes_group_closure(
    tmp_path: Path,
):
    encyclopedia = [
        _entry(
            "encyclopedia-1",
            "encyclopedia",
            "熊猫正文",
            entity="animal-panda",
        ),
        _entry(
            "encyclopedia-2",
            "encyclopedia",
            "熊猫的另一条来源",
            entity="animal-panda",
            parents=["encyclopedia-1"],
        ),
    ]
    encyclopedia_path, recipe_path = _write_sources(tmp_path, encyclopedia=encyclopedia)
    output = tmp_path / "catalog"

    catalog = _build_verified(encyclopedia_path, recipe_path, output)

    assert catalog.manifest.mode == "verified"
    assert catalog.manifest.complete is True
    assert catalog.manifest.entry_count == 4
    assert (
        catalog.manifest.catalog_sha256
        == hashlib.sha256(
            canonical_json_bytes(
                catalog.manifest.model_dump(mode="json", exclude={"catalog_sha256"})
            )
        ).hexdigest()
    )
    groups = catalog.leakage_group_by_entry_id
    assert groups["encyclopedia-1"] == groups["encyclopedia-2"]
    assert catalog.manifest_bytes.endswith(b"\n")
    assert catalog.manifest_bytes == (
        canonical_json_bytes(catalog.manifest.model_dump(mode="json")) + b"\n"
    )
    assert (
        load_kb_catalog(
            output,
            expected_catalog_sha256=catalog.manifest.catalog_sha256,
        ).entries
        == catalog.entries
    )
    with pytest.raises(KBCatalogError, match="external expected catalog"):
        load_kb_catalog(output)
    with pytest.raises(KBCatalogError, match="external expected lock"):
        load_kb_catalog(output, expected_catalog_sha256="0" * 64)

    with pytest.raises(FileExistsError):
        _build_verified(encyclopedia_path, recipe_path, output)


def test_verified_catalog_requires_external_source_locks_and_rejects_rewritten_source(
    tmp_path: Path,
):
    encyclopedia_path, recipe_path = _write_sources(tmp_path)
    original_encyclopedia_sha256 = _sha256_file(encyclopedia_path)
    recipe_sha256 = _sha256_file(recipe_path)

    with pytest.raises(KBCatalogError, match="requires expected SHA-256"):
        build_kb_catalog(
            encyclopedia_path,
            recipe_path,
            tmp_path / "missing-locks",
        )

    encyclopedia_path.write_bytes(encyclopedia_path.read_bytes() + b"\n")
    with pytest.raises(KBCatalogError, match="does not match expected lock"):
        build_kb_catalog(
            encyclopedia_path,
            recipe_path,
            tmp_path / "rewritten-source",
            expected_encyclopedia_sha256=original_encyclopedia_sha256,
            expected_recipe_sha256=recipe_sha256,
        )
    assert not (tmp_path / "rewritten-source").exists()


def test_formal_catalog_rejects_unverified_entries_and_provisional_is_explicit(
    tmp_path: Path,
):
    encyclopedia_path, recipe_path = _write_sources(
        tmp_path,
        encyclopedia=[
            _entry(
                "encyclopedia-1",
                "encyclopedia",
                "未经正式验证",
                verification_status="unverified",
            ),
            _entry("encyclopedia-2", "encyclopedia", "第二条"),
        ],
    )
    with pytest.raises(KBCatalogError, match="not eligible"):
        _build_verified(encyclopedia_path, recipe_path, tmp_path / "formal")
    with pytest.raises(KBCatalogError, match="positive limit_per_kind"):
        build_kb_catalog(
            encyclopedia_path,
            recipe_path,
            tmp_path / "bad-smoke",
            allow_provisional=True,
        )

    smoke = build_kb_catalog(
        encyclopedia_path,
        recipe_path,
        tmp_path / "smoke",
        limit_per_kind=1,
        allow_provisional=True,
    )
    assert smoke.manifest.mode == "provisional"
    assert smoke.manifest.complete is False
    with pytest.raises(KBCatalogError, match="explicit opt-in"):
        load_kb_catalog(tmp_path / "smoke")
    assert (
        load_kb_catalog(tmp_path / "smoke", allow_provisional=True).manifest.entry_count
        == 2
    )


def test_allow_provisional_cannot_bypass_verified_catalog_lock(tmp_path: Path):
    encyclopedia_path, recipe_path = _write_sources(tmp_path)
    catalog = _build_verified(
        encyclopedia_path,
        recipe_path,
        tmp_path / "verified",
    )

    with pytest.raises(KBCatalogError, match="external expected catalog"):
        load_kb_catalog(tmp_path / "verified", allow_provisional=True)

    loaded = load_kb_catalog(
        tmp_path / "verified",
        allow_provisional=True,
        expected_catalog_sha256=catalog.manifest.catalog_sha256,
    )
    assert loaded.manifest.mode == "verified"


def test_catalog_rejects_duplicate_json_keys_and_duplicate_entry_ids(tmp_path: Path):
    encyclopedia_path, recipe_path = _write_sources(tmp_path)
    encyclopedia_path.write_text(
        '{"schema_version":2,"schema_version":2}\n', encoding="utf-8"
    )
    with pytest.raises(KBCatalogError, match="invalid"):
        _build_verified(encyclopedia_path, recipe_path, tmp_path / "duplicate-key")

    encyclopedia_path, recipe_path = _write_sources(
        tmp_path,
        encyclopedia=[
            _entry("shared", "encyclopedia", "百科"),
            _entry("encyclopedia-2", "encyclopedia", "另一百科"),
        ],
        recipes=[
            _entry("shared", "recipe", "菜谱"),
            _entry("recipe-2", "recipe", "另一菜谱"),
        ],
    )
    with pytest.raises(KBCatalogError, match="globally unique"):
        _build_verified(encyclopedia_path, recipe_path, tmp_path / "duplicate-id")


def test_catalog_detects_source_change_during_build(tmp_path: Path, monkeypatch):
    encyclopedia_path, recipe_path = _write_sources(tmp_path)
    original_loader = kb_catalog.load_kb_catalog
    mutated = False

    def mutate_after_staging_load(root, **kwargs):
        nonlocal mutated
        result = original_loader(root, **kwargs)
        if not mutated and ".staging-" in Path(root).name:
            encyclopedia_path.write_bytes(encyclopedia_path.read_bytes() + b"\n")
            mutated = True
        return result

    monkeypatch.setattr(kb_catalog, "load_kb_catalog", mutate_after_staging_load)
    with pytest.raises(KBCatalogError, match="changed after snapshot"):
        _build_verified(encyclopedia_path, recipe_path, tmp_path / "catalog")
    assert not (tmp_path / "catalog").exists()


def test_catalog_rejects_tampered_and_extra_artifacts(tmp_path: Path):
    encyclopedia_path, recipe_path = _write_sources(tmp_path)
    output = tmp_path / "catalog"
    catalog = _build_verified(encyclopedia_path, recipe_path, output)
    entries = output / "entries.jsonl"
    entries.write_bytes(entries.read_bytes() + b" ")
    with pytest.raises(KBCatalogError, match="artifact byte size mismatch"):
        load_kb_catalog(
            output,
            expected_catalog_sha256=catalog.manifest.catalog_sha256,
        )

    output2 = tmp_path / "catalog-2"
    catalog2 = _build_verified(encyclopedia_path, recipe_path, output2)
    (output2 / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(KBCatalogError, match="artifact set"):
        load_kb_catalog(
            output2,
            expected_catalog_sha256=catalog2.manifest.catalog_sha256,
        )


def test_catalog_recomputes_groups_instead_of_trusting_coordinated_self_hashes(
    tmp_path: Path,
):
    encyclopedia_path, recipe_path = _write_sources(tmp_path)
    output = tmp_path / "catalog"
    catalog = _build_verified(encyclopedia_path, recipe_path, output)
    groups_path = output / "groups.jsonl"
    groups = [
        json.loads(line)
        for line in groups_path.read_text(encoding="utf-8").splitlines()
    ]
    groups[0]["leakage_group_id"] = "kbgrp-coordinated-but-false"
    groups_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in groups)
    groups_path.write_bytes(groups_bytes)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["groups.jsonl"].update(
        {
            "bytes": len(groups_bytes),
            "sha256": hashlib.sha256(groups_bytes).hexdigest(),
        }
    )
    manifest["group_binding_sha256"] = hashlib.sha256(groups_bytes).hexdigest()
    unsigned = dict(manifest)
    unsigned.pop("catalog_sha256")
    manifest["catalog_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(KBCatalogError, match="external expected lock"):
        load_kb_catalog(
            output,
            expected_catalog_sha256=catalog.manifest.catalog_sha256,
        )


def test_catalog_rejects_symlink_source_when_supported(tmp_path: Path):
    encyclopedia_path, recipe_path = _write_sources(tmp_path)
    link = tmp_path / "encyclopedia-link.jsonl"
    try:
        os.symlink(encyclopedia_path, link)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(KBCatalogError, match="regular non-symlink"):
        _build_verified(link, recipe_path, tmp_path / "catalog")
