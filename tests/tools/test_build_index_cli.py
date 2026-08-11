from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from skillchain.data.kb_catalog import KBEntryV2, canonical_jsonl_bytes


def _script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_index.py"
    spec = importlib.util.spec_from_file_location("complete_tools_build_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(kind: str, entry_id: str, text: str) -> KBEntryV2:
    return KBEntryV2(
        schema_version=2,
        entry_id=entry_id,
        title=entry_id,
        text=text,
        kind=kind,
        origin="dump",
        source_dataset=f"fixture-{kind}",
        source_revision="revision-1",
        source_record_id=entry_id,
        source_uri=f"https://example.test/{kind}/{entry_id}",
        license_id="CC0-1.0",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        entity_group_id=f"entity-{entry_id}",
        verification_status="source_verified",
    )


def test_registry_cli_publishes_and_verifies_create_only_manifest(tmp_path, capsys):
    script = _script()
    output = tmp_path / "registry.json"

    assert script.main(["registry", "--output", str(output)]) == 0
    published = json.loads(capsys.readouterr().out)
    assert published["tool_count"] == 7
    assert len(published["tools"]) == 7
    assert script.main(["registry-status", "--output", str(output)]) == 0
    assert (
        json.loads(capsys.readouterr().out)["registry_sha256"]
        == published["registry_sha256"]
    )
    assert script.main(["registry", "--output", str(output)]) == 2
    assert "拒绝覆盖" in capsys.readouterr().err


def test_model_manifest_cli_binds_all_files(tmp_path, capsys):
    script = _script()
    weights = tmp_path / "weights.onnx"
    labels = tmp_path / "labels.json"
    config = tmp_path / "config.json"
    weights.write_bytes(b"model")
    labels.write_bytes(b"{}\n")
    config.write_bytes(b"{}\n")
    manifest = tmp_path / "detector.json"

    assert (
        script.main(
            [
                "model-manifest",
                "--kind",
                "object_detector",
                "--output",
                str(manifest),
                "--model-id",
                "fixture-detector",
                "--backend-name",
                "fixture",
                "--backend-version",
                "1",
                "--artifact",
                f"weights={weights}",
                "--artifact",
                f"labels={labels}",
                "--artifact",
                f"config={config}",
            ]
        )
        == 0
    )
    published = json.loads(capsys.readouterr().out)
    assert published["artifact_kind"] == "object_detector"
    assert (
        script.main(
            [
                "model-status",
                "--manifest",
                str(manifest),
                "--manifest-sha256",
                published["manifest_sha256"],
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["manifest_sha256"]
        == published["manifest_sha256"]
    )


def test_kb_cli_builds_and_verifies_one_atomic_bundle(tmp_path, capsys):
    script = _script()
    encyclopedia = tmp_path / "encyclopedia.jsonl"
    recipes = tmp_path / "recipes.jsonl"
    encyclopedia.write_bytes(
        canonical_jsonl_bytes([_entry("encyclopedia", "wiki-cotton", "棉花 植物 纺织")])
    )
    recipes.write_bytes(
        canonical_jsonl_bytes([_entry("recipe", "recipe-soup", "番茄 汤 做法")])
    )
    catalog = tmp_path / "catalog"
    bundle = tmp_path / "bundle"
    encyclopedia_sha256 = hashlib.sha256(encyclopedia.read_bytes()).hexdigest()
    recipes_sha256 = hashlib.sha256(recipes.read_bytes()).hexdigest()

    assert (
        script.main(
            [
                "kb-catalog",
                "--encyclopedia",
                str(encyclopedia),
                "--recipes",
                str(recipes),
                "--output",
                str(catalog),
            ]
        )
        == 2
    )
    assert "requires expected SHA-256" in capsys.readouterr().err

    assert (
        script.main(
            [
                "kb-catalog",
                "--encyclopedia",
                str(encyclopedia),
                "--recipes",
                str(recipes),
                "--encyclopedia-sha256",
                encyclopedia_sha256,
                "--recipes-sha256",
                recipes_sha256,
                "--output",
                str(catalog),
            ]
        )
        == 0
    )
    catalog_manifest = json.loads(capsys.readouterr().out)
    assert catalog_manifest["mode"] == "verified"
    assert (
        script.main(
            [
                "kb",
                "--catalog",
                str(catalog),
                "--catalog-sha256",
                catalog_manifest["catalog_sha256"],
                "--output",
                str(bundle),
            ]
        )
        == 0
    )
    bundle_manifest = json.loads(capsys.readouterr().out)
    assert bundle_manifest["complete"] is True
    assert (
        script.main(
            [
                "kb-status",
                "--catalog",
                str(catalog),
                "--catalog-sha256",
                catalog_manifest["catalog_sha256"],
                "--output",
                str(bundle),
                "--bundle-sha256",
                bundle_manifest["bundle_sha256"],
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["encyclopedia_entries"] == 1
    assert status["recipe_entries"] == 1
