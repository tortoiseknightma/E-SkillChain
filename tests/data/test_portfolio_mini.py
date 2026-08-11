from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from PIL import Image
import pytest

from skillchain.data.asset_catalog import (
    inventory_dataset_asset,
    publish_asset_catalog,
)
from skillchain.data import portfolio_mini
from skillchain.data.portfolio_mini import (
    assemble_portfolio_mini,
    build_portfolio_mini_capability_assignments,
    load_portfolio_mini_drafts,
)
from skillchain.synthesis.planning import load_capability_assignments
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import parse_canonical_json


def _write_image(path: Path, seed: int) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(
        "RGB",
        (16, 16),
        color=(seed % 251, (seed * 7) % 251, (seed * 19) % 251),
    )
    for index in range(16):
        image.putpixel(
            (index, (index * (seed % 13 + 1)) % 16),
            ((seed * 3) % 251, index * 11, (seed + index * 5) % 251),
        )
    image.save(path, format="PNG")
    return path.read_bytes()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _prepared_inputs(tmp_path: Path, monkeypatch):
    seed = 1

    abo_root = tmp_path / "abo"
    retained: list[str] = []
    approved: list[dict] = []
    fingerprints: list[dict] = []
    for index in range(35):
        pair_id = f"pair-{index:04d}"
        image_id = f"image-{index:04d}"
        local_path = f"images/{image_id}.png"
        content = _write_image(abo_root / local_path, seed)
        seed += 1
        retained.append(pair_id)
        approved.append(
            {
                "main_image_id": image_id,
                "main_local_path": local_path,
                "other_image_id": f"other-{index:04d}",
                "other_local_path": f"images/other-{index:04d}.png",
                "pair_id": pair_id,
            }
        )
        fingerprints.append(
            {
                "bytes": len(content),
                "image_id": image_id,
                "local_path": local_path,
                "source_image_sha256": sha256_bytes(content),
            }
        )
    unsigned_receipt = {
        "approved_pairs": approved,
        "formal_use_allowed": False,
        "image_fingerprints": fingerprints,
        "retained_pair_count": 35,
        "retained_pair_ids": retained,
        "schema_version": 1,
        "status": "local_pair_leakage_audit_complete",
    }
    receipt = {
        **unsigned_receipt,
        "receipt_self_sha256": sha256_bytes(
            canonical_json_bytes(unsigned_receipt)
        ),
    }
    abo_receipt = tmp_path / "abo-receipt.json"
    abo_receipt.write_bytes(canonical_json_bytes(receipt))

    fashion_asset_root = tmp_path / "fashion-assets"
    fashion_drafts = []
    previous_fashion_path: Path | None = None
    for index in range(28):
        local_path = f"fashioniq/images/fashion-{index:04d}.png"
        source_record_id = {
            5: "dress:B0007WFGWS",
            6: "dress:B0007WIZYE",
        }.get(index, f"dress:fixture-{index:04d}")
        image_path = fashion_asset_root / local_path
        if index == 6:
            assert previous_fashion_path is not None
            image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous_fashion_path, image_path)
        else:
            _write_image(image_path, seed)
            seed += 1
        previous_fashion_path = image_path
        fashion_drafts.append(
            portfolio_mini.DatasetAssetDraft(
                source_dataset="fashioniq",
                source_revision="fashion-test-v1",
                source_record_id=source_record_id,
                transform_policy_version="fashioniq-original-byte-reference-v1",
                local_path=local_path,
                product_id=f"fashioniq:{index:04d}",
                license_id="LicenseRef-FashionIQ-Test",
                cloud_upload_allowed=False,
                public_demo_allowed=False,
            )
        )
    fashion_adapter = tmp_path / "fashion-adapter"
    fashion_adapter.mkdir()
    fashion_manifest = canonical_json_bytes({"adapter": "fashion-test"})
    (fashion_adapter / "manifest.json").write_bytes(fashion_manifest)
    fashion_sha = sha256_bytes(fashion_manifest)
    monkeypatch.setattr(
        portfolio_mini,
        "load_verified_fashioniq_adapter_bundle",
        lambda root, *, expected_manifest_sha256: SimpleNamespace(
            drafts=tuple(fashion_drafts)
        ),
    )

    rpc_adapter = tmp_path / "rpc-adapter"
    rpc_drafts = []
    for index in range(35):
        local_path = f"{rpc_adapter.name}/images/rpc-{index:04d}.png"
        _write_image(rpc_adapter.parent / local_path, seed)
        seed += 1
        rpc_drafts.append(
            portfolio_mini.DatasetAssetDraft(
                source_dataset="rpc",
                source_revision="rpc-test-v1",
                source_record_id=f"val:{index:04d}",
                transform_policy_version="rpc-val-exact-zip-member-v1",
                local_path=local_path,
                license_id="CC-BY-NC-SA-4.0",
                cloud_upload_allowed=False,
                public_demo_allowed=False,
            )
        )
    rpc_manifest = canonical_json_bytes({"adapter": "rpc-test"})
    (rpc_adapter / "manifest.json").write_bytes(rpc_manifest)
    rpc_sha = sha256_bytes(rpc_manifest)
    monkeypatch.setattr(
        portfolio_mini,
        "load_verified_rpc_val_adapter",
        lambda root, *, expected_manifest_file_sha256: SimpleNamespace(
            drafts=tuple(rpc_drafts)
        ),
    )

    output_root = tmp_path / "clean" / "query_images"
    inat_root = output_root / "encyclopedia"
    inat_rows = []
    for index in range(27):
        filename = f"inat-{index:04d}.png"
        _write_image(inat_root / filename, seed)
        seed += 1
        inat_rows.append(
            {
                "observation_id": index + 1,
                "photo_id": index + 100,
                "scientific_name": f"Species {index}",
                "common_name": None,
                "iconic_taxon": "Plantae",
                "license": "cc-by",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": f"Photographer {index}",
                "observation_url": f"https://example.test/observations/{index}",
                "api_photo_url": f"https://example.test/square/{index}.jpg",
                "original_url": f"https://example.test/original/{index}.jpg",
                "source_url": f"https://example.test/medium/{index}.jpg",
                "image": filename,
            }
        )
    inat_manifest = inat_root / "manifest.jsonl"
    _write_jsonl(inat_manifest, inat_rows)

    document_root = output_root / "utility_docs"
    document_rows = []
    for index in range(30):
        filename = f"commons-{index:04d}.png"
        _write_image(document_root / filename, seed)
        seed += 1
        document_rows.append(
            {
                "page_id": index + 1,
                "title": f"Document {index}",
                "document_category": "Receipts",
                "width": 16,
                "height": 16,
                "mime": "image/png",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": f"Author {index}",
                "description_url": f"https://example.test/wiki/{index}",
                "original_url": f"https://example.test/original/{index}.png",
                "source_url": f"https://example.test/thumb/{index}.png",
                "image": filename,
                "source": "wikimedia-commons",
                "cloud_upload_allowed": True,
                "public_demo_allowed": False,
            }
        )
    document_manifest = document_root / "manifest.jsonl"
    _write_jsonl(document_manifest, document_rows)

    food_root = tmp_path / "food"
    food_rows = []
    for index in range(30):
        filename = f"isia-{index:04d}.png"
        _write_image(food_root / filename, seed)
        seed += 1
        food_rows.append(
            {
                "image": filename,
                "category": f"Food {index}",
                "source": "isia-food500",
                "source_member": f"archive/food-{index:04d}.png",
                "source_page": "http://example.test/isia",
                "source_archive": "http://example.test/isia.zip",
                "source_archive_sha256": "a" * 64,
                "source_volume_index": 10,
                "license": "not-specified-by-publisher",
                "license_note": "local research only",
                "distribution": "local-only-not-redistributed",
                "retrieved_at": "2026-07-10",
            }
        )
    food_manifest = food_root / "manifest.jsonl"
    _write_jsonl(food_manifest, food_rows)

    return {
        "abo_audit_receipt": abo_receipt,
        "abo_original_root": abo_root,
        "fashioniq_adapter_root": fashion_adapter,
        "fashioniq_manifest_sha256": fashion_sha,
        "fashioniq_asset_root": fashion_asset_root,
        "rpc_adapter_root": rpc_adapter,
        "rpc_manifest_sha256": rpc_sha,
        "inaturalist_manifest": inat_manifest,
        "inaturalist_asset_root": inat_root,
        "commons_document_manifest": document_manifest,
        "commons_document_asset_root": document_root,
        "isia_food_manifest": food_manifest,
        "isia_food_asset_root": food_root,
        "output_root": output_root,
    }


def test_assemble_and_assign_portfolio_mini(tmp_path: Path, monkeypatch) -> None:
    arguments = _prepared_inputs(tmp_path, monkeypatch)
    result = assemble_portfolio_mini(**arguments)

    assert result.asset_count == 184
    assert result.selection_manifest_sha256 == sha256_bytes(
        result.selection_manifest_path.read_bytes()
    )
    assert {
        directory: len(
            [
                path
                for path in (result.output_root / directory).iterdir()
                if path.suffix.lower() == ".png"
            ]
        )
        for directory in (
            "exact_match",
            "multi_product",
            "divergent_rec",
            "encyclopedia",
            "utility",
        )
    } == {
        "exact_match": 35,
        "multi_product": 35,
        "divergent_rec": 27,
        "encyclopedia": 27,
        "utility": 60,
    }
    assert (result.output_root / "encyclopedia" / "manifest.jsonl").is_file()

    drafts = load_portfolio_mini_drafts(result.dataset_assets_path)
    permissions = defaultdict(set)
    for draft in drafts:
        permissions[draft.source_dataset].add(draft.cloud_upload_allowed)
    assert permissions["abo"] == {False}
    assert permissions["fashioniq"] == {False}
    assert permissions["rpc"] == {False}
    assert permissions["inaturalist"] == {None}
    assert permissions["wikimedia_commons_documents"] == {True}
    assert permissions["isia_food500"] == {None}

    manifest = parse_canonical_json(
        result.selection_manifest_path.read_bytes(),
        label="selection manifest",
    )
    assert manifest["track"] == "portfolio"
    assert manifest["formal_status"] == "non_formal"
    assert manifest["formal_eligible"] is False
    assert manifest["pool_counts"] == {
        "encyclopedia": 27,
        "exact": 35,
        "multi": 35,
        "style": 27,
        "utility_document": 30,
        "utility_recipe": 30,
    }
    style_records = {
        row["source_record_id"]
        for row in manifest["selections"]
        if row["pool"] == "style"
    }
    assert "dress:B0007WFGWS" in style_records
    assert "dress:B0007WIZYE" not in style_records
    assert "dress:fixture-0027" in style_records

    repeated = assemble_portfolio_mini(**arguments)
    assert repeated.selection_manifest_sha256 == result.selection_manifest_sha256

    asset_root = result.output_root.parent
    assets = tuple(inventory_dataset_asset(draft, asset_root) for draft in drafts)
    catalog_dir = tmp_path / "asset-catalog"
    publish_asset_catalog(
        assets,
        catalog_dir,
        asset_root,
        coverage_roots=(
            "query_images/exact_match",
            "query_images/multi_product",
            "query_images/divergent_rec",
            "query_images/encyclopedia",
            "query_images/utility",
        ),
    )
    assignment_path = tmp_path / "capability-assignments.jsonl"
    assignment_result = build_portfolio_mini_capability_assignments(
        selection_manifest=result.selection_manifest_path,
        asset_catalog_dir=catalog_dir,
        asset_root=asset_root,
        output_path=assignment_path,
    )
    assert assignment_result.assignment_count == 254
    assignments = load_capability_assignments(
        assignment_path,
        expected_sha256=assignment_result.assignment_sha256,
    )
    by_intent = Counter(item.canonical_intent for item in assignments)
    assert by_intent == {
        "exact_match": 35,
        "multi_product": 35,
        "divergent_rec": 62,
        "encyclopedia": 62,
        "utility": 60,
    }
    exact_paths = {
        item.image_path
        for item in assignments
        if item.canonical_intent == "exact_match"
    }
    exact_cross = Counter(
        item.image_path
        for item in assignments
        if item.image_path in exact_paths
    )
    assert set(exact_cross.values()) == {3}
    assert {
        item.source_artifact_sha256 for item in assignments
    } == {result.selection_manifest_sha256}
    assert assignment_path.read_bytes() == canonical_jsonl_bytes(assignments)

    exact_image = next((result.output_root / "exact_match").glob("*.png"))
    exact_image.write_bytes(b"conflicting bytes")
    with pytest.raises(FileExistsError, match="conflicts with selection"):
        assemble_portfolio_mini(**arguments)


def test_global_duplicate_content_remains_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arguments = _prepared_inputs(tmp_path, monkeypatch)
    abo_image = arguments["abo_original_root"] / "images" / "image-0000.png"
    food_image = arguments["isia_food_asset_root"] / "isia-0000.png"
    shutil.copyfile(abo_image, food_image)

    with pytest.raises(
        portfolio_mini.PortfolioMiniError,
        match="exact duplicate image bytes",
    ):
        assemble_portfolio_mini(**arguments)
