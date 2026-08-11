from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from PIL import Image, ImageDraw
import pyarrow as pa
import pyarrow.parquet as pq

from skillchain.data.asset_catalog import DatasetAssetDraft
from skillchain.data.portfolio_core_assets import (
    CORE_R2_CAPABILITY_ASSET_FLOORS,
    CandidateCapabilityBinding,
    PortfolioCoreAssetCandidate,
    default_source_policy_bytes,
    load_candidate_inventory,
    load_portfolio_core_document_selection_manifest,
    preflight_portfolio_core_assets,
)
from skillchain.data import portfolio_core_inventory
from skillchain.data.portfolio_core_inventory import (
    build_portfolio_core_r2_candidate_catalog,
    build_regular_portfolio_core_inventory,
    build_isia_recipe_extension_candidates,
    load_isia_recipe_extension_manifest,
    materialize_portfolio_core_document_extension,
    materialize_isia_food500_recipe_extension,
    merge_portfolio_core_candidate_inventories,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


_MINI_POOLS = {
    "exact": (35, "abo", "exact_match", "product.exact_match", "exact_match"),
    "multi": (
        35,
        "rpc",
        "multi_product",
        "product.multi_search",
        "multi_product",
    ),
    "style": (
        27,
        "fashioniq",
        "divergent_rec",
        "product.style_recommendation",
        "divergent_rec",
    ),
    "encyclopedia": (
        27,
        "inaturalist",
        "encyclopedia",
        "knowledge.visual_encyclopedia",
        "encyclopedia",
    ),
    "utility_document": (
        30,
        "wikimedia_commons_documents",
        "utility",
        "utility.document_reading",
        "utility",
    ),
    "utility_recipe": (
        30,
        "isia_food500",
        "utility",
        "utility.recipe_guidance",
        "utility",
    ),
}


def _image(path: Path, seed: int) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64))
    draw = ImageDraw.Draw(image)
    for y in range(8):
        for x in range(8):
            value = (
                seed * 1103515245 + x * 12345 + y * 214013 + x * y * 2531011
            ) & 0xFFFFFFFF
            draw.rectangle(
                (x * 8, y * 8, x * 8 + 7, y * 8 + 7),
                fill=((value >> 16) & 255, (value >> 8) & 255, value & 255),
            )
    draw.line(
        (0, seed % 64, 63, (seed * 7) % 64),
        fill=((seed * 19) % 255, (seed * 31) % 255, (seed * 47) % 255),
        width=2,
    )
    image.save(path, format="PNG")
    return path.read_bytes()


def _jpeg_image_bytes(tmp_path: Path, seed: int) -> bytes:
    source = _image(tmp_path / f"source-{seed:04d}.png", seed)
    with Image.open(io.BytesIO(source)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=90)
    return output.getvalue()


def _document_extension_inputs(tmp_path: Path) -> dict[str, Path]:
    mini_manifest, dev_root = _mini_selection(tmp_path)
    cord_root = tmp_path / "cord" / portfolio_core_inventory.CORD_V2_FIXED_COMMIT
    cord_data = cord_root / "data"
    images = [_jpeg_image_bytes(tmp_path / "cord-source", 10_000 + index) for index in range(80)]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "image": pa.array(
                [
                    {"bytes": content, "path": f"receipt-{index:04d}.jpg"}
                    for index, content in enumerate(images)
                ],
                type=image_type,
            ),
            "ground_truth": pa.array(["{}"] * len(images), type=pa.string()),
        }
    )
    cord_data.mkdir(parents=True)
    pq.write_table(
        table,
        cord_data / "test-00000-of-00001-fixture.parquet",
    )
    return {
        "dev_manifest": mini_manifest,
        "dev_root": dev_root,
        "cord_root": cord_root,
    }


def _sroie_fixture(
    tmp_path: Path,
    *,
    count: int = 80,
) -> tuple[Path, Path]:
    raw_root = tmp_path / "sroie-raw"
    archive = raw_root / "sroie" / "official" / "task3-images.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for index in range(count):
            target.writestr(
                f"images/receipt-{index:04d}.jpg",
                _jpeg_image_bytes(tmp_path / "sroie-source", 20_000 + index),
            )
    manifest = tmp_path / "sroie-acquisition.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "manifest_id": "sroie-fixture-v1",
                "license_status": "pending_fixture_review",
                "official_source_url": "https://example.test/sroie",
                "artifacts": [
                    {
                        "path": "sroie/official/task3-images.zip",
                        "role": "task3_test_images",
                        "bytes": archive.stat().st_size,
                        "sha256": sha256_bytes(archive.read_bytes()),
                        "expected": {
                            "physical_members": count,
                            "logical_members_by_extension": {".jpg": count},
                            "duplicate_logical_groups": 0,
                            "conflicting_duplicate_groups": 0,
                        },
                    }
                ],
                "pair_checks": [],
            }
        )
    )
    return manifest, raw_root


def _external_jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    """Mimic deterministic source materializers without canonical key order."""

    return b"".join(
        (
            json.dumps(
                {key: row[key] for key in reversed(tuple(row))},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _mini_selection(tmp_path: Path) -> tuple[Path, Path]:
    asset_root = tmp_path / "dev-assets"
    drafts: list[DatasetAssetDraft] = []
    selections: list[dict[str, object]] = []
    seed = 1
    for pool, (quota, source, intent, capability, directory) in _MINI_POOLS.items():
        for index in range(quota):
            suffix = ".png"
            image_path = f"query_images/{directory}/{pool}-{index:04d}{suffix}"
            if source == "fashioniq" and index == 0:
                source_record_id = "dress:old-dev"
            elif source == "inaturalist":
                source_record_id = f"photo:{index + 1}"
            elif source == "wikimedia_commons_documents":
                source_record_id = f"page:{index + 1}"
            elif source == "isia_food500":
                source_record_id = f"archive/food-{index:04d}.jpg"
            else:
                source_record_id = f"{source}:dev:{index:04d}"
            draft = DatasetAssetDraft(
                source_dataset=source,
                source_revision=f"{source}-dev-v1",
                source_record_id=source_record_id,
                transform_policy_version=f"{source}-identity-v1",
                local_path=image_path,
                product_id=(f"{source}:{index:04d}" if source == "abo" else None),
                license_id=f"LicenseRef-{source}",
                source_url=f"https://example.test/{source}/{index}",
                attribution=f"Attribution {source}",
                cloud_upload_allowed=False,
                public_demo_allowed=False,
            )
            drafts.append(draft)
            if source == "wikimedia_commons_documents":
                content = _image(asset_root / image_path, seed)
                image_bytes = len(content)
                image_sha256 = sha256_bytes(content)
                seed += 1
            else:
                image_bytes = 1
                image_sha256 = hashlib.sha256(image_path.encode()).hexdigest()
            selections.append(
                {
                    "canonical_capability": capability,
                    "canonical_intent": intent,
                    "cloud_upload_allowed": draft.cloud_upload_allowed,
                    "image_bytes": image_bytes,
                    "image_path": image_path,
                    "image_sha256": image_sha256,
                    "license_id": draft.license_id,
                    "pool": pool,
                    "product_id": draft.product_id,
                    "public_demo_allowed": draft.public_demo_allowed,
                    "selection_id": f"portfolio-mini.fixture.{pool}.{index:04d}",
                    "source_artifact_sha256": "a" * 64,
                    "source_dataset": source,
                    "source_local_path": f"original/{source}/{index:04d}.png",
                    "source_record_id": draft.source_record_id,
                    "source_revision": draft.source_revision,
                    "transform_policy_version": draft.transform_policy_version,
                }
            )
    drafts_bytes = canonical_jsonl_bytes(drafts)
    output_root = tmp_path / "dev-selection"
    output_root.mkdir()
    (output_root / "dataset-assets.jsonl").write_bytes(drafts_bytes)
    manifest = {
        "asset_count": 184,
        "assignment_count_expected": 254,
        "capability_asset_counts": {},
        "dataset_assets": {
            "bytes": len(drafts_bytes),
            "path": "dataset-assets.jsonl",
            "rows": 184,
            "sha256": sha256_bytes(drafts_bytes),
        },
        "formal_eligible": False,
        "formal_status": "non_formal",
        "intent_asset_counts": {},
        "permission_policy": "preserve-source-value-no-upgrade-v1",
        "pool_counts": {key: value[0] for key, value in _MINI_POOLS.items()},
        "query_image_root": "query_images",
        "schema_version": 1,
        "selection_policy_version": "portfolio-mini-query-images-v1",
        "selections": selections,
        "source_artifacts": [],
        "track": "portfolio",
    }
    selection_path = output_root / "selection-manifest.json"
    selection_path.write_bytes(canonical_json_bytes(manifest))
    return selection_path, asset_root


def _regular_sources(tmp_path: Path, monkeypatch, *, document_count: int = 30):
    seed = 1000
    fashion_root = tmp_path / "fashion"
    fashion_drafts: list[DatasetAssetDraft] = []
    for index in range(131):
        local_path = f"images/fashion-{index:04d}.png"
        _image(fashion_root / local_path, seed)
        seed += 1
        fashion_drafts.append(
            DatasetAssetDraft(
                source_dataset="fashioniq",
                source_revision="fashion-core-v1",
                source_record_id=(
                    "dress:old-dev" if index == 0 else f"dress:core-{index:04d}"
                ),
                transform_policy_version="fashioniq-original-byte-reference-v1",
                local_path=local_path,
                product_id=f"fashioniq:{index:04d}",
                license_id="LicenseRef-FashionIQ",
                cloud_upload_allowed=False,
                public_demo_allowed=False,
            )
        )
    monkeypatch.setattr(
        portfolio_core_inventory,
        "load_verified_fashioniq_adapter_bundle",
        lambda root, *, expected_manifest_sha256: SimpleNamespace(
            drafts=tuple(fashion_drafts)
        ),
    )

    inat_root = tmp_path / "inat"
    inat_rows: list[dict[str, object]] = []
    for index in range(149):
        image_name = f"inat-{index:04d}.png"
        _image(inat_root / image_name, seed)
        seed += 1
        inat_rows.append(
            {
                "api_photo_url": f"https://example.test/api/{index}",
                "attribution": f"iNaturalist photographer {index}",
                "cloud_upload_allowed": False,
                "common_name": None,
                "iconic_taxon": "Plantae",
                "image": image_name,
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "observation_id": index + 1,
                "observation_url": f"https://example.test/obs/{index}",
                "original_url": f"https://example.test/original/{index}",
                "photo_id": index + 1,
                "public_demo_allowed": False,
                "scientific_name": f"Species {index}",
                "source_url": f"https://example.test/source/{index}",
            }
        )
    inat_manifest = tmp_path / "inat.jsonl"
    inat_manifest.write_bytes(_external_jsonl_bytes(inat_rows))

    isia_root = tmp_path / "isia"
    isia_rows: list[dict[str, object]] = []
    for index in range(146):
        image_name = f"isia-{index:04d}.png"
        _image(isia_root / image_name, seed)
        seed += 1
        isia_rows.append(
            {
                "cloud_upload_allowed": False,
                "category": f"food-{index % 5}",
                "distribution": "local approved archive",
                "image": image_name,
                "license": "ISIA-Food500-Research",
                "license_note": f"ISIA Food-500 attribution {index}",
                "public_demo_allowed": False,
                "retrieved_at": "2026-08-04T00:00:00Z",
                "source": "isia_food500",
                "source_archive": "ISIA_Food500.zip",
                "source_archive_sha256": "b" * 64,
                "source_member": (
                    f"archive/food-{index:04d}.jpg"
                    if index < 30
                    else f"archive/core-{index:04d}.jpg"
                ),
                "source_page": "https://example.test/isia-food500",
                "source_volume_index": 1,
            }
        )
    isia_manifest = tmp_path / "isia.jsonl"
    isia_manifest.write_bytes(_external_jsonl_bytes(isia_rows))

    document_root = tmp_path / "documents"
    document_rows: list[dict[str, object]] = []
    for index in range(document_count):
        image_name = f"document-{index:04d}.png"
        _image(document_root / image_name, index + 1)
        document_rows.append(
            {
                "attribution": f"Commons contributor {index}",
                "cloud_upload_allowed": False,
                "description_url": f"https://commons.test/page/{index}",
                "document_category": "receipt",
                "height": 8,
                "image": image_name,
                "license": "CC-BY-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "mime": "image/png",
                "original_url": f"https://commons.test/original/{index}",
                "page_id": index + 1,
                "public_demo_allowed": False,
                "source": "wikimedia_commons",
                "source_url": f"https://commons.test/source/{index}",
                "title": f"Document {index}",
                "width": 8,
            }
        )
    document_manifest = tmp_path / "documents.jsonl"
    document_manifest.write_bytes(_external_jsonl_bytes(document_rows))
    return {
        "fashion_root": fashion_root,
        "inat_manifest": inat_manifest,
        "inat_root": inat_root,
        "isia_manifest": isia_manifest,
        "isia_root": isia_root,
        "document_manifest": document_manifest,
        "document_root": document_root,
    }


def _synthetic_inventory(
    path: Path,
    *,
    source_id: str,
    pool: str,
    intent: str,
    capability: str,
    count: int,
) -> None:
    rows: list[PortfolioCoreAssetCandidate] = []
    for index in range(count):
        local_path = f"query_images/{pool}/{source_id}-core-{index:04d}.png"
        digest = hashlib.sha256(f"{source_id}:{index}".encode()).hexdigest()
        draft = DatasetAssetDraft(
            source_dataset=source_id,
            source_revision=f"{source_id}-fixture-v1",
            source_record_id=f"{source_id}:{index:04d}",
            local_path=local_path,
            license_id=f"LicenseRef-{source_id}",
            cloud_upload_allowed=False,
            public_demo_allowed=False,
        )
        rows.append(
            PortfolioCoreAssetCandidate(
                candidate_id=f"{source_id}.core.fixture-{index:04d}",
                source_id=source_id,
                pool=pool,
                source_local_path=local_path,
                destination_path=local_path,
                expected_bytes=1,
                expected_sha256=digest,
                draft=draft,
                capability_bindings=(
                    CandidateCapabilityBinding(
                        canonical_intent=intent,
                        canonical_capability=capability,
                    ),
                ),
            )
        )
    path.write_bytes(canonical_jsonl_bytes(rows))


def _recipe_zip_image(index: int) -> bytes:
    image = Image.new("RGB", (280, 260), color=(251, 251, 251))
    draw = ImageDraw.Draw(image)
    x = (index * 17) % 210
    y = (index * 29) % 180
    draw.rectangle((x, 15, x + 55, 245), fill=(index * 31 % 255, 10, 40))
    draw.ellipse((25, y, 245, y + 45), fill=(20, index * 19 % 255, 180))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=91)
    return output.getvalue()


def _isia_recipe_extension_inputs(tmp_path: Path) -> dict[str, Path]:
    archive = tmp_path / "ISIA_Food500.zip"
    members: list[tuple[str, str, bytes]] = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for index in range(16):
            category = "Alpha" if index % 2 == 0 else "Beta"
            member = f"ISIA_Food500/images/{category}/food-{index:04d}.jpg"
            content = _recipe_zip_image(index)
            target.writestr(member, content)
            members.append((member, category, content))
    archive_sha256 = sha256_bytes(archive.read_bytes())

    carry_root = tmp_path / "isia-carry"
    carry_root.mkdir()
    rows: list[dict[str, object]] = []
    for index, (member, category, content) in enumerate(members[:4], start=1):
        image_name = f"isia-{index:04d}.jpg"
        (carry_root / image_name).write_bytes(content)
        rows.append(
            {
                "category": category,
                "cloud_upload_allowed": False,
                "distribution": "local-only-not-redistributed",
                "image": image_name,
                "license": "not-specified-by-publisher",
                "license_note": "Fixture local-only use",
                "public_demo_allowed": False,
                "retrieved_at": "2026-08-04",
                "source": "isia_food500",
                "source_archive": "fixture://ISIA_Food500.zip",
                "source_archive_sha256": archive_sha256,
                "source_member": member,
                "source_page": "https://example.test/isia",
                "source_volume_index": 10,
            }
        )
    carry_manifest = tmp_path / "isia-carry-manifest.jsonl"
    carry_manifest.write_bytes(_external_jsonl_bytes(rows))
    return {
        "archive": archive,
        "carry_manifest": carry_manifest,
        "carry_root": carry_root,
    }


def test_builds_fresh_regular_inventory_and_carries_document_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mini_manifest, _mini_root = _mini_selection(tmp_path)
    sources = _regular_sources(tmp_path, monkeypatch)
    recipe_root = tmp_path / "recipe"
    _image(recipe_root / "images" / "recipe-a.jpg", 3001)
    _image(recipe_root / "images" / "recipe-b.jpg", 3002)
    recipe_inventory = recipe_root / "inventory.json"
    recipe_inventory.write_bytes(
        canonical_json_bytes(
            {
                "entities": [
                    {
                        "entity_id": "entity-a",
                        "recipes": [
                            {
                                "image_ids": ["recipe-a.jpg", "recipe-b.jpg"],
                                "recipe_id": "recipe-a",
                            }
                        ],
                    }
                ],
                "image_count": 2,
                "schema_version": 1,
            }
        )
    )
    output = tmp_path / "regular-candidates.jsonl"
    result = build_regular_portfolio_core_inventory(
        dev_mini_selection_manifest=mini_manifest,
        fashioniq_adapter_root=tmp_path / "verified-fashion-adapter",
        fashioniq_manifest_sha256="f" * 64,
        fashioniq_asset_root=sources["fashion_root"],
        inaturalist_manifest=sources["inat_manifest"],
        inaturalist_asset_root=sources["inat_root"],
        commons_document_manifest=sources["document_manifest"],
        commons_document_asset_root=sources["document_root"],
        isia_food_manifest=sources["isia_manifest"],
        isia_food_asset_root=sources["isia_root"],
        recipe1m_plus_inventory=recipe_inventory,
        recipe1m_plus_root=recipe_root,
        recipe1m_plus_license_id="OwnerApproved-Recipe1MPlus-Local-Portfolio",
        recipe1m_plus_attribution="Recipe1M+ approved bounded selection",
        output_path=output,
    )

    assert result.include_count == 398
    assert result.reserve_count == 2
    assert result.source_counts == {
        "fashioniq": 130,
        "inaturalist": 122,
        "isia_food500": 116,
        "wikimedia_commons_documents": 30,
    }
    assert result.pool_unique_content_counts == {
        "divergent_rec": 130,
        "encyclopedia": 122,
        "utility": 146,
    }
    candidates, digest = load_candidate_inventory(output)
    assert digest == result.output_sha256
    assert all(
        item.draft.source_record_id != "dress:old-dev"
        for item in candidates
        if item.source_id == "fashioniq"
    )
    isia = [item for item in candidates if item.source_id == "isia_food500"]
    assert len(isia) == 116
    assert all(
        item.capability_bindings[0].canonical_capability
        == "utility.recipe_guidance"
        for item in isia
    )
    assert all(item.draft.source_url for item in isia)
    inaturalist_revision = (
        "manifest-" + sha256_bytes(sources["inat_manifest"].read_bytes())
    )
    assert {
        item.draft.source_revision
        for item in candidates
        if item.source_id == "inaturalist"
    } == {inaturalist_revision}
    documents = [
        item
        for item in candidates
        if item.source_id == "wikimedia_commons_documents"
    ]
    assert len(documents) == 30
    assert all(
        item.capability_bindings[0].canonical_capability
        == "utility.document_reading"
        for item in documents
    )
    reserves = [item for item in candidates if item.selection == "reserve"]
    assert len(reserves) == 2
    assert {item.source_id for item in reserves} == {"recipe1m_plus"}

    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(default_source_policy_bytes())
    preflight = preflight_portfolio_core_assets(
        inventory_path=output,
        source_policy_path=policy_path,
        source_roots={
            "fashioniq": sources["fashion_root"],
            "inaturalist": sources["inat_root"],
            "isia_food500": sources["isia_root"],
            "wikimedia_commons_documents": sources["document_root"],
        },
    )
    assert preflight.ready_candidate_count == 398
    assert preflight.blocked_candidates == {}
    assert preflight.capacity_shortfalls == {
        "exact_match": 157,
        "multi_product": 105,
    }


def test_expanded_document_inventory_binds_carry_forward_fresh_and_r2_capacity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mini_manifest, _mini_root = _mini_selection(tmp_path)
    sources = _regular_sources(tmp_path, monkeypatch, document_count=95)
    output = tmp_path / "regular-r2-documents.jsonl"
    document_selection = tmp_path / "document-selection.json"

    result = build_regular_portfolio_core_inventory(
        dev_mini_selection_manifest=mini_manifest,
        fashioniq_adapter_root=tmp_path / "verified-fashion-adapter",
        fashioniq_manifest_sha256="f" * 64,
        fashioniq_asset_root=sources["fashion_root"],
        inaturalist_manifest=sources["inat_manifest"],
        inaturalist_asset_root=sources["inat_root"],
        commons_document_manifest=sources["document_manifest"],
        commons_document_asset_root=sources["document_root"],
        isia_food_manifest=sources["isia_manifest"],
        isia_food_asset_root=sources["isia_root"],
        output_path=output,
        document_fresh_count=60,
        document_reserve_count=5,
        document_selection_manifest_path=document_selection,
    )

    assert result.include_count == 458
    assert result.reserve_count == 5
    assert result.source_counts["wikimedia_commons_documents"] == 90
    assert result.document_selection_manifest_path == document_selection.absolute()
    assert result.document_selection_manifest_sha256

    manifest, manifest_sha256 = load_portfolio_core_document_selection_manifest(
        document_selection
    )
    assert manifest_sha256 == result.document_selection_manifest_sha256
    by_role = {}
    for role in ("carry_forward", "fresh", "reserve"):
        by_role[role] = [
            row for row in manifest.selections if row.selection_role == role
        ]
    assert {role: len(rows) for role, rows in by_role.items()} == {
        "carry_forward": 30,
        "fresh": 60,
        "reserve": 5,
    }
    assert {
        row.source_record_id for row in by_role["carry_forward"]
    } == {f"page:{index}" for index in range(1, 31)}
    for field in ("source_record_id", "content_sha256", "component_key"):
        values = [
            getattr(row, field)
            for rows in by_role.values()
            for row in rows
        ]
        assert len(values) == len(set(values))

    candidates, _ = load_candidate_inventory(output)
    documents = [
        candidate
        for candidate in candidates
        if candidate.source_id == "wikimedia_commons_documents"
    ]
    assert len(documents) == 95
    assert len([item for item in documents if item.selection == "include"]) == 90
    assert len([item for item in documents if item.selection == "reserve"]) == 5
    assert {
        item.destination_path.rsplit("-", 1)[-1].split(".", 1)[0]
        for item in documents
    } == {f"{index:04d}" for index in range(1, 96)}

    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(default_source_policy_bytes())
    preflight = preflight_portfolio_core_assets(
        inventory_path=output,
        source_policy_path=policy_path,
        source_roots={
            "fashioniq": sources["fashion_root"],
            "inaturalist": sources["inat_root"],
            "isia_food500": sources["isia_root"],
            "wikimedia_commons_documents": sources["document_root"],
        },
        capacity_profile="r2",
        document_selection_manifest_path=document_selection,
    )
    assert preflight.source_ready
    assert not preflight.capacity_ready
    assert preflight.document_selection_counts == {
        "carry_forward": 30,
        "fresh": 60,
        "reserve": 5,
    }
    assert preflight.document_selection_role_source_counts == {
        "wikimedia_commons_documents:carry_forward": 30,
        "wikimedia_commons_documents:fresh": 60,
        "wikimedia_commons_documents:reserve": 5,
    }
    assert preflight.document_selection_shortfalls == {
        "fresh": 60,
        "fresh_cord": 30,
        "fresh_sroie": 30,
        "reserve_cord": 5,
        "reserve_sroie": 5,
    }
    assert preflight.capability_capacity_shortfalls == {
        "knowledge.visual_encyclopedia": 30,
        "product.exact_match": 340,
        "product.multi_search": 105,
        "product.style_recommendation": 20,
        "utility.recipe_guidance": 94,
    }
    assert (
        CORE_R2_CAPABILITY_ASSET_FLOORS["utility.document_reading"]
        not in preflight.capability_capacity_shortfalls.values()
    )

    repeated = build_regular_portfolio_core_inventory(
        dev_mini_selection_manifest=mini_manifest,
        fashioniq_adapter_root=tmp_path / "verified-fashion-adapter",
        fashioniq_manifest_sha256="f" * 64,
        fashioniq_asset_root=sources["fashion_root"],
        inaturalist_manifest=sources["inat_manifest"],
        inaturalist_asset_root=sources["inat_root"],
        commons_document_manifest=sources["document_manifest"],
        commons_document_asset_root=sources["document_root"],
        isia_food_manifest=sources["isia_manifest"],
        isia_food_asset_root=sources["isia_root"],
        output_path=output,
        document_fresh_count=60,
        document_reserve_count=5,
        document_selection_manifest_path=document_selection,
    )
    assert repeated.output_sha256 == result.output_sha256
    assert (
        repeated.document_selection_manifest_sha256
        == result.document_selection_manifest_sha256
    )


def test_document_extension_cord_checkpoint_is_create_only_and_reports_sroie_gap(
    tmp_path: Path,
) -> None:
    inputs = _document_extension_inputs(tmp_path)
    output = tmp_path / "documents-cord-checkpoint"

    result = materialize_portfolio_core_document_extension(
        dev_mini_selection_manifest=inputs["dev_manifest"],
        dev_mini_asset_root=inputs["dev_root"],
        cord_root=inputs["cord_root"],
        sroie_acquisition_manifest=tmp_path / "not-used.json",
        sroie_raw_root=tmp_path / "not-used-raw",
        output_root=output,
        sroie_fresh_count=0,
        sroie_reserve_count=0,
    )

    assert result.source_counts == {
        "cord": 35,
        "wikimedia_commons_documents": 30,
    }
    assert result.role_counts == {
        "carry_forward": 30,
        "fresh_cord": 30,
        "reserve": 5,
    }
    candidates, digest = load_candidate_inventory(result.candidate_inventory_path)
    assert digest == result.candidate_inventory_sha256
    assert len(candidates) == 65
    assert sum(item.selection == "include" for item in candidates) == 60
    assert sum(item.selection == "reserve" for item in candidates) == 5

    manifest, manifest_sha256 = load_portfolio_core_document_selection_manifest(
        result.selection_manifest_path
    )
    assert manifest_sha256 == result.selection_manifest_sha256
    assert {row.source_id for row in manifest.selections} == {
        "cord",
        "wikimedia_commons_documents",
    }
    assert {row.selection_role for row in manifest.selections} == {
        "carry_forward",
        "fresh_cord",
        "reserve",
    }

    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(default_source_policy_bytes())
    preflight = preflight_portfolio_core_assets(
        inventory_path=result.candidate_inventory_path,
        source_policy_path=policy_path,
        source_roots={
            "cord": output,
            "wikimedia_commons_documents": output,
        },
        capacity_profile="r2",
        document_selection_manifest_path=result.selection_manifest_path,
    )
    assert preflight.source_ready
    assert preflight.document_selection_shortfalls == {
        "fresh": 30,
        "fresh_sroie": 30,
        "reserve_sroie": 5,
    }
    assert preflight.document_selection_role_source_counts == {
        "cord:fresh_cord": 30,
        "cord:reserve": 5,
        "wikimedia_commons_documents:carry_forward": 30,
    }

    repeated = materialize_portfolio_core_document_extension(
        dev_mini_selection_manifest=inputs["dev_manifest"],
        dev_mini_asset_root=inputs["dev_root"],
        cord_root=inputs["cord_root"],
        sroie_acquisition_manifest=tmp_path / "not-used.json",
        sroie_raw_root=tmp_path / "not-used-raw",
        output_root=output,
        sroie_fresh_count=0,
        sroie_reserve_count=0,
    )
    assert repeated.candidate_inventory_sha256 == result.candidate_inventory_sha256
    assert repeated.selection_manifest_sha256 == result.selection_manifest_sha256


def test_document_extension_uses_cord_and_sroie_image_only_sources(
    tmp_path: Path,
) -> None:
    inputs = _document_extension_inputs(tmp_path)
    sroie_manifest, sroie_raw_root = _sroie_fixture(tmp_path)
    output = tmp_path / "documents-r2"

    result = materialize_portfolio_core_document_extension(
        dev_mini_selection_manifest=inputs["dev_manifest"],
        dev_mini_asset_root=inputs["dev_root"],
        cord_root=inputs["cord_root"],
        sroie_acquisition_manifest=sroie_manifest,
        sroie_raw_root=sroie_raw_root,
        output_root=output,
    )

    assert result.source_counts == {
        "cord": 35,
        "sroie": 35,
        "wikimedia_commons_documents": 30,
    }
    assert result.role_counts == {
        "carry_forward": 30,
        "fresh_cord": 30,
        "fresh_sroie": 30,
        "reserve": 10,
    }
    candidates, digest = load_candidate_inventory(result.candidate_inventory_path)
    assert digest == result.candidate_inventory_sha256
    assert len(candidates) == 100
    assert sum(item.selection == "include" for item in candidates) == 90
    assert sum(item.selection == "reserve" for item in candidates) == 10
    assert {item.source_id for item in candidates} == {
        "cord",
        "sroie",
        "wikimedia_commons_documents",
    }

    manifest, manifest_sha256 = load_portfolio_core_document_selection_manifest(
        result.selection_manifest_path
    )
    assert manifest_sha256 == result.selection_manifest_sha256
    assert manifest.source_manifest_sha256.keys() == {
        "cord",
        "sroie",
        "wikimedia_commons_documents",
    }
    assert len({row.source_record_id for row in manifest.selections}) == 100
    assert len({row.content_sha256 for row in manifest.selections}) == 100
    assert len({row.component_fingerprint for row in manifest.selections}) == 100

    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(default_source_policy_bytes())
    preflight = preflight_portfolio_core_assets(
        inventory_path=result.candidate_inventory_path,
        source_policy_path=policy_path,
        source_roots={
            "cord": output,
            "sroie": output,
            "wikimedia_commons_documents": output,
        },
        capacity_profile="r2",
        document_selection_manifest_path=result.selection_manifest_path,
    )
    assert preflight.source_ready
    assert preflight.document_selection_shortfalls == {}
    assert preflight.document_selection_counts == {
        "carry_forward": 30,
        "fresh_cord": 30,
        "fresh_sroie": 30,
        "reserve": 10,
    }
    assert preflight.document_selection_role_source_counts == {
        "cord:fresh_cord": 30,
        "cord:reserve": 5,
        "sroie:fresh_sroie": 30,
        "sroie:reserve": 5,
        "wikimedia_commons_documents:carry_forward": 30,
    }
    assert "utility.document_reading" not in preflight.capability_capacity_shortfalls


def test_isia_recipe_extension_carries_existing_bindings_and_fills_selected_tail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mini_manifest, _mini_root = _mini_selection(tmp_path)
    inputs = _isia_recipe_extension_inputs(tmp_path)
    output_root = tmp_path / "isia-r2-extension"
    extension = materialize_isia_food500_recipe_extension(
        dev_mini_selection_manifest=mini_manifest,
        carry_forward_manifest=inputs["carry_manifest"],
        carry_forward_asset_root=inputs["carry_root"],
        raw_archive_path=inputs["archive"],
        output_root=output_root,
        include_count=10,
        reserve_count=2,
        carry_forward_count=4,
        expected_archive_sha256=sha256_bytes(inputs["archive"].read_bytes()),
        expected_archive_bytes=inputs["archive"].stat().st_size,
    )
    assert extension.carry_forward_count == 4
    assert extension.fresh_count == 6
    assert extension.reserve_count == 2
    extension_manifest, extension_sha256 = load_isia_recipe_extension_manifest(
        extension.selection_manifest_path
    )
    assert extension_sha256 == extension.selection_manifest_sha256
    assert {row.selection_role for row in extension_manifest.selections} == {
        "carry_forward",
        "fresh",
        "reserve",
    }
    assert {
        role: sum(
            row.selection_role == role for row in extension_manifest.selections
        )
        for role in ("carry_forward", "fresh", "reserve")
    } == {"carry_forward": 4, "fresh": 6, "reserve": 2}
    for index in range(1, 5):
        assert (output_root / f"isia-{index:04d}.jpg").read_bytes() == (
            inputs["carry_root"] / f"isia-{index:04d}.jpg"
        ).read_bytes()
    assert len(list(output_root.glob("*.jpg"))) == 12

    repeated = materialize_isia_food500_recipe_extension(
        dev_mini_selection_manifest=mini_manifest,
        carry_forward_manifest=inputs["carry_manifest"],
        carry_forward_asset_root=inputs["carry_root"],
        raw_archive_path=inputs["archive"],
        output_root=output_root,
        include_count=10,
        reserve_count=2,
        carry_forward_count=4,
        expected_archive_sha256=sha256_bytes(inputs["archive"].read_bytes()),
        expected_archive_bytes=inputs["archive"].stat().st_size,
    )
    assert repeated.selection_manifest_sha256 == extension.selection_manifest_sha256

    recipe_candidates_path = tmp_path / "isia-r2-candidates.jsonl"
    recipe_candidate_result = build_isia_recipe_extension_candidates(
        dev_mini_selection_manifest=mini_manifest,
        staging_manifest=extension.output_staging_manifest_path,
        staging_asset_root=extension.output_root,
        extension_manifest=extension.selection_manifest_path,
        output_path=recipe_candidates_path,
    )
    assert recipe_candidate_result.include_count == 10
    assert recipe_candidate_result.reserve_count == 2
    direct_recipe_candidates, _ = load_candidate_inventory(recipe_candidates_path)
    assert len(direct_recipe_candidates) == 12

    sources = _regular_sources(tmp_path, monkeypatch)
    inventory_path = tmp_path / "regular-with-isia-extension.jsonl"
    inventory = build_regular_portfolio_core_inventory(
        dev_mini_selection_manifest=mini_manifest,
        fashioniq_adapter_root=tmp_path / "verified-fashion-adapter",
        fashioniq_manifest_sha256="f" * 64,
        fashioniq_asset_root=sources["fashion_root"],
        inaturalist_manifest=sources["inat_manifest"],
        inaturalist_asset_root=sources["inat_root"],
        commons_document_manifest=sources["document_manifest"],
        commons_document_asset_root=sources["document_root"],
        isia_food_manifest=extension.output_staging_manifest_path,
        isia_food_asset_root=extension.output_root,
        isia_food_quota=10,
        isia_food_selection_manifest=extension.selection_manifest_path,
        output_path=inventory_path,
    )
    assert inventory.source_counts["isia_food500"] == 10
    candidates, _ = load_candidate_inventory(inventory_path)
    recipe_candidates = [
        row for row in candidates if row.source_id == "isia_food500"
    ]
    assert len([row for row in recipe_candidates if row.selection == "include"]) == 10
    assert len([row for row in recipe_candidates if row.selection == "reserve"]) == 2


def test_merge_reaches_all_core_pool_floors_deterministically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mini_manifest, _mini_root = _mini_selection(tmp_path)
    sources = _regular_sources(tmp_path, monkeypatch)
    regular = tmp_path / "regular.jsonl"
    build_regular_portfolio_core_inventory(
        dev_mini_selection_manifest=mini_manifest,
        fashioniq_adapter_root=tmp_path / "verified-fashion-adapter",
        fashioniq_manifest_sha256="f" * 64,
        fashioniq_asset_root=sources["fashion_root"],
        inaturalist_manifest=sources["inat_manifest"],
        inaturalist_asset_root=sources["inat_root"],
        commons_document_manifest=sources["document_manifest"],
        commons_document_asset_root=sources["document_root"],
        isia_food_manifest=sources["isia_manifest"],
        isia_food_asset_root=sources["isia_root"],
        output_path=regular,
    )
    abo = tmp_path / "abo.jsonl"
    rpc = tmp_path / "rpc.jsonl"
    _synthetic_inventory(
        abo,
        source_id="abo",
        pool="exact_match",
        intent="exact_match",
        capability="product.exact_match",
        count=157,
    )
    _synthetic_inventory(
        rpc,
        source_id="rpc",
        pool="multi_product",
        intent="multi_product",
        capability="product.multi_search",
        count=105,
    )
    output = tmp_path / "merged.jsonl"
    result = merge_portfolio_core_candidate_inventories(
        inventory_paths=(rpc, regular, abo),
        output_path=output,
    )
    assert result.include_count == 660
    assert result.reserve_count == 0
    assert result.pool_unique_content_counts == {
        "divergent_rec": 130,
        "encyclopedia": 122,
        "exact_match": 157,
        "multi_product": 105,
        "utility": 146,
    }
    repeated = merge_portfolio_core_candidate_inventories(
        inventory_paths=(abo, rpc, regular),
        output_path=output,
    )
    assert repeated.output_sha256 == result.output_sha256


def test_r2_catalog_replaces_legacy_utility_sources_and_adds_unique_abo_triplets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The v5 catalog adds bindings only; it neither rewrites v4 nor adds images."""

    def candidate(
        *,
        source_id: str,
        index: int,
        pool: str,
        bindings: tuple[CandidateCapabilityBinding, ...],
    ) -> PortfolioCoreAssetCandidate:
        destination = f"query_images/{pool}/{source_id}-{index:04d}.png"
        return PortfolioCoreAssetCandidate(
            candidate_id=f"{source_id}.core.r2-{index:04d}",
            source_id=source_id,
            pool=pool,
            source_local_path=destination,
            destination_path=destination,
            expected_bytes=1,
            expected_sha256=hashlib.sha256(destination.encode()).hexdigest(),
            draft=DatasetAssetDraft(
                source_dataset=source_id,
                source_revision="fixture-r2-v1",
                source_record_id=f"{source_id}:record:{index:04d}",
                local_path=destination,
                license_id=f"LicenseRef-{source_id}",
                cloud_upload_allowed=False,
                public_demo_allowed=False,
            ),
            capability_bindings=bindings,
        )

    exact = CandidateCapabilityBinding(
        canonical_intent="exact_match",
        canonical_capability="product.exact_match",
    )
    style = CandidateCapabilityBinding(
        canonical_intent="divergent_rec",
        canonical_capability="product.style_recommendation",
    )
    encyclopedia = CandidateCapabilityBinding(
        canonical_intent="encyclopedia",
        canonical_capability="knowledge.visual_encyclopedia",
    )
    document = CandidateCapabilityBinding(
        canonical_intent="utility",
        canonical_capability="utility.document_reading",
    )
    recipe = CandidateCapabilityBinding(
        canonical_intent="utility",
        canonical_capability="utility.recipe_guidance",
    )
    abo_cross = candidate(
        source_id="abo",
        index=0,
        pool="exact_match",
        bindings=(exact, style, encyclopedia),
    )
    abo_exact = [
        candidate(
            source_id="abo",
            index=index,
            pool="exact_match",
            bindings=(exact,),
        )
        for index in range(1, 8)
    ]
    legacy_recipe = candidate(
        source_id="isia_food500",
        index=1,
        pool="utility",
        bindings=(recipe,),
    )
    legacy_document = candidate(
        source_id="wikimedia_commons_documents",
        index=1,
        pool="utility",
        bindings=(document,),
    )
    retained_rpc = candidate(
        source_id="rpc",
        index=1,
        pool="multi_product",
        bindings=(
            CandidateCapabilityBinding(
                canonical_intent="multi_product",
                canonical_capability="product.multi_search",
            ),
        ),
    )
    retained_fashion_value = candidate(
        source_id="fashioniq",
        index=1,
        pool="divergent_rec",
        bindings=(style,),
    ).model_dump(mode="json")
    retained_fashion_value["source_local_path"] = "fashioniq/images/legacy.png"
    retained_fashion_value["draft"]["local_path"] = "fashioniq/images/legacy.png"
    retained_fashion = PortfolioCoreAssetCandidate.model_validate(
        retained_fashion_value,
        strict=True,
    )
    v4_rows = [
        abo_cross,
        *abo_exact,
        retained_rpc,
        retained_fashion,
        legacy_recipe,
        legacy_document,
    ]
    v4_path = tmp_path / "v4.jsonl"
    v4_content = canonical_jsonl_bytes(v4_rows)
    v4_path.write_bytes(v4_content)

    r2_recipe = candidate(
        source_id="isia_food500",
        index=2,
        pool="utility",
        bindings=(recipe,),
    )
    recipe_path = tmp_path / "recipe-r2.jsonl"
    recipe_path.write_bytes(canonical_jsonl_bytes([r2_recipe]))
    r2_documents = [
        candidate(
            source_id=source_id,
            index=2,
            pool="utility",
            bindings=(document,),
        )
        for source_id in ("wikimedia_commons_documents", "cord", "sroie")
    ]
    document_path = tmp_path / "documents-r2.jsonl"
    document_path.write_bytes(canonical_jsonl_bytes(r2_documents))

    component_by_candidate = {
        abo_cross.candidate_id: "component-existing-cross",
        abo_exact[0].candidate_id: "component-repeated",
        abo_exact[1].candidate_id: "component-repeated",
        abo_exact[2].candidate_id: "component-0003",
        abo_exact[3].candidate_id: "component-0004",
        abo_exact[4].candidate_id: "component-0005",
        abo_exact[5].candidate_id: "component-0006",
        abo_exact[6].candidate_id: "component-0007",
    }
    fake_assets = [
        SimpleNamespace(
            local_path=row.destination_path,
            sha256=row.expected_sha256,
            near_duplicate_cluster_id=component_by_candidate[row.candidate_id],
        )
        for row in (abo_cross, *abo_exact)
    ]
    fake_catalog = SimpleNamespace(
        assets=tuple(fake_assets),
        manifest=SimpleNamespace(catalog_sha256="c" * 64),
    )
    monkeypatch.setattr(
        portfolio_core_inventory,
        "load_asset_catalog",
        lambda *_args, **_kwargs: fake_catalog,
    )

    output = tmp_path / "v5.jsonl"
    selection = tmp_path / "v5-selection.json"
    result = build_portfolio_core_r2_candidate_catalog(
        v4_inventory_path=v4_path,
        recipe_inventory_path=recipe_path,
        document_inventory_path=document_path,
        v4_asset_catalog_dir=tmp_path / "v4-catalog",
        v4_asset_root=tmp_path / "v4-assets",
        output_path=output,
        selection_manifest_path=selection,
        superseded_v5_inventory_sha256="a" * 64,
        superseded_v5_selection_manifest_sha256="b" * 64,
    )

    assert v4_path.read_bytes() == v4_content
    assert result.selected_abo_candidate_ids == tuple(
        row.candidate_id for row in abo_exact[2:]
    )
    merged, merged_sha256 = load_candidate_inventory(output)
    assert merged_sha256 == result.output_sha256
    assert len(merged) == 14
    assert {row.candidate_id for row in merged} == {
        *(
            row.candidate_id
            for row in (abo_cross, *abo_exact, retained_rpc, retained_fashion)
        ),
        r2_recipe.candidate_id,
        *(row.candidate_id for row in r2_documents),
    }
    rebased_fashion = next(
        row for row in merged if row.candidate_id == retained_fashion.candidate_id
    )
    assert rebased_fashion.source_local_path == rebased_fashion.destination_path
    assert rebased_fashion.draft.local_path == rebased_fashion.destination_path
    for row in merged:
        if row.candidate_id in result.selected_abo_candidate_ids:
            assert row.capability_bindings == (style, encyclopedia, exact)

    selection_value = json.loads(selection.read_text(encoding="utf-8"))
    assert selection_value["output_inventory_sha256"] == result.output_sha256
    assert selection_value["selection_rule"]["component_member_count"] == 1
    assert selection_value["v4_rebased_source_ids"] == ["fashioniq"]
    assert selection_value["superseded_v5"] == {
        "inventory_sha256": "a" * 64,
        "repair_reason": (
            "v5 source preflight could not resolve retained FashionIQ and "
            "iNaturalist staging paths; v6 rebases those source bindings to "
            "verified v4 materialized destinations"
        ),
        "selection_manifest_sha256": "b" * 64,
    }
    assert selection_value["selected_abo_cross_intent_candidates"] == [
        {
            "added_capability_bindings": [
                style.model_dump(mode="json"),
                encyclopedia.model_dump(mode="json"),
            ],
            "candidate_id": row.candidate_id,
            "component_id": component_by_candidate[row.candidate_id],
            "content_sha256": row.expected_sha256,
            "destination_path": row.destination_path,
            "prior_capability_bindings": [exact.model_dump(mode="json")],
            "source_record_id": row.draft.source_record_id,
        }
        for row in abo_exact[2:]
    ]
    repeated = build_portfolio_core_r2_candidate_catalog(
        v4_inventory_path=v4_path,
        recipe_inventory_path=recipe_path,
        document_inventory_path=document_path,
        v4_asset_catalog_dir=tmp_path / "v4-catalog",
        v4_asset_root=tmp_path / "v4-assets",
        output_path=output,
        selection_manifest_path=selection,
        superseded_v5_inventory_sha256="a" * 64,
        superseded_v5_selection_manifest_sha256="b" * 64,
    )
    assert repeated.output_sha256 == result.output_sha256
    assert repeated.selection_manifest_sha256 == result.selection_manifest_sha256
