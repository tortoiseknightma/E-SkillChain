from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pydantic import ValidationError
import pytest

from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    GalleryAssetReference,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.data.gallery_eligibility import (
    GalleryEligibilityError,
    GalleryEligibilityManifest,
    build_gallery_eligibility_manifest,
    load_gallery_artifact,
    load_gallery_eligibility_manifest,
    load_query_artifact,
    verify_gallery_eligibility,
)
from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from skillchain.taxonomy import (
    TASK_SPEC_VERSION,
    TAXONOMY_VERSION,
    capability_for_intent,
    requires_card_for_intent,
)


def _write_image(path: Path, *, diagonal: bool) -> None:
    image = Image.new("RGB", (64, 64), "white")
    pixels = image.load()
    assert pixels is not None
    for y in range(64):
        for x in range(64):
            if (diagonal and x < y) or (not diagonal and (x // 8 + y // 8) % 2):
                pixels[x, y] = (0, 0, 0)
    image.save(path, format="PNG")


def _query(catalog, asset_id: str, *, query_id: str = "q-safe") -> Query:
    resolution = catalog.resolve_asset_id(asset_id)
    intent = "multi_product"
    capability = capability_for_intent(intent)
    text = f"mechanical query {query_id}"
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=TASK_SPEC_VERSION,
        query_id=query_id,
        asset_id=asset_id,
        image_path=resolution.local_path,
        leakage_group_id=resolution.leakage_group_id,
        template_family="gallery-verifier-test-v1",
        generator_batch_id="gallery-verifier-test-batch",
        text=text,
        turns=[{"role": "user", "content": text}],
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        requires_card=requires_card_for_intent(intent),
        split="dev_mini",
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="gallery-verifier-test",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _evidence(tmp_path: Path):
    asset_root = tmp_path / "assets"
    image_root = asset_root / "images"
    image_root.mkdir(parents=True)
    _write_image(image_root / "query.png", diagonal=True)
    _write_image(image_root / "gallery.png", diagonal=False)
    drafts = (
        DatasetAssetDraft(
            source_dataset="eligibility-fixture",
            source_revision="fixture-v1",
            source_record_id="query",
            local_path="images/query.png",
            product_id="product-query",
            license_id="test-only",
        ),
        DatasetAssetDraft(
            source_dataset="eligibility-fixture",
            source_revision="fixture-v1",
            source_record_id="gallery",
            local_path="images/gallery.png",
            product_id="product-gallery",
            license_id="test-only",
        ),
    )
    assets = tuple(inventory_dataset_asset(draft, asset_root) for draft in drafts)
    catalog_dir = tmp_path / "catalog"
    publish_asset_catalog(
        assets,
        catalog_dir,
        asset_root,
        NearDuplicatePolicy(max_phash_hamming_distance=0),
        coverage_roots=["images"],
    )
    catalog = load_asset_catalog(catalog_dir, asset_root, verify_files=True)
    by_record = {asset.source_record_id: asset for asset in catalog.assets}
    query_path = tmp_path / "queries.jsonl"
    gallery_path = tmp_path / "gallery.jsonl"
    query_path.write_bytes(
        canonical_jsonl_bytes([_query(catalog, by_record["query"].asset_id)])
    )
    gallery_path.write_bytes(
        canonical_jsonl_bytes(
            [
                GalleryAssetReference(
                    asset_id=by_record["gallery"].asset_id,
                    image_path=by_record["gallery"].local_path,
                )
            ]
        )
    )
    manifest = build_gallery_eligibility_manifest(
        query_path,
        gallery_path,
        catalog,
    )
    manifest_path = tmp_path / "eligibility.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return {
        "asset_root": asset_root,
        "catalog_dir": catalog_dir,
        "catalog": catalog,
        "query_path": query_path,
        "gallery_path": gallery_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "assets": by_record,
    }


def _rewrite_manifest(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    unsigned = dict(value)
    unsigned.pop("eligibility_sha256", None)
    value["eligibility_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    path.write_bytes(canonical_json_bytes(value))


def test_build_and_verify_recomputes_all_transitive_evidence(tmp_path):
    evidence = _evidence(tmp_path)

    loaded_manifest = load_gallery_eligibility_manifest(evidence["manifest_path"])
    verified = verify_gallery_eligibility(
        evidence["manifest_path"],
        evidence["query_path"],
        evidence["gallery_path"],
        evidence["catalog"],
    )

    assert isinstance(loaded_manifest, GalleryEligibilityManifest)
    assert verified.manifest == loaded_manifest
    assert verified.manifest.eligible is True
    assert verified.manifest.status == "ok"
    assert verified.manifest.report.violation_count == 0
    assert verified.manifest.catalog_sha256 == evidence["catalog"].catalog_sha256
    assert verified.query_bytes == evidence["query_path"].read_bytes()
    assert verified.gallery_bytes == evidence["gallery_path"].read_bytes()
    assert [query.query_id for query in verified.queries] == ["q-safe"]
    assert [asset.asset_id for asset in verified.gallery_assets] == [
        evidence["assets"]["gallery"].asset_id
    ]


def test_success_manifest_type_rejects_blocked_or_violating_payload(tmp_path):
    evidence = _evidence(tmp_path)
    payload = evidence["manifest"].model_dump(mode="json")
    payload["eligible"] = False
    payload["status"] = "blocked"
    with pytest.raises(ValidationError):
        GalleryEligibilityManifest.model_validate(payload)

    payload = evidence["manifest"].model_dump(mode="json")
    payload["report"]["missing_exact_match_positive"] = ["q-spoofed"]
    with pytest.raises(ValidationError, match="must not contain violations"):
        GalleryEligibilityManifest.model_validate(payload)


def test_artifact_loaders_reject_duplicate_keys_and_noncanonical_jsonl(tmp_path):
    evidence = _evidence(tmp_path)
    evidence["query_path"].write_text(
        '{"schema_version":2,"schema_version":2}\n', encoding="utf-8"
    )
    with pytest.raises(GalleryEligibilityError, match="duplicate key"):
        load_query_artifact(evidence["query_path"], evidence["catalog"])

    reference = GalleryAssetReference(
        asset_id=evidence["assets"]["gallery"].asset_id,
        image_path=evidence["assets"]["gallery"].local_path,
    )
    evidence["gallery_path"].write_text(
        json.dumps(reference.model_dump(mode="json"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GalleryEligibilityError, match="canonical JSONL"):
        load_gallery_artifact(evidence["gallery_path"], evidence["catalog"])


def test_gallery_loader_requires_unique_image_paths(tmp_path):
    evidence = _evidence(tmp_path)
    shared_path = evidence["assets"]["gallery"].local_path
    evidence["gallery_path"].write_bytes(
        canonical_jsonl_bytes(
            [
                GalleryAssetReference(
                    asset_id=evidence["assets"]["gallery"].asset_id,
                    image_path=shared_path,
                ),
                GalleryAssetReference(
                    asset_id=evidence["assets"]["query"].asset_id,
                    image_path=shared_path,
                ),
            ]
        )
    )

    with pytest.raises(GalleryEligibilityError, match="image_path must be unique"):
        load_gallery_artifact(evidence["gallery_path"], evidence["catalog"])


def test_manifest_loader_rejects_non_regular_file(tmp_path):
    directory = tmp_path / "eligibility.json"
    directory.mkdir()

    with pytest.raises(GalleryEligibilityError, match="regular file"):
        load_gallery_eligibility_manifest(directory)


def test_manifest_loader_rejects_symbolic_link(tmp_path):
    evidence = _evidence(tmp_path)
    link = tmp_path / "eligibility-link.json"
    try:
        link.symlink_to(evidence["manifest_path"])
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(GalleryEligibilityError, match="symbolic link"):
        load_gallery_eligibility_manifest(link)


def test_verifier_rejects_duplicate_keys_and_noncanonical_manifest(tmp_path):
    evidence = _evidence(tmp_path)
    original = evidence["manifest_path"].read_text(encoding="utf-8")
    evidence["manifest_path"].write_text(
        '{"schema_version":1,' + original[1:], encoding="utf-8"
    )
    with pytest.raises(GalleryEligibilityError, match="duplicate key"):
        verify_gallery_eligibility(
            evidence["manifest_path"],
            evidence["query_path"],
            evidence["gallery_path"],
            evidence["catalog"],
        )

    evidence = _evidence(tmp_path / "noncanonical")
    value = json.loads(evidence["manifest_path"].read_text(encoding="utf-8"))
    evidence["manifest_path"].write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GalleryEligibilityError, match="canonical JSON"):
        verify_gallery_eligibility(
            evidence["manifest_path"],
            evidence["query_path"],
            evidence["gallery_path"],
            evidence["catalog"],
        )


def test_verifier_rejects_self_hash_or_catalog_policy_tampering(tmp_path):
    evidence = _evidence(tmp_path)
    value = json.loads(evidence["manifest_path"].read_text(encoding="utf-8"))
    value["eligibility_sha256"] = "0" * 64
    evidence["manifest_path"].write_bytes(canonical_json_bytes(value))
    with pytest.raises(GalleryEligibilityError, match="self hash"):
        verify_gallery_eligibility(
            evidence["manifest_path"],
            evidence["query_path"],
            evidence["gallery_path"],
            evidence["catalog"],
        )

    evidence = _evidence(tmp_path / "policy")
    _rewrite_manifest(
        evidence["manifest_path"],
        lambda manifest: manifest.__setitem__("catalog_policy_version", "spoof-v1"),
    )
    with pytest.raises(GalleryEligibilityError, match="catalog_policy_version"):
        verify_gallery_eligibility(
            evidence["manifest_path"],
            evidence["query_path"],
            evidence["gallery_path"],
            evidence["catalog"],
        )


def test_verifier_rejects_query_or_gallery_artifact_substitution(tmp_path):
    evidence = _evidence(tmp_path)
    replacement = _query(
        evidence["catalog"],
        evidence["assets"]["query"].asset_id,
        query_id="q-replaced",
    )
    evidence["query_path"].write_bytes(canonical_jsonl_bytes([replacement]))
    with pytest.raises(GalleryEligibilityError, match="query_artifact_sha256"):
        verify_gallery_eligibility(
            evidence["manifest_path"],
            evidence["query_path"],
            evidence["gallery_path"],
            evidence["catalog"],
        )

    evidence = _evidence(tmp_path / "gallery-substitution")
    replacement_gallery = GalleryAssetReference(
        asset_id=evidence["assets"]["query"].asset_id,
        image_path=evidence["assets"]["query"].local_path,
    )
    evidence["gallery_path"].write_bytes(canonical_jsonl_bytes([replacement_gallery]))
    with pytest.raises(GalleryEligibilityError, match="gallery_artifact_sha256"):
        verify_gallery_eligibility(
            evidence["manifest_path"],
            evidence["query_path"],
            evidence["gallery_path"],
            evidence["catalog"],
        )


def test_verifier_rejects_rehashed_spoofed_report(tmp_path):
    evidence = _evidence(tmp_path)

    def spoof_report(manifest: dict) -> None:
        manifest["report"]["query_count"] += 1

    _rewrite_manifest(evidence["manifest_path"], spoof_report)
    with pytest.raises(GalleryEligibilityError, match="recomputed audit"):
        verify_gallery_eligibility(
            evidence["manifest_path"],
            evidence["query_path"],
            evidence["gallery_path"],
            evidence["catalog"],
        )


def test_verifier_rejects_metadata_only_catalog(tmp_path):
    evidence = _evidence(tmp_path)
    metadata_only = load_asset_catalog(
        evidence["catalog_dir"],
        evidence["asset_root"],
        verify_files=False,
    )

    with pytest.raises(GalleryEligibilityError, match="verify_files"):
        verify_gallery_eligibility(
            evidence["manifest_path"],
            evidence["query_path"],
            evidence["gallery_path"],
            metadata_only,
        )
