from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from skillchain.data.asset_catalog import GalleryAssetReference, load_asset_catalog
from skillchain.data.asset_catalog_cli import main
from skillchain.data.gallery_eligibility import verify_gallery_eligibility
from skillchain.schemas import LabelDecision, Query
from skillchain.synthesis.store import canonical_jsonl_bytes
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


def _draft(local_path: str, record_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_dataset": "cli-fixture",
        "source_revision": "fixture-revision-v1",
        "source_record_id": record_id,
        "transform_policy_version": "identity-v1",
        "local_path": local_path,
        "product_id": f"product-{record_id}",
        "derivation_parent_asset_ids": [],
        "derivation_parent_asset_id": None,
        "license_id": "test-only",
        "source_url": None,
        "attribution": None,
        "cloud_upload_allowed": False,
        "public_demo_allowed": False,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _query(catalog, asset_id: str, query_id: str, intent: str) -> Query:
    resolution = catalog.resolve_asset_id(asset_id)
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
        template_family="cli-gallery-test-v1",
        generator_batch_id="cli-gallery-test-batch",
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
                annotator_id="cli-gallery-test",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _fixture(tmp_path: Path):
    asset_root = tmp_path / "assets"
    coverage = asset_root / "query_images"
    coverage.mkdir(parents=True)
    _write_image(coverage / "one.png", diagonal=True)
    _write_image(coverage / "two.png", diagonal=False)
    drafts = tmp_path / "drafts.jsonl"
    _write_jsonl(
        drafts,
        [
            _draft("query_images/one.png", "one"),
            _draft("query_images/two.png", "two"),
        ],
    )
    return asset_root, drafts, tmp_path / "catalog"


def _build_args(asset_root: Path, drafts: Path, output: Path) -> list[str]:
    return [
        "build",
        "--drafts",
        str(drafts),
        "--asset-root",
        str(asset_root),
        "--output",
        str(output),
        "--coverage-root",
        "query_images",
        "--max-phash-distance",
        "0",
    ]


def test_build_and_verify_emit_machine_readable_summaries(tmp_path, capsys):
    asset_root, drafts, output = _fixture(tmp_path)

    assert main(_build_args(asset_root, drafts, output)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    built = json.loads(captured.out)
    assert built["status"] == "ok"
    assert built["command"] == "build"
    assert built["asset_count"] == 2
    assert built["coverage_roots"] == ["query_images"]
    assert built["verified_files"] is True
    assert len(built["catalog_sha256"]) == 64
    assert built["near_duplicate_policy"]["max_phash_hamming_distance"] == 0
    assert built["near_duplicate_policy"]["scipy_version"]

    assert (
        main(
            [
                "verify",
                "--catalog",
                str(output),
                "--asset-root",
                str(asset_root),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    verified = json.loads(captured.out)
    assert verified["status"] == "ok"
    assert verified["command"] == "verify"
    assert verified["catalog_sha256"] == built["catalog_sha256"]
    assert verified["asset_count"] == built["asset_count"]
    assert verified["component_count"] == built["component_count"]


def test_build_rejects_blank_jsonl_line_before_publication(tmp_path, capsys):
    asset_root, drafts, output = _fixture(tmp_path)
    original = drafts.read_text(encoding="utf-8").splitlines()
    drafts.write_text(f"{original[0]}\n\n{original[1]}\n", encoding="utf-8")

    assert main(_build_args(asset_root, drafts, output)) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert "blank" in error["message"]
    assert not output.exists()


def test_build_rejects_duplicate_draft_row_before_publication(tmp_path, capsys):
    asset_root, drafts, output = _fixture(tmp_path)
    first = drafts.read_text(encoding="utf-8").splitlines()[0]
    drafts.write_text(f"{first}\n{first}\n", encoding="utf-8")

    assert main(_build_args(asset_root, drafts, output)) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert "duplicate" in error["message"]
    assert not output.exists()


def test_build_rejects_invalid_or_non_strict_json(tmp_path, capsys):
    asset_root, drafts, output = _fixture(tmp_path)
    drafts.write_text(
        '{"source_dataset":"fixture","source_dataset":"duplicate"}\n',
        encoding="utf-8",
    )

    assert main(_build_args(asset_root, drafts, output)) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert "duplicate key" in error["message"]
    assert not output.exists()


def test_formal_build_requires_coverage_root(tmp_path, capsys):
    asset_root, drafts, output = _fixture(tmp_path)
    args = _build_args(asset_root, drafts, output)
    coverage_index = args.index("--coverage-root")
    del args[coverage_index : coverage_index + 2]

    assert main(args) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert "--coverage-root" in error["message"]
    assert not output.exists()


def test_verify_rejects_asset_byte_tampering(tmp_path, capsys):
    asset_root, drafts, output = _fixture(tmp_path)
    assert main(_build_args(asset_root, drafts, output)) == 0
    capsys.readouterr()
    (asset_root / "query_images" / "one.png").write_bytes(b"tampered")

    assert (
        main(
            [
                "verify",
                "--catalog",
                str(output),
                "--asset-root",
                str(asset_root),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["status"] == "error"
    assert error["command"] == "verify"


def test_audit_gallery_emits_verified_eligibility_report(tmp_path, capsys):
    asset_root, drafts, output = _fixture(tmp_path)
    assert main(_build_args(asset_root, drafts, output)) == 0
    capsys.readouterr()
    catalog = load_asset_catalog(output, asset_root)
    ids = {asset.source_record_id: asset.asset_id for asset in catalog.assets}
    queries = tmp_path / "queries.jsonl"
    gallery = tmp_path / "gallery.jsonl"
    eligibility = tmp_path / "eligibility.json"
    queries.write_bytes(
        canonical_jsonl_bytes([_query(catalog, ids["one"], "q-safe", "multi_product")])
    )
    gallery.write_bytes(
        canonical_jsonl_bytes(
            [
                GalleryAssetReference(
                    asset_id=ids["two"],
                    image_path=catalog.resolve_asset_id(ids["two"]).local_path,
                )
            ]
        )
    )

    assert (
        main(
            [
                "audit-gallery",
                "--catalog",
                str(output),
                "--asset-root",
                str(asset_root),
                "--queries",
                str(queries),
                "--gallery-assets",
                str(gallery),
                "--output",
                str(eligibility),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["eligible"] is True
    assert result["report"]["query_count"] == 1
    assert result["query_artifact_sha256"]
    assert result["gallery_artifact_sha256"]
    assert result["eligibility_policy_version"] == "query-gallery-eligibility-v1"
    assert result["catalog_policy_version"] == catalog.manifest.catalog_policy_version
    assert (
        json.loads(eligibility.read_text(encoding="utf-8"))["eligibility_sha256"]
        == result["eligibility_sha256"]
    )
    verified = verify_gallery_eligibility(
        eligibility,
        queries,
        gallery,
        catalog,
    )
    assert verified.manifest.eligibility_sha256 == result["eligibility_sha256"]


def test_audit_gallery_blocks_query_asset_exposed_in_gallery(tmp_path, capsys):
    asset_root, drafts, output = _fixture(tmp_path)
    assert main(_build_args(asset_root, drafts, output)) == 0
    capsys.readouterr()
    catalog = load_asset_catalog(output, asset_root)
    query_asset = next(
        asset for asset in catalog.assets if asset.source_record_id == "one"
    )
    queries = tmp_path / "queries.jsonl"
    gallery = tmp_path / "gallery.jsonl"
    queries.write_bytes(
        canonical_jsonl_bytes(
            [_query(catalog, query_asset.asset_id, "q-self", "exact_match")]
        )
    )
    gallery.write_bytes(
        canonical_jsonl_bytes(
            [
                GalleryAssetReference(
                    asset_id=query_asset.asset_id,
                    image_path=query_asset.local_path,
                )
            ]
        )
    )

    assert (
        main(
            [
                "audit-gallery",
                "--catalog",
                str(output),
                "--asset-root",
                str(asset_root),
                "--queries",
                str(queries),
                "--gallery-assets",
                str(gallery),
                "--output",
                str(tmp_path / "blocked-eligibility.json"),
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    result = json.loads(captured.err)
    assert result["status"] == "blocked"
    assert result["eligible"] is False
    assert result["report"]["exact_or_near_duplicate_violations"] == {
        "q-self": [query_asset.asset_id]
    }
