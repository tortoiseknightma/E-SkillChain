from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random

from PIL import Image

from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    publish_asset_catalog,
)
from skillchain.tools.document_ocr import build_document_safety_approval
from skillchain.tools.document_safety import build_document_safety_record
from skillchain.tools.serialization import canonical_jsonl_bytes, sha256_bytes


def _script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_document_safety.py"
    spec = importlib.util.spec_from_file_location("build_document_safety_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, Path, bytes]:
    asset_root = tmp_path / "assets"
    image_path = asset_root / "document.png"
    image_path.parent.mkdir(parents=True)
    randomizer = random.Random(17)
    pixels = bytes(randomizer.randrange(256) for _ in range(32 * 32 * 3))
    Image.frombytes("RGB", (32, 32), pixels).save(image_path, format="PNG")
    asset = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="document-safety-cli-test",
            source_revision="fixture-v1",
            source_record_id="document-1",
            local_path="document.png",
            license_id="CC0-1.0",
            source_url="https://example.test/document-1",
            attribution="test fixture",
            cloud_upload_allowed=False,
            public_demo_allowed=False,
        ),
        asset_root,
    )
    catalog_dir = tmp_path / "asset-catalog"
    publish_asset_catalog(
        [asset],
        catalog_dir,
        asset_root,
        NearDuplicatePolicy(
            policy_version="document-safety-cli-test-phash-v1",
            max_phash_hamming_distance=0,
        ),
        coverage_roots=["."],
    )
    approval = build_document_safety_approval(
        asset_id=asset.asset_id,
        image_sha256=asset.sha256,
        decision="approved_no_pii",
        policy_version="pii-review-v1",
        reviewer_id="privacy-reviewer-1",
    )
    record = build_document_safety_record("query-1", approval)
    ledger = canonical_jsonl_bytes((record.model_dump(mode="json"),))
    return asset_root, catalog_dir, ledger


def _build_arguments(
    ledger_path: Path,
    ledger_sha256: str,
    asset_root: Path,
    catalog_dir: Path,
    output: Path,
) -> list[str]:
    return [
        "build",
        "--approvals",
        str(ledger_path),
        "--review-ledger-sha256",
        ledger_sha256,
        "--asset-catalog",
        str(catalog_dir),
        "--asset-root",
        str(asset_root),
        "--output",
        str(output),
    ]


def test_build_and_status_verify_a_pinned_canonical_ledger(
    tmp_path: Path, capsys
) -> None:
    script = _script()
    asset_root, catalog_dir, ledger = _fixture(tmp_path)
    ledger_path = tmp_path / "review-ledger.jsonl"
    ledger_path.write_bytes(ledger)
    ledger_sha256 = sha256_bytes(ledger)
    output = tmp_path / "document-safety"

    assert (
        script.main(
            _build_arguments(
                ledger_path,
                ledger_sha256,
                asset_root,
                catalog_dir,
                output,
            )
        )
        == 0
    )
    published = json.loads(capsys.readouterr().out)
    assert published["review_ledger_sha256"] == ledger_sha256
    assert (output / "approvals.jsonl").read_bytes() == ledger

    common = [
        "--review-ledger-sha256",
        ledger_sha256,
        "--asset-catalog",
        str(catalog_dir),
        "--asset-root",
        str(asset_root),
        "--output",
        str(output),
    ]
    assert script.main(["status", *common]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["catalog_sha256"] == published["catalog_sha256"]
    assert script.main(["verify", *common]) == 0
    assert (
        json.loads(capsys.readouterr().out)["catalog_sha256"]
        == published["catalog_sha256"]
    )


def test_build_rejects_external_hash_mismatch_with_fixed_error(
    tmp_path: Path, capsys
) -> None:
    script = _script()
    asset_root, catalog_dir, ledger = _fixture(tmp_path)
    ledger_path = tmp_path / "review-ledger.jsonl"
    ledger_path.write_bytes(ledger)

    assert (
        script.main(
            _build_arguments(
                ledger_path,
                "0" * 64,
                asset_root,
                catalog_dir,
                tmp_path / "document-safety",
            )
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "build-document-safety: failed\n"


def test_build_rejects_noncanonical_and_schema_invalid_ledgers(
    tmp_path: Path, capsys
) -> None:
    script = _script()
    asset_root, catalog_dir, ledger = _fixture(tmp_path)
    output = tmp_path / "document-safety"

    noncanonical_path = tmp_path / "noncanonical.jsonl"
    noncanonical = b'{"query_id": "query-1"}\n'
    noncanonical_path.write_bytes(noncanonical)
    assert (
        script.main(
            _build_arguments(
                noncanonical_path,
                sha256_bytes(noncanonical),
                asset_root,
                catalog_dir,
                output,
            )
        )
        == 2
    )
    assert capsys.readouterr().err == "build-document-safety: failed\n"

    invalid_path = tmp_path / "invalid.jsonl"
    row = json.loads(ledger)
    row["unexpected"] = True
    invalid = canonical_jsonl_bytes((row,))
    invalid_path.write_bytes(invalid)
    assert (
        script.main(
            _build_arguments(
                invalid_path,
                sha256_bytes(invalid),
                asset_root,
                catalog_dir,
                output,
            )
        )
        == 2
    )
    assert capsys.readouterr().err == "build-document-safety: failed\n"
    assert not output.exists()


def test_build_is_create_only_and_status_requires_the_expected_hash(
    tmp_path: Path, capsys
) -> None:
    script = _script()
    asset_root, catalog_dir, ledger = _fixture(tmp_path)
    ledger_path = tmp_path / "review-ledger.jsonl"
    ledger_path.write_bytes(ledger)
    ledger_sha256 = sha256_bytes(ledger)
    output = tmp_path / "document-safety"
    arguments = _build_arguments(
        ledger_path,
        ledger_sha256,
        asset_root,
        catalog_dir,
        output,
    )

    assert script.main(arguments) == 0
    capsys.readouterr()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert script.main(arguments) == 2
    assert capsys.readouterr().err == "build-document-safety: failed\n"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before

    assert (
        script.main(
            [
                "status",
                "--review-ledger-sha256",
                "f" * 64,
                "--asset-catalog",
                str(catalog_dir),
                "--asset-root",
                str(asset_root),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert capsys.readouterr().err == "build-document-safety: failed\n"
