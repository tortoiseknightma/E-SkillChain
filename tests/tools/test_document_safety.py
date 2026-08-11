from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil

from PIL import Image
import pytest

from skillchain.data.asset_catalog import (
    AssetCatalog,
    DatasetAssetDraft,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.schemas import DatasetAsset
from skillchain.tools import document_safety as safety_module
from skillchain.tools.document_ocr import (
    DocumentSafetyApproval,
    build_document_safety_approval,
)
from skillchain.tools.document_safety import (
    DOCUMENT_SAFETY_CATALOG_POLICY_VERSION,
    DocumentSafetyCatalogError,
    DocumentSafetyRecord,
    build_document_safety_catalog,
    build_document_safety_record,
    document_safety_catalog_digest,
    document_safety_review_ledger_digest,
    load_document_safety_catalog,
    publish_document_safety_catalog,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True)
class AssetFixture:
    asset_root: Path
    catalog: AssetCatalog
    parent: DatasetAsset
    redacted: DatasetAsset
    unrelated: DatasetAsset


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)
    pixels = bytes(randomizer.randrange(256) for _ in range(32 * 32 * 3))
    Image.frombytes("RGB", (32, 32), pixels).save(path, format="PNG")


def _inventory(
    asset_root: Path,
    local_path: str,
    *,
    seed: int,
    record_id: str,
    parents: list[str] | None = None,
) -> DatasetAsset:
    _write_image(asset_root / local_path, seed)
    return inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="document-safety-test",
            source_revision="fixture-v1",
            source_record_id=record_id,
            local_path=local_path,
            derivation_parent_asset_ids=parents or [],
            license_id="CC0-1.0",
            source_url=f"https://example.test/{record_id}",
            attribution="test fixture",
            cloud_upload_allowed=False,
            public_demo_allowed=False,
        ),
        asset_root,
    )


def _asset_fixture(base: Path, *, seed_offset: int = 0) -> AssetFixture:
    asset_root = base / "assets"
    parent = _inventory(
        asset_root,
        "documents/original.png",
        seed=11 + seed_offset,
        record_id="original",
    )
    redacted = _inventory(
        asset_root,
        "documents/redacted.png",
        seed=12 + seed_offset,
        record_id="redacted",
        parents=[parent.asset_id],
    )
    unrelated = _inventory(
        asset_root,
        "documents/unrelated.png",
        seed=13 + seed_offset,
        record_id="unrelated",
    )
    catalog_dir = base / "asset-catalog"
    publish_asset_catalog(
        [redacted, unrelated, parent],
        catalog_dir,
        asset_root,
        NearDuplicatePolicy(
            policy_version="document-safety-test-phash-v1",
            max_phash_hamming_distance=0,
        ),
        coverage_roots=["."],
    )
    return AssetFixture(
        asset_root=asset_root,
        catalog=load_asset_catalog(catalog_dir, asset_root, verify_files=True),
        parent=parent,
        redacted=redacted,
        unrelated=unrelated,
    )


def _approval(
    asset: DatasetAsset,
    *,
    redaction_parent_sha256: str | None = None,
    reviewer: str = "privacy-reviewer-1",
) -> DocumentSafetyApproval:
    decision = (
        "approved_redacted"
        if redaction_parent_sha256 is not None
        else "approved_no_pii"
    )
    return build_document_safety_approval(
        asset_id=asset.asset_id,
        image_sha256=asset.sha256,
        decision=decision,
        policy_version="pii-review-v1",
        reviewer_id=reviewer,
        redaction_parent_sha256=redaction_parent_sha256,
    )


def _records(fixture: AssetFixture) -> tuple[DocumentSafetyRecord, ...]:
    return (
        build_document_safety_record(
            "query-z",
            _approval(
                fixture.redacted,
                redaction_parent_sha256=fixture.parent.sha256,
            ),
        ),
        build_document_safety_record("query-a", _approval(fixture.unrelated)),
    )


def _replace_approvals_and_rebind(root: Path, rows: list[dict]) -> None:
    approvals_bytes = b"".join(canonical_json_bytes(row) for row in rows)
    (root / "approvals.jsonl").write_bytes(approvals_bytes)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["approvals"] = {
        "path": "approvals.jsonl",
        "count": len(rows),
        "bytes": len(approvals_bytes),
        "sha256": sha256_bytes(approvals_bytes),
    }
    manifest["review_ledger_sha256"] = sha256_bytes(approvals_bytes)
    manifest["approval_count"] = len(rows)
    manifest["query_count"] = len({row["query_id"] for row in rows})
    manifest["approved_no_pii_count"] = sum(
        row["approval"]["decision"] == "approved_no_pii" for row in rows
    )
    manifest["approved_redacted_count"] = sum(
        row["approval"]["decision"] == "approved_redacted" for row in rows
    )
    manifest["approval_policy_versions"] = sorted(
        {row["approval"]["policy_version"] for row in rows}
    )
    manifest["catalog_sha256"] = document_safety_catalog_digest(manifest)
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def _publish_records(
    records: tuple[DocumentSafetyRecord, ...] | list[DocumentSafetyRecord],
    output: Path,
    catalog: AssetCatalog,
):
    records = tuple(records)
    return publish_document_safety_catalog(
        records,
        output,
        catalog,
        expected_review_ledger_sha256=document_safety_review_ledger_digest(records),
    )


def _load_pinned(root: Path, fixture: AssetFixture):
    return load_document_safety_catalog(
        root,
        fixture.catalog,
        expected_review_ledger_sha256=document_safety_review_ledger_digest(
            _records(fixture)
        ),
    )


def _published(tmp_path: Path) -> tuple[AssetFixture, Path]:
    fixture = _asset_fixture(tmp_path)
    output = tmp_path / "document-safety"
    _publish_records(_records(fixture), output, fixture.catalog)
    return fixture, output


def test_round_trip_sorts_records_binds_catalog_and_resolves_exact_pair(
    tmp_path: Path,
) -> None:
    fixture = _asset_fixture(tmp_path)
    output = tmp_path / "document-safety"

    records = _records(fixture)
    catalog = build_document_safety_catalog(
        records,
        output,
        fixture.catalog,
        expected_review_ledger_sha256=document_safety_review_ledger_digest(records),
    )

    assert [record.query_id for record in catalog.approvals] == [
        "query-a",
        "query-z",
    ]
    assert catalog.manifest.policy_version == DOCUMENT_SAFETY_CATALOG_POLICY_VERSION
    assert catalog.manifest.asset_catalog_sha256 == fixture.catalog.catalog_sha256
    assert catalog.manifest.review_ledger_sha256 == sha256_bytes(
        catalog.approvals_bytes
    )
    assert tuple(
        artifact.path for artifact in catalog.manifest.asset_catalog_artifacts
    ) == ("assets.jsonl", "components.jsonl", "manifest.json")
    assert catalog.manifest.approval_count == 2
    assert catalog.manifest.query_count == 2
    assert catalog.manifest.approved_no_pii_count == 1
    assert catalog.manifest.approved_redacted_count == 1
    assert catalog.manifest.catalog_sha256 == document_safety_catalog_digest(
        catalog.manifest
    )
    assert len(catalog.formal_runtime_binding_sha256) == 64

    expected = next(
        record.approval for record in catalog.approvals if record.query_id == "query-z"
    )
    assert catalog.approval_for("query-z", fixture.redacted.asset_id) == expected
    assert catalog.approval_for("query-z", fixture.parent.asset_id) is None
    assert catalog.approval_for("missing", fixture.redacted.asset_id) is None
    assert catalog.approval_for(None, fixture.redacted.asset_id) is None  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        catalog._by_query_asset[("query-z", fixture.parent.asset_id)] = expected  # type: ignore[index]

    reloaded = _load_pinned(output, fixture)
    assert reloaded.manifest == catalog.manifest
    assert reloaded.approvals == catalog.approvals
    assert (
        reloaded.formal_runtime_binding_sha256 == catalog.formal_runtime_binding_sha256
    )
    assert reloaded.manifest_bytes == (output / "manifest.json").read_bytes()
    assert reloaded.approvals_bytes == (output / "approvals.jsonl").read_bytes()


def test_publisher_is_strictly_create_only_even_for_identical_content(
    tmp_path: Path,
) -> None:
    fixture, output = _published(tmp_path)
    before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }

    with pytest.raises(FileExistsError):
        _publish_records(_records(fixture), output, fixture.catalog)

    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == before


def test_publisher_does_not_remove_an_existing_empty_destination(
    tmp_path: Path,
) -> None:
    fixture = _asset_fixture(tmp_path)
    output = tmp_path / "document-safety"
    output.mkdir()

    with pytest.raises(FileExistsError):
        _publish_records(_records(fixture), output, fixture.catalog)

    assert output.is_dir()
    assert not tuple(output.iterdir())


def test_publish_race_never_replaces_or_removes_a_new_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _asset_fixture(tmp_path)
    output = tmp_path / "document-safety"
    original_create = safety_module._create_staging_directory

    def create_staging_then_race(destination: Path) -> Path:
        staging = original_create(destination)
        destination.mkdir()
        return staging

    monkeypatch.setattr(
        safety_module,
        "_create_staging_directory",
        create_staging_then_race,
    )
    with pytest.raises(FileExistsError):
        _publish_records(_records(fixture), output, fixture.catalog)

    assert output.is_dir()
    assert not tuple(output.iterdir())
    assert not list(tmp_path.glob(".document-safety.staging-*"))


def test_atomic_publish_failure_never_exposes_a_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _asset_fixture(tmp_path)
    output = tmp_path / "document-safety"

    def fail_atomic_publish(staging: Path, destination: Path) -> None:
        assert destination == output
        assert {path.name for path in staging.iterdir()} == {
            "approvals.jsonl",
            "manifest.json",
        }
        raise OSError("simulated atomic publish failure")

    monkeypatch.setattr(
        safety_module,
        "atomic_publish_new_directory",
        fail_atomic_publish,
    )
    with pytest.raises(OSError, match="simulated atomic publish failure"):
        _publish_records(_records(fixture), output, fixture.catalog)

    assert not output.exists()
    assert not list(tmp_path.glob(".document-safety.staging-*"))


def test_publisher_rejects_metadata_only_asset_catalog(tmp_path: Path) -> None:
    fixture = _asset_fixture(tmp_path)
    metadata_only = load_asset_catalog(
        fixture.catalog.root,
        fixture.asset_root,
        verify_files=False,
    )

    with pytest.raises(DocumentSafetyCatalogError, match="verify_files=True"):
        _publish_records(
            _records(fixture),
            tmp_path / "document-safety",
            metadata_only,
        )


def test_publisher_requires_exact_external_review_ledger_digest(
    tmp_path: Path,
) -> None:
    fixture = _asset_fixture(tmp_path)

    with pytest.raises(DocumentSafetyCatalogError, match="pinned external"):
        publish_document_safety_catalog(
            _records(fixture),
            tmp_path / "document-safety",
            fixture.catalog,
            expected_review_ledger_sha256="0" * 64,
        )


def test_publisher_rejects_approval_image_hash_mismatch(tmp_path: Path) -> None:
    fixture = _asset_fixture(tmp_path)
    mismatched = build_document_safety_approval(
        asset_id=fixture.redacted.asset_id,
        image_sha256=fixture.unrelated.sha256,
        decision="approved_no_pii",
        policy_version="pii-review-v1",
        reviewer_id="privacy-reviewer-1",
    )

    with pytest.raises(DocumentSafetyCatalogError, match="image hash"):
        _publish_records(
            [build_document_safety_record("query-a", mismatched)],
            tmp_path / "document-safety",
            fixture.catalog,
        )


def test_publisher_rejects_uncataloged_approval_asset(tmp_path: Path) -> None:
    fixture = _asset_fixture(tmp_path)
    uncataloged = build_document_safety_approval(
        asset_id="asset.v2." + "1" * 64,
        image_sha256=fixture.unrelated.sha256,
        decision="approved_no_pii",
        policy_version="pii-review-v1",
        reviewer_id="privacy-reviewer-1",
    )

    with pytest.raises(DocumentSafetyCatalogError, match="uncataloged"):
        _publish_records(
            [build_document_safety_record("query-a", uncataloged)],
            tmp_path / "document-safety",
            fixture.catalog,
        )


def test_publisher_does_not_trust_a_mutated_catalog_resolver_index(
    tmp_path: Path,
) -> None:
    fixture = _asset_fixture(tmp_path)
    fake_asset_id = "asset.v2." + "2" * 64
    fixture.catalog._by_asset_id[fake_asset_id] = fixture.catalog.resolve_asset_id(
        fixture.unrelated.asset_id
    )
    uncataloged = build_document_safety_approval(
        asset_id=fake_asset_id,
        image_sha256=fixture.unrelated.sha256,
        decision="approved_no_pii",
        policy_version="pii-review-v1",
        reviewer_id="privacy-reviewer-1",
    )

    with pytest.raises(DocumentSafetyCatalogError, match="uncataloged"):
        _publish_records(
            [build_document_safety_record("query-a", uncataloged)],
            tmp_path / "document-safety",
            fixture.catalog,
        )


def test_redacted_approval_requires_matching_direct_catalog_parent(
    tmp_path: Path,
) -> None:
    fixture = _asset_fixture(tmp_path)
    wrong_parent = _approval(
        fixture.redacted,
        redaction_parent_sha256=fixture.unrelated.sha256,
    )

    with pytest.raises(DocumentSafetyCatalogError, match="direct catalog lineage"):
        _publish_records(
            [build_document_safety_record("query-a", wrong_parent)],
            tmp_path / "document-safety",
            fixture.catalog,
        )


def test_redacted_approval_cannot_invent_lineage_for_parentless_asset(
    tmp_path: Path,
) -> None:
    fixture = _asset_fixture(tmp_path)
    invented_parent = _approval(
        fixture.unrelated,
        redaction_parent_sha256=fixture.parent.sha256,
    )

    with pytest.raises(DocumentSafetyCatalogError, match="direct catalog lineage"):
        _publish_records(
            [build_document_safety_record("query-a", invented_parent)],
            tmp_path / "document-safety",
            fixture.catalog,
        )


def test_duplicate_query_asset_authorizations_are_rejected(tmp_path: Path) -> None:
    fixture = _asset_fixture(tmp_path)
    approval = _approval(fixture.unrelated)
    duplicate = build_document_safety_record("query-a", approval)

    with pytest.raises(DocumentSafetyCatalogError, match="must be unique"):
        _publish_records(
            [duplicate, duplicate],
            tmp_path / "document-safety",
            fixture.catalog,
        )


def test_publisher_revalidates_model_copy_and_rejects_bad_approval_self_hash(
    tmp_path: Path,
) -> None:
    fixture = _asset_fixture(tmp_path)
    approval = _approval(fixture.unrelated).model_copy(
        update={"reviewer_id": "unhashed-reviewer-change"}
    )
    record = DocumentSafetyRecord.model_construct(
        schema_version=1,
        query_id="query-a",
        approval=approval,
    )

    with pytest.raises(DocumentSafetyCatalogError, match="record 1 is invalid"):
        _publish_records(
            [record],
            tmp_path / "document-safety",
            fixture.catalog,
        )


def test_loader_rejects_reordered_rows_even_with_recomputed_outer_hashes(
    tmp_path: Path,
) -> None:
    fixture, output = _published(tmp_path)
    rows = [
        json.loads(line)
        for line in (output / "approvals.jsonl").read_text().splitlines()
    ]
    _replace_approvals_and_rebind(output, list(reversed(rows)))
    rebound_sha256 = sha256_bytes((output / "approvals.jsonl").read_bytes())

    with pytest.raises(DocumentSafetyCatalogError, match="canonical query/asset order"):
        load_document_safety_catalog(
            output,
            fixture.catalog,
            expected_review_ledger_sha256=rebound_sha256,
        )


def test_loader_rejects_duplicate_pair_even_with_recomputed_outer_hashes(
    tmp_path: Path,
) -> None:
    fixture, output = _published(tmp_path)
    rows = [
        json.loads(line)
        for line in (output / "approvals.jsonl").read_text().splitlines()
    ]
    _replace_approvals_and_rebind(output, [rows[0], rows[0], rows[1]])
    rebound_sha256 = sha256_bytes((output / "approvals.jsonl").read_bytes())

    with pytest.raises(DocumentSafetyCatalogError, match="must be unique"):
        load_document_safety_catalog(
            output,
            fixture.catalog,
            expected_review_ledger_sha256=rebound_sha256,
        )


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path)
    content = (output / "approvals.jsonl").read_bytes()
    tampered = content.replace(
        b'"query_id":"query-a"',
        b'"query_id":"query-a","query_id":"query-a"',
        1,
    )
    assert tampered != content
    (output / "approvals.jsonl").write_bytes(tampered)

    with pytest.raises(DocumentSafetyCatalogError, match="duplicate key"):
        _load_pinned(output, fixture)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda content: b" " + content,
        lambda content: content.rstrip(b"\n"),
        lambda content: content + b"\n",
    ],
    ids=["leading-space", "partial-last-line", "blank-line"],
)
def test_loader_rejects_noncanonical_approval_jsonl(
    tmp_path: Path,
    mutator,
) -> None:
    fixture, output = _published(tmp_path)
    approvals_path = output / "approvals.jsonl"
    approvals_path.write_bytes(mutator(approvals_path.read_bytes()))

    with pytest.raises(DocumentSafetyCatalogError):
        _load_pinned(output, fixture)


def test_loader_rejects_tampered_approval_self_hash_after_outer_rebind(
    tmp_path: Path,
) -> None:
    fixture, output = _published(tmp_path)
    rows = [
        json.loads(line)
        for line in (output / "approvals.jsonl").read_text().splitlines()
    ]
    rows[0]["approval"]["reviewer_id"] = "attacker"
    _replace_approvals_and_rebind(output, rows)
    rebound_sha256 = sha256_bytes((output / "approvals.jsonl").read_bytes())

    with pytest.raises(DocumentSafetyCatalogError, match="line 1 is invalid"):
        load_document_safety_catalog(
            output,
            fixture.catalog,
            expected_review_ledger_sha256=rebound_sha256,
        )


def test_pinned_review_ledger_rejects_fully_rehashed_query_forgery(
    tmp_path: Path,
) -> None:
    fixture, output = _published(tmp_path)
    rows = [
        json.loads(line)
        for line in (output / "approvals.jsonl").read_text().splitlines()
    ]
    rows[0]["query_id"] = "forged-query"
    _replace_approvals_and_rebind(output, rows)

    with pytest.raises(DocumentSafetyCatalogError, match="pinned review ledger"):
        _load_pinned(output, fixture)


def test_loader_rejects_manifest_self_hash_tampering(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path)
    manifest = json.loads((output / "manifest.json").read_bytes())
    manifest["catalog_sha256"] = "0" * 64
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(DocumentSafetyCatalogError, match="manifest.json is invalid"):
        _load_pinned(output, fixture)


def test_loader_rejects_static_approval_byte_tampering(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path)
    approvals_path = output / "approvals.jsonl"
    approvals_path.write_bytes(
        approvals_path.read_bytes().replace(
            b"privacy-reviewer-1", b"privacy-reviewer-2"
        )
    )

    with pytest.raises(DocumentSafetyCatalogError):
        _load_pinned(output, fixture)


def test_loader_rejects_extra_artifacts(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path)
    (output / "untracked.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DocumentSafetyCatalogError, match="artifact set"):
        _load_pinned(output, fixture)


def test_loader_rejects_symlinked_safety_artifact(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path)
    approvals_path = output / "approvals.jsonl"
    backing = tmp_path / "approvals-backing.jsonl"
    shutil.move(approvals_path, backing)
    try:
        approvals_path.symlink_to(backing)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(DocumentSafetyCatalogError, match="non-symlink"):
        _load_pinned(output, fixture)


def test_loader_rejects_symlinked_asset_catalog_artifact(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path)
    manifest_path = fixture.catalog.root / "manifest.json"
    backing = tmp_path / "asset-manifest-backing.json"
    shutil.move(manifest_path, backing)
    try:
        manifest_path.symlink_to(backing)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(DocumentSafetyCatalogError, match="stable regular files"):
        _load_pinned(output, fixture)


def test_loader_rejects_symlinked_approved_asset_even_when_bytes_match(
    tmp_path: Path,
) -> None:
    fixture, output = _published(tmp_path)
    asset_path = fixture.asset_root / fixture.unrelated.local_path
    backing = fixture.asset_root / "same-image-bytes.bin"
    shutil.move(asset_path, backing)
    try:
        asset_path.symlink_to(backing)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(DocumentSafetyCatalogError, match="symlink"):
        _load_pinned(output, fixture)


def test_loader_rejects_modified_approved_image_bytes(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path)
    path = fixture.asset_root / fixture.unrelated.local_path
    _write_image(path, seed=999)

    with pytest.raises(DocumentSafetyCatalogError, match="full verification"):
        _load_pinned(output, fixture)


def test_loader_rejects_modified_asset_catalog_bytes(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path)
    manifest_path = fixture.catalog.root / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(DocumentSafetyCatalogError, match="full verification"):
        _load_pinned(output, fixture)


def test_loader_rejects_a_different_verified_asset_catalog(tmp_path: Path) -> None:
    fixture, output = _published(tmp_path / "primary")
    other = _asset_fixture(tmp_path / "other", seed_offset=100)

    with pytest.raises(DocumentSafetyCatalogError, match="does not bind"):
        load_document_safety_catalog(
            output,
            other.catalog,
            expected_review_ledger_sha256=document_safety_review_ledger_digest(
                _records(fixture)
            ),
        )


def test_build_detects_asset_catalog_mutation_before_publish_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _asset_fixture(tmp_path)
    output = tmp_path / "document-safety"
    catalog_manifest = fixture.catalog.root / "manifest.json"
    original_manifest = catalog_manifest.read_bytes()
    original_writer = safety_module._write_staging_bundle

    def write_then_mutate(staging, approvals_bytes, manifest_bytes):
        original_writer(staging, approvals_bytes, manifest_bytes)
        catalog_manifest.write_bytes(original_manifest + b" ")

    monkeypatch.setattr(safety_module, "_write_staging_bundle", write_then_mutate)
    try:
        with pytest.raises(DocumentSafetyCatalogError, match="full verification"):
            _publish_records(_records(fixture), output, fixture.catalog)
    finally:
        catalog_manifest.write_bytes(original_manifest)

    assert not output.exists()
    assert not list(tmp_path.glob(".document-safety.staging-*"))


def test_existing_broken_symlink_destination_is_not_replaced(tmp_path: Path) -> None:
    fixture = _asset_fixture(tmp_path)
    output = tmp_path / "document-safety"
    try:
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(FileExistsError):
        _publish_records(_records(fixture), output, fixture.catalog)
    assert output.is_symlink()
