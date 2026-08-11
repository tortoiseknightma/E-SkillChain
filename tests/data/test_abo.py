from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from skillchain.data import abo
from skillchain.data.abo import (
    ABOAcquisitionReceipt,
    ABOExactCoverageError,
    ABOGlobalCatalogScope,
    ABOGlobalFinalizationError,
    ABOImageManifestRecord,
    ABOImageReviewDecision,
    ABOItemAcquisitionReceipt,
    ABOListingRecord,
    ABOOfficialImageReceipt,
    ABOPermissionMatrix,
    ABOProvenanceError,
    ABORawArtifactReceipt,
    ABORawRecordLocator,
    ABOSourceLock,
    ABO_FORMAL_PERMISSIONS,
    ABO_FORMAL_PURPOSES,
    LICENSE_ID,
    VerifiedABOSourceApproval,
    acquire_targeted_abo_original_images,
    audit_abo_exact_coverage,
    build_abo_exact_mini,
    create_abo_candidate_review_packet,
    finalize_abo_exact_candidates,
    load_verified_abo_candidate_bundle,
    load_verified_abo_global_finalization,
    load_verified_abo_source_approval,
    load_verified_abo_source_bundle,
)
from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.data.source_lock import (
    AcquisitionIdentity,
    RequiredSourceLock,
    build_artifact_scope,
)
from skillchain.data.source_review import (
    SourcePermissions,
    SourceReviewPolicy,
    SourceReviewRecord,
    SourceReviewRequirement,
)
from skillchain.synthesis.store import canonical_json_bytes, canonical_jsonl_bytes


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _image_bytes(seed: int, *, same_as: bytes | None = None) -> bytes:
    if same_as is not None:
        return same_as
    rng = random.Random(seed)
    pixels = bytes(rng.randrange(256) for _ in range(96 * 96 * 3))
    buffer = io.BytesIO()
    Image.frombytes("RGB", (96, 96), pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_abo_models_accept_plus_in_official_image_ids() -> None:
    listing = ABOListingRecord(
        item_id="B002V1H37Y",
        main_image_id="71OkUYGf+eL",
        other_image_id=("61Kp5bPJkhL",),
    )

    assert listing.main_image_id == "71OkUYGf+eL"


@dataclass
class _Fixture:
    root: Path
    raw_cache_root: Path
    listings_path: Path
    image_manifest_path: Path
    direct_license_path: Path
    registry_license_path: Path
    receipt_path: Path
    receipt_sha256: str
    lock_path: Path
    lock_sha256: str
    review_path: Path
    review_sha256: str
    lock: ABOSourceLock
    receipt: ABOAcquisitionReceipt
    listings: tuple[ABOListingRecord, ...]
    images: tuple[ABOImageManifestRecord, ...]
    reviews: tuple[ABOImageReviewDecision, ...]
    image_bytes_by_id: dict[str, bytes]
    source_approval: VerifiedABOSourceApproval
    required_raw_root: Path
    required_lock_path: Path
    required_lock_sha256: str
    license_evidence_path: Path
    source_review_policy_path: Path
    source_review_policy_sha256: str
    source_review_portfolio_sha256: str
    source_review_ledger_path: Path
    source_review_ledger_sha256: str

    def source_kwargs(self) -> dict:
        return {
            "source_lock_path": self.lock_path,
            "expected_source_lock_sha256": self.lock_sha256,
            "acquisition_receipt_path": self.receipt_path,
            "expected_acquisition_receipt_sha256": self.receipt_sha256,
            "raw_cache_root": self.raw_cache_root,
            "listings_path": self.listings_path,
            "image_manifest_path": self.image_manifest_path,
            "direct_license_path": self.direct_license_path,
            "registry_license_path": self.registry_license_path,
        }

    def build_kwargs(self, output_name: str = "candidate-bundle") -> dict:
        return {
            "source_approval": self.source_approval,
            **self.source_kwargs(),
            "review_ledger_path": self.review_path,
            "expected_review_ledger_sha256": self.review_sha256,
            "output_dir": self.root / output_name,
            "minimum_pair_candidates": 1,
            "maximum_pair_candidates": len(self.listings),
        }

    def approval_kwargs(self) -> dict:
        return {
            "required_source_lock_path": self.required_lock_path,
            "expected_required_source_lock_sha256": self.required_lock_sha256,
            "required_source_raw_root": self.required_raw_root,
            "license_evidence_path": self.license_evidence_path,
            "source_review_policy_path": self.source_review_policy_path,
            "expected_source_review_policy_sha256": (self.source_review_policy_sha256),
            "expected_source_review_portfolio_sha256": (
                self.source_review_portfolio_sha256
            ),
            "source_review_ledger_path": self.source_review_ledger_path,
            "expected_source_review_ledger_sha256": (self.source_review_ledger_sha256),
        }


def _make_fixture(
    tmp_path: Path,
    *,
    item_count: int = 1,
    third_view: bool = False,
    reject_third_view: bool = False,
    duplicate_second_query: bool = False,
    duplicate_second_gallery: bool = False,
    write_original_images: bool = True,
) -> _Fixture:
    raw_cache_root = tmp_path / "raw-cache"
    raw_cache_root.mkdir(parents=True)
    listing_lines: list[bytes] = []
    metadata_lines: list[bytes] = [b"image_id,height,width,path\n"]
    listings: list[ABOListingRecord] = []
    images: list[ABOImageManifestRecord] = []
    receipt_items: list[ABOItemAcquisitionReceipt] = []
    image_bytes_by_id: dict[str, bytes] = {}

    listing_uri = (
        "https://amazon-berkeley-objects.s3.amazonaws.com/"
        "listings/metadata/listings_0.json.gz"
    )
    metadata_uri = (
        "https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz"
    )
    listing_cache_path = "raw/listings/listings_0.json.gz"
    metadata_cache_path = "raw/images/images.csv.gz"

    for item_index in range(1, item_count + 1):
        item_id = f"ITEM-{item_index:03d}"
        main_id = f"MAIN-{item_index:03d}"
        other_ids = [f"OTHER-{item_index:03d}-A"]
        if third_view:
            other_ids.append(f"OTHER-{item_index:03d}-B")
        listing_payload = {
            "item_id": item_id,
            "main_image_id": main_id,
            "other_image_id": other_ids,
            "item_name": [{"language_tag": "en_US", "value": "fixture"}],
        }
        listing_line = (
            json.dumps(listing_payload, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        listing_lines.append(listing_line)
        listing = ABOListingRecord(
            item_id=item_id,
            main_image_id=main_id,
            other_image_id=tuple(sorted(other_ids)),
        )
        listings.append(listing)

        receipt_images = []
        for image_position, image_id in enumerate([main_id, *sorted(other_ids)]):
            role = "main" if image_id == main_id else "other"
            metadata_path = f"{item_index:02d}/{image_id}.png"
            if duplicate_second_query and item_index == 2 and role == "main":
                image_bytes = image_bytes_by_id["MAIN-001"]
            elif (
                duplicate_second_gallery
                and item_index == 2
                and image_id == "OTHER-002-A"
            ):
                image_bytes = image_bytes_by_id["OTHER-001-A"]
            else:
                image_bytes = _image_bytes(item_index * 20 + image_position + 1)
            image_bytes_by_id[image_id] = image_bytes
            metadata_line = f"{image_id},96,96,{metadata_path}\n".encode()
            metadata_lines.append(metadata_line)
            official_uri = (
                "https://amazon-berkeley-objects.s3.amazonaws.com/"
                f"images/original/{metadata_path}"
            )
            etag = f'"etag-{image_id}"'
            images.append(
                ABOImageManifestRecord(
                    image_id=image_id,
                    path=metadata_path,
                    official_image_uri=official_uri,
                    official_image_etag=etag,
                    source_image_sha256=_sha(image_bytes),
                )
            )
            receipt_images.append(
                ABOOfficialImageReceipt(
                    item_id=item_id,
                    image_id=image_id,
                    image_role=role,
                    metadata_locator=ABORawRecordLocator(
                        artifact_uri=metadata_uri,
                        decompressed_line_number=len(metadata_lines),
                        record_bytes_sha256=_sha(metadata_line),
                    ),
                    metadata_path=metadata_path,
                    official_image_uri=official_uri,
                    image_etag=etag,
                    source_image_sha256=_sha(image_bytes),
                    cache_relative_path=f"images/original/{metadata_path}",
                )
            )
            if write_original_images:
                destination = raw_cache_root / "images" / "original" / metadata_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(image_bytes)
        receipt_items.append(
            ABOItemAcquisitionReceipt(
                item_id=item_id,
                listing_locator=ABORawRecordLocator(
                    artifact_uri=listing_uri,
                    decompressed_line_number=len(listing_lines),
                    record_bytes_sha256=_sha(listing_line),
                ),
                images=tuple(receipt_images),
            )
        )

    listing_gzip = gzip.compress(b"".join(listing_lines), mtime=0)
    metadata_gzip = gzip.compress(b"".join(metadata_lines), mtime=0)
    listing_raw_path = raw_cache_root / listing_cache_path
    metadata_raw_path = raw_cache_root / metadata_cache_path
    listing_raw_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_raw_path.parent.mkdir(parents=True, exist_ok=True)
    listing_raw_path.write_bytes(listing_gzip)
    metadata_raw_path.write_bytes(metadata_gzip)
    listing_artifact = ABORawArtifactReceipt(
        artifact_kind="listing_shard",
        official_uri=listing_uri,
        etag='"listing-etag-v1"',
        raw_sha256=_sha(listing_gzip),
        cache_relative_path=listing_cache_path,
    )
    metadata_artifact = ABORawArtifactReceipt(
        artifact_kind="image_metadata",
        official_uri=metadata_uri,
        etag='"images-csv-etag-v1"',
        raw_sha256=_sha(metadata_gzip),
        cache_relative_path=metadata_cache_path,
    )
    receipt = ABOAcquisitionReceipt(
        source_revision="abo-2022-11-15",
        archive_identity="s3-etag-abo-2022-11-15",
        listing_artifact=listing_artifact,
        image_metadata_artifact=metadata_artifact,
        items=tuple(receipt_items),
    )
    source = tmp_path / "source-contract"
    source.mkdir()
    receipt_path = source / "acquisition-receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    listings_tuple = tuple(listings)
    images_tuple = tuple(sorted(images, key=lambda value: value.image_id))
    listings_path = source / "listings.jsonl"
    image_manifest_path = source / "image-manifest.jsonl"
    listings_path.write_bytes(canonical_jsonl_bytes(listings_tuple))
    image_manifest_path.write_bytes(canonical_jsonl_bytes(images_tuple))

    direct_license_path = source / "LICENSE.direct.txt"
    registry_license_path = source / "LICENSE.registry.txt"
    direct_license_path.write_bytes(b"Official ABO LICENSE: CC BY 4.0\n")
    registry_license_path.write_bytes(b"AWS Registry: CC BY-NC 4.0\n")
    commit = "a" * 40
    lock = ABOSourceLock(
        source_revision=receipt.source_revision,
        archive_identity=receipt.archive_identity,
        acquisition_receipt_sha256=_sha(receipt_path.read_bytes()),
        listing_artifact=listing_artifact,
        image_metadata_artifact=metadata_artifact,
        listings_sha256=_sha(listings_path.read_bytes()),
        image_manifest_sha256=_sha(image_manifest_path.read_bytes()),
        direct_license_sha256=_sha(direct_license_path.read_bytes()),
        registry_license_sha256=_sha(registry_license_path.read_bytes()),
        direct_license_id="CC-BY-4.0",
        registry_license_id="CC-BY-NC-4.0",
        dataset_uri="https://amazon-berkeley-objects.s3.amazonaws.com/index.html",
        archive_uri=(
            "https://s3.us-east-1.amazonaws.com/amazon-berkeley-objects/"
            "archives/abo-images-original.tar"
        ),
        direct_license_uri=(
            "https://raw.githubusercontent.com/amazon-science/"
            f"amazon-berkeley-objects/{commit}/LICENSE.txt"
        ),
        registry_license_uri=("https://registry.opendata.aws/amazon-berkeley-objects/"),
        attribution="Amazon Berkeley Objects dataset creators",
        permissions=ABOPermissionMatrix(),
    )
    lock_path = source / "source-lock.json"
    lock_path.write_bytes(canonical_json_bytes(lock))

    listing_hashes = {
        listing.item_id: _sha(canonical_json_bytes(listing))
        for listing in listings_tuple
    }
    image_hashes = {
        image.image_id: _sha(canonical_json_bytes(image)) for image in images_tuple
    }
    reviews = []
    for item in receipt.items:
        for image in item.images:
            is_third = image.image_id.endswith("-B")
            reviews.append(
                ABOImageReviewDecision(
                    item_id=item.item_id,
                    image_id=image.image_id,
                    listing_record_sha256=listing_hashes[item.item_id],
                    image_record_sha256=image_hashes[image.image_id],
                    source_image_sha256=image.source_image_sha256,
                    decision=(
                        "reject_auxiliary_graphic"
                        if reject_third_view and is_third
                        else "approve_catalog_product_photo"
                    ),
                    reviewer_kind="human",
                    reviewer_id="synthetic-test-reviewer",
                    reviewed_at="2026-07-21T01:02:03Z",
                )
            )
    reviews_tuple = tuple(
        sorted(reviews, key=lambda value: (value.item_id, value.image_id))
    )
    review_path = source / "human-review.jsonl"
    review_path.write_bytes(canonical_jsonl_bytes(reviews_tuple))

    required_paths = tuple(
        sorted(
            path.relative_to(raw_cache_root).as_posix()
            for path in raw_cache_root.rglob("*")
            if path.is_file()
        )
    )
    required_scope = build_artifact_scope(
        raw_cache_root,
        scope_id="formal_abo_inputs",
        mode="explicit_files",
        paths=required_paths,
    )
    identities = []
    receipt_by_path = {
        receipt.listing_artifact.cache_relative_path: (
            receipt.listing_artifact.official_uri,
            receipt.listing_artifact.etag,
        ),
        receipt.image_metadata_artifact.cache_relative_path: (
            receipt.image_metadata_artifact.official_uri,
            receipt.image_metadata_artifact.etag,
        ),
    }
    for item in receipt.items:
        for image in item.images:
            receipt_by_path[image.cache_relative_path] = (
                image.official_image_uri,
                image.image_etag,
            )
    for logical_path in required_paths:
        path = raw_cache_root / logical_path
        url, etag = receipt_by_path[logical_path]
        identities.append(
            AcquisitionIdentity(
                logical_path=logical_path,
                url=url,
                bytes=path.stat().st_size,
                etag=etag,
                local_sha256=_sha(path.read_bytes()),
            )
        )
    required_lock = RequiredSourceLock(
        source_id="abo",
        source_revision="formal-abo-fixture-2026-07-25",
        lock_plan_sha256="4" * 64,
        artifact_scopes=(required_scope,),
        acquisition_identities=tuple(identities),
    )
    required_lock_path = source / "required-source-lock.json"
    required_lock_path.write_bytes(canonical_json_bytes(required_lock))
    required_lock_sha256 = _sha(required_lock_path.read_bytes())

    license_evidence = {
        "schema_version": 1,
        "source_id": "abo",
        "license_id": "CC-BY-NC-4.0",
        "evidence_status": "explicit_dataset_license",
    }
    license_evidence_path = source / "source-review-license-evidence.json"
    license_evidence_path.write_bytes(canonical_json_bytes(license_evidence))
    license_evidence_sha256 = _sha(license_evidence_path.read_bytes())
    portfolio_sha256 = "5" * 64
    policy = SourceReviewPolicy(
        policy_id="abo-formal-fixture-v1",
        portfolio_sha256=portfolio_sha256,
        requirements=(
            SourceReviewRequirement(
                source_id="abo",
                required=True,
                purposes=ABO_FORMAL_PURPOSES,
                required_permissions=ABO_FORMAL_PERMISSIONS,
                pii_review="not_applicable",
            ),
        ),
    )
    policy_path = source / "source-review-policy.json"
    policy_path.write_bytes(canonical_json_bytes(policy))
    policy_sha256 = _sha(policy_path.read_bytes())
    source_permissions = SourcePermissions(
        download_allowed=True,
        local_research_allowed=True,
        local_embedding_allowed=True,
        remote_embedding_allowed=True,
        redistribution_allowed=False,
        public_demo_allowed=True,
    )
    source_review = SourceReviewRecord(
        source_id="abo",
        source_revision=required_lock.source_revision,
        source_lock_sha256=required_lock_sha256,
        license_id="CC-BY-NC-4.0",
        license_evidence_sha256=license_evidence_sha256,
        decision="approved",
        reviewer_id="project-owner",
        reviewed_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
        purposes=ABO_FORMAL_PURPOSES,
        permissions=source_permissions,
        pii_status="not_applicable",
        notes="Fixture owner approval for local formal testing.",
    )
    source_review_ledger_path = source / "owner-source-review-ledger.jsonl"
    source_review_ledger_path.write_bytes(canonical_jsonl_bytes((source_review,)))
    source_review_ledger_sha256 = _sha(source_review_ledger_path.read_bytes())
    source_approval = load_verified_abo_source_approval(
        required_source_lock_path=required_lock_path,
        expected_required_source_lock_sha256=required_lock_sha256,
        required_source_raw_root=raw_cache_root,
        license_evidence_path=license_evidence_path,
        source_review_policy_path=policy_path,
        expected_source_review_policy_sha256=policy_sha256,
        expected_source_review_portfolio_sha256=portfolio_sha256,
        source_review_ledger_path=source_review_ledger_path,
        expected_source_review_ledger_sha256=source_review_ledger_sha256,
    )
    return _Fixture(
        root=tmp_path,
        raw_cache_root=raw_cache_root,
        listings_path=listings_path,
        image_manifest_path=image_manifest_path,
        direct_license_path=direct_license_path,
        registry_license_path=registry_license_path,
        receipt_path=receipt_path,
        receipt_sha256=_sha(receipt_path.read_bytes()),
        lock_path=lock_path,
        lock_sha256=_sha(lock_path.read_bytes()),
        review_path=review_path,
        review_sha256=_sha(review_path.read_bytes()),
        lock=lock,
        receipt=receipt,
        listings=listings_tuple,
        images=images_tuple,
        reviews=reviews_tuple,
        image_bytes_by_id=image_bytes_by_id,
        source_approval=source_approval,
        required_raw_root=raw_cache_root,
        required_lock_path=required_lock_path,
        required_lock_sha256=required_lock_sha256,
        license_evidence_path=license_evidence_path,
        source_review_policy_path=policy_path,
        source_review_policy_sha256=policy_sha256,
        source_review_portfolio_sha256=portfolio_sha256,
        source_review_ledger_path=source_review_ledger_path,
        source_review_ledger_sha256=source_review_ledger_sha256,
    )


def _write_model(path: Path, value) -> str:
    path.write_bytes(canonical_json_bytes(value))
    return _sha(path.read_bytes())


def _verified_candidate(fixture: _Fixture, result):
    source = load_verified_abo_source_bundle(**fixture.source_kwargs())
    return load_verified_abo_candidate_bundle(
        result.output_dir,
        expected_bundle_manifest_sha256=result.bundle_manifest_sha256,
        source_bundle=source,
        source_approval=fixture.source_approval,
        review_ledger_path=fixture.review_path,
        expected_review_ledger_sha256=fixture.review_sha256,
    )


def test_raw_receipt_rederives_identity_and_build_is_explicitly_provisional(tmp_path):
    fixture = _make_fixture(tmp_path)

    source = load_verified_abo_source_bundle(**fixture.source_kwargs())
    result = build_abo_exact_mini(**fixture.build_kwargs())
    verified = _verified_candidate(fixture, result)

    assert source.receipt_sha256 == fixture.receipt_sha256
    assert (
        source.listing_raw_snapshot.sha256
        == fixture.receipt.listing_artifact.raw_sha256
    )
    assert result.audit.eligibility_scope == "local_provisional_only"
    assert result.audit.global_eligibility_status.startswith("not_evaluated")
    assert result.audit.provisional_pair_item_count == 1
    assert result.manifest.bundle_policy_version == "abo-exact-candidate-bundle-v2"
    assert result.manifest.status.startswith("provisional_requires")
    assert result.manifest.required_source_lock_sha256 == fixture.required_lock_sha256
    assert (
        result.manifest.source_review_policy_sha256
        == fixture.source_review_policy_sha256
    )
    assert (
        result.manifest.source_review_ledger_sha256
        == fixture.source_review_ledger_sha256
    )
    assert result.manifest.source_review_license_evidence_sha256 == _sha(
        fixture.license_evidence_path.read_bytes()
    )
    assert all(
        pair.eligibility_status.startswith("provisional_requires")
        for pair in result.pairs
    )
    assert {draft.product_id for draft in result.drafts} == {"abo:ITEM-001"}
    assert all(
        draft.local_path.startswith("images/original/") for draft in result.drafts
    )
    assert verified.manifest_sha256 == result.bundle_manifest_sha256
    assert (
        verified.catalog.catalog_sha256
        == result.manifest.provisional_asset_catalog_sha256
    )
    assert all(
        row.source_collection == "official_original" for row in verified.provenance
    )
    assert all(
        row.acquisition_receipt_sha256 == fixture.receipt_sha256
        for row in verified.provenance
    )


@pytest.mark.parametrize(
    "updates",
    (
        {"source_revision": "different-reviewed-revision-2026-07-25"},
        {"source_lock_sha256": "f" * 64},
        {"license_id": "different-license"},
        {"license_evidence_sha256": "e" * 64},
    ),
)
def test_source_approval_rejects_exact_identity_mismatches(tmp_path, updates):
    fixture = _make_fixture(tmp_path)
    changed = fixture.source_approval.review_record.model_copy(update=updates)
    fixture.source_review_ledger_path.write_bytes(canonical_jsonl_bytes((changed,)))
    kwargs = fixture.approval_kwargs()
    kwargs["expected_source_review_ledger_sha256"] = _sha(
        fixture.source_review_ledger_path.read_bytes()
    )

    with pytest.raises(ABOProvenanceError, match="owner source approval is invalid"):
        load_verified_abo_source_approval(**kwargs)


@pytest.mark.parametrize(
    "permission",
    ("local_embedding_allowed", "remote_embedding_allowed", "public_demo_allowed"),
)
def test_source_approval_rejects_permission_removal(tmp_path, permission):
    fixture = _make_fixture(tmp_path)
    permissions = fixture.source_approval.review_record.permissions.model_copy(
        update={permission: False}
    )
    changed = fixture.source_approval.review_record.model_copy(
        update={"permissions": permissions}
    )
    fixture.source_review_ledger_path.write_bytes(canonical_jsonl_bytes((changed,)))
    kwargs = fixture.approval_kwargs()
    kwargs["expected_source_review_ledger_sha256"] = _sha(
        fixture.source_review_ledger_path.read_bytes()
    )

    with pytest.raises(ABOProvenanceError, match="owner source approval is invalid"):
        load_verified_abo_source_approval(**kwargs)


def test_source_approval_requires_the_externally_pinned_owner_ledger(tmp_path):
    fixture = _make_fixture(tmp_path)
    fixture.source_review_ledger_path.unlink()

    with pytest.raises(ABOProvenanceError, match="source-review ledger"):
        load_verified_abo_source_approval(**fixture.approval_kwargs())


def test_formal_build_rejects_underdeclared_approval_before_output(tmp_path):
    fixture = _make_fixture(tmp_path)
    underdeclared = load_verified_abo_source_approval(
        **fixture.approval_kwargs(),
        permissions=("local_research_allowed",),
    )
    output = tmp_path / "underdeclared"
    kwargs = fixture.build_kwargs(output.name)
    kwargs["source_approval"] = underdeclared

    with pytest.raises(
        ABOProvenanceError,
        match="requires local research and embedding approval",
    ):
        build_abo_exact_mini(**kwargs)
    assert not output.exists()


def test_required_lock_must_cover_every_consumed_original_before_output(tmp_path):
    fixture = _make_fixture(tmp_path)
    raw_metadata_paths = (
        fixture.receipt.listing_artifact.cache_relative_path,
        fixture.receipt.image_metadata_artifact.cache_relative_path,
    )
    narrowed_lock = fixture.source_approval.required_source_lock.lock.model_copy(
        update={
            "artifact_scopes": (
                build_artifact_scope(
                    fixture.required_raw_root,
                    scope_id="formal_abo_inputs",
                    mode="explicit_files",
                    paths=tuple(sorted(raw_metadata_paths)),
                ),
            ),
        }
    )
    fixture.required_lock_path.write_bytes(canonical_json_bytes(narrowed_lock))
    narrowed_lock_sha256 = _sha(fixture.required_lock_path.read_bytes())
    changed_review = fixture.source_approval.review_record.model_copy(
        update={"source_lock_sha256": narrowed_lock_sha256}
    )
    fixture.source_review_ledger_path.write_bytes(
        canonical_jsonl_bytes((changed_review,))
    )
    kwargs = fixture.approval_kwargs()
    kwargs["expected_required_source_lock_sha256"] = narrowed_lock_sha256
    kwargs["expected_source_review_ledger_sha256"] = _sha(
        fixture.source_review_ledger_path.read_bytes()
    )
    narrowed_approval = load_verified_abo_source_approval(**kwargs)
    output = tmp_path / "uncovered-original"
    build_kwargs = fixture.build_kwargs(output.name)
    build_kwargs["source_approval"] = narrowed_approval

    with pytest.raises(
        ABOProvenanceError,
        match="does not cover consumed RAW file",
    ):
        build_abo_exact_mini(**build_kwargs)
    assert not output.exists()


def test_source_approval_toctou_removes_staging_and_publishes_nothing(
    tmp_path, monkeypatch
):
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "approval-toctou"
    original = abo._build_candidate_bundle_manifest

    def mutate_ledger_after_manifest(**kwargs):
        manifest = original(**kwargs)
        fixture.source_review_ledger_path.write_bytes(
            fixture.source_review_ledger_path.read_bytes() + b" "
        )
        return manifest

    monkeypatch.setattr(
        abo,
        "_build_candidate_bundle_manifest",
        mutate_ledger_after_manifest,
    )
    with pytest.raises(
        ABOProvenanceError,
        match="source-review ledger changed during ABO processing",
    ):
        build_abo_exact_mini(**fixture.build_kwargs(output.name))
    assert not output.exists()
    assert not any(tmp_path.glob(".abo-*"))


def test_formal_build_rejects_constructed_ungranted_approval_handle(tmp_path):
    fixture = _make_fixture(tmp_path)
    kwargs = fixture.build_kwargs("ungranted-approval")
    kwargs["source_approval"] = replace(
        fixture.source_approval,
        _verification_token=None,
    )

    with pytest.raises(ABOProvenanceError, match="VerifiedABOSourceApproval"):
        build_abo_exact_mini(**kwargs)
    assert not (tmp_path / "ungranted-approval").exists()


def test_coverage_has_one_complete_disposition_per_candidate_asset(tmp_path):
    fixture = _make_fixture(tmp_path, third_view=True, reject_third_view=True)

    result = build_abo_exact_mini(**fixture.build_kwargs())

    audit = result.audit
    assert audit.candidate_asset_count == 3
    assert len(audit.asset_dispositions) == 3
    assert len({row.asset_key for row in audit.asset_dispositions}) == 3
    assert (
        sum(
            row.disposition == "provisional_candidate"
            for row in audit.asset_dispositions
        )
        == 2
    )
    rejected = [row for row in audit.asset_dispositions if row.image_id.endswith("-B")]
    assert rejected[0].reason == "review_rejected"


def test_standalone_audit_and_missing_manual_approval_fail_closed(tmp_path):
    fixture = _make_fixture(tmp_path)
    report = audit_abo_exact_coverage(
        source_approval=fixture.source_approval,
        **fixture.source_kwargs(),
        review_ledger_path=fixture.review_path,
        expected_review_ledger_sha256=fixture.review_sha256,
        minimum_pair_candidates=1,
        maximum_pair_candidates=1,
    )
    assert report.passed_provisional_gate is True

    retained = tuple(row for row in fixture.reviews if row.image_id.startswith("MAIN"))
    fixture.review_path.write_bytes(canonical_jsonl_bytes(retained))
    kwargs = fixture.build_kwargs("blocked")
    kwargs["expected_review_ledger_sha256"] = _sha(fixture.review_path.read_bytes())
    with pytest.raises(ABOExactCoverageError) as captured:
        build_abo_exact_mini(**kwargs)
    assert captured.value.audit.passed_provisional_gate is False
    assert any(
        row.reason == "missing_human_approval"
        for row in captured.value.audit.asset_dispositions
    )


def test_forged_normalized_relation_cannot_self_attest_formal_identity(tmp_path):
    fixture = _make_fixture(tmp_path)
    forged = ABOListingRecord(
        item_id="ITEM-001",
        main_image_id="OTHER-001-A",
        other_image_id=("MAIN-001",),
    )
    fixture.listings_path.write_bytes(canonical_jsonl_bytes([forged]))
    changed_lock = fixture.lock.model_copy(
        update={"listings_sha256": _sha(fixture.listings_path.read_bytes())}
    )
    fixture.lock_path.write_bytes(canonical_json_bytes(changed_lock))
    kwargs = fixture.source_kwargs()
    kwargs["expected_source_lock_sha256"] = _sha(fixture.lock_path.read_bytes())

    with pytest.raises(ABOProvenanceError, match="raw listing|relation"):
        load_verified_abo_source_bundle(**kwargs)


def test_raw_row_drift_is_detected_before_normalized_data_is_used(tmp_path):
    fixture = _make_fixture(tmp_path)
    raw_path = (
        fixture.raw_cache_root / fixture.receipt.listing_artifact.cache_relative_path
    )
    rows = gzip.decompress(raw_path.read_bytes())
    raw_path.write_bytes(
        gzip.compress(rows.replace(b'"fixture"', b'"drifted"'), mtime=0)
    )

    with pytest.raises(ABOProvenanceError, match="raw listing shard drifted"):
        load_verified_abo_source_bundle(**fixture.source_kwargs())


@pytest.mark.parametrize(
    "uri",
    [
        "https://amazon-berkeley-objects.s3.amazonaws.com.evil.test/images/original/00/X.png",
        "https://s3.us-east-1.amazonaws.com/amazon-berkeley-objects-evil/images/original/00/X.png",
        "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/00/X.png",
        "https://amazon-berkeley-objects.s3.amazonaws.com/spins/00/X.png",
        "https://amazon-berkeley-objects.s3.amazonaws.com/renders/00/X.png",
    ],
)
def test_evil_small_spin_and_render_image_uris_are_rejected(uri):
    with pytest.raises(ValidationError):
        ABOOfficialImageReceipt(
            item_id="ITEM-001",
            image_id="IMAGE-001",
            image_role="main",
            metadata_locator=ABORawRecordLocator(
                artifact_uri=(
                    "https://amazon-berkeley-objects.s3.amazonaws.com/"
                    "images/metadata/images.csv.gz"
                ),
                decompressed_line_number=2,
                record_bytes_sha256="a" * 64,
            ),
            metadata_path="00/X.png",
            official_image_uri=uri,
            image_etag='"etag"',
            source_image_sha256="b" * 64,
            cache_relative_path="images/original/00/X.png",
        )


def test_listing_receipt_uri_must_name_one_exact_official_shard():
    with pytest.raises(ValidationError):
        ABORawArtifactReceipt(
            artifact_kind="listing_shard",
            official_uri=(
                "https://amazon-berkeley-objects.s3.amazonaws.com/"
                "listings/metadata/unreviewed.json.gz"
            ),
            etag='"etag"',
            raw_sha256="a" * 64,
            cache_relative_path="raw/listings/unreviewed.json.gz",
        )


def test_github_license_uri_requires_exact_owner_repo_commit_and_path(tmp_path):
    fixture = _make_fixture(tmp_path)
    bad_values = [
        "https://raw.githubusercontent.com/evil/amazon-berkeley-objects/"
        + "a" * 40
        + "/LICENSE.txt",
        "https://raw.githubusercontent.com/amazon-science/amazon-berkeley-objects-evil/"
        + "a" * 40
        + "/LICENSE.txt",
        "https://raw.githubusercontent.com/amazon-science/amazon-berkeley-objects/main/LICENSE.txt",
        "https://raw.githubusercontent.com/amazon-science/amazon-berkeley-objects/"
        + "a" * 40
        + "/docs/not-a-license.txt",
        "https://raw.githubusercontent.com/amazon-science/amazon-berkeley-objects/"
        + "a" * 40
        + "/docs/LICENSE.txt",
    ]
    for value in bad_values:
        with pytest.raises(ValidationError):
            ABOSourceLock.model_validate(
                {**fixture.lock.model_dump(mode="python"), "direct_license_uri": value}
            )

    official_s3 = (
        "https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-4.0.txt"
    )
    assert (
        ABOSourceLock.model_validate(
            {
                **fixture.lock.model_dump(mode="python"),
                "direct_license_uri": official_s3,
            }
        ).direct_license_uri
        == official_s3
    )
    with pytest.raises(ValidationError):
        ABOSourceLock.model_validate(
            {
                **fixture.lock.model_dump(mode="python"),
                "direct_license_uri": (
                    "https://amazon-berkeley-objects.s3.amazonaws.com/"
                    "nested/LICENSE-CC-BY-4.0.txt"
                ),
            }
        )


class _FakeResponse:
    def __init__(self, content: bytes, etag: str) -> None:
        self.content = content
        self.headers = {"ETag": etag}

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, *, timeout: int) -> _FakeResponse:
        assert timeout == 120
        self.calls.append(url)
        return self.responses[url]


def test_targeted_acquisition_fetches_only_receipt_original_images(tmp_path):
    fixture = _make_fixture(tmp_path, write_original_images=False)
    responses = {
        image.official_image_uri: _FakeResponse(
            fixture.image_bytes_by_id[image.image_id], image.image_etag
        )
        for item in fixture.receipt.items
        for image in item.images
    }
    session = _FakeSession(responses)

    paths = acquire_targeted_abo_original_images(
        acquisition_receipt_path=fixture.receipt_path,
        expected_acquisition_receipt_sha256=fixture.receipt_sha256,
        raw_cache_root=fixture.raw_cache_root,
        session=session,
    )

    assert set(session.calls) == set(responses)
    assert len(paths) == 2
    assert all("images/original" in path.as_posix() for path in paths)
    assert not any("small" in path.as_posix() for path in paths)


def test_review_packet_is_evidence_only_and_does_not_fabricate_manual_ledger(tmp_path):
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "review-packet.jsonl"

    entries = create_abo_candidate_review_packet(
        source_approval=fixture.source_approval,
        **fixture.source_kwargs(),
        output_path=output,
    )

    assert len(entries) == 2
    assert all(entry.packet_status == "human_decision_required" for entry in entries)
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert all("decision" not in row for row in rows)
    assert all("reviewer_id" not in row for row in rows)


def test_local_catalog_cannot_masquerade_as_cross_source_global(tmp_path):
    fixture = _make_fixture(tmp_path)
    result = build_abo_exact_mini(**fixture.build_kwargs())
    local = load_asset_catalog(
        result.output_dir / "provisional-catalog",
        result.output_dir,
        verify_files=True,
    )
    forged_scope = ABOGlobalCatalogScope(
        catalog_sha256=local.catalog_sha256,
        asset_count=local.manifest.asset_count,
        coverage_roots=local.manifest.coverage_roots,
        source_datasets=("abo", "invented-other-source"),
    )
    scope_path = tmp_path / "forged-global-scope.json"
    scope_sha = _write_model(scope_path, forged_scope)

    with pytest.raises(ABOProvenanceError, match="exactly describe"):
        finalize_abo_exact_candidates(
            candidate_bundle=_verified_candidate(fixture, result),
            global_catalog=local,
            global_scope_path=scope_path,
            expected_global_scope_sha256=scope_sha,
            output_path=tmp_path / "finalized.json",
            minimum_finalized_pairs=1,
        )


def _build_global_catalog(result, root: Path, *, other_bytes: bytes | None = None):
    global_root = root / "global-assets"
    for draft in result.drafts:
        source = result.output_dir / draft.local_path
        destination = global_root / draft.local_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    candidate_assets = [
        asset.model_copy(update={"near_duplicate_cluster_id": None})
        for asset in load_asset_catalog(
            result.output_dir / "provisional-catalog",
            result.output_dir,
            verify_files=True,
        ).assets
    ]
    other_path = global_root / "other-source" / "other.png"
    other_path.parent.mkdir(parents=True, exist_ok=True)
    other_path.write_bytes(_image_bytes(999) if other_bytes is None else other_bytes)
    other = inventory_dataset_asset(
        DatasetAssetDraft(
            source_dataset="other-source",
            source_revision="other-2026-07-21",
            source_record_id="other:1",
            local_path="other-source/other.png",
            product_id="other:1",
            license_id="CC0-1.0",
            cloud_upload_allowed=False,
            public_demo_allowed=False,
        ),
        global_root,
    )
    catalog_dir = root / "global-catalog"
    publish_asset_catalog(
        [*candidate_assets, other],
        catalog_dir,
        global_root,
        coverage_roots=["images/original", "other-source"],
    )
    catalog = load_asset_catalog(catalog_dir, global_root, verify_files=True)
    scope = ABOGlobalCatalogScope(
        catalog_sha256=catalog.catalog_sha256,
        asset_count=catalog.manifest.asset_count,
        coverage_roots=catalog.manifest.coverage_roots,
        source_datasets=tuple(
            sorted({source.source_dataset for source in catalog.manifest.sources})
        ),
    )
    scope_path = root / "global-scope.json"
    scope_sha = _write_model(scope_path, scope)
    return catalog, scope_path, scope_sha


def test_clean_cross_source_global_catalog_can_finalize(tmp_path):
    fixture = _make_fixture(tmp_path)
    result = build_abo_exact_mini(**fixture.build_kwargs())
    catalog, scope_path, scope_sha = _build_global_catalog(result, tmp_path)

    output_path = tmp_path / "finalized.json"
    candidate = _verified_candidate(fixture, result)
    finalized = finalize_abo_exact_candidates(
        candidate_bundle=candidate,
        global_catalog=catalog,
        global_scope_path=scope_path,
        expected_global_scope_sha256=scope_sha,
        output_path=output_path,
        minimum_finalized_pairs=1,
    )

    assert finalized.finalized_pair_count == 1
    assert finalized.pair_dispositions[0].reason == "globally_clean"
    verified = load_verified_abo_global_finalization(
        output_path,
        expected_finalization_sha256=_sha(output_path.read_bytes()),
        candidate_bundle=candidate,
        global_catalog=catalog,
        global_scope_path=scope_path,
        expected_global_scope_sha256=scope_sha,
    )
    assert verified.manifest == finalized


def test_global_finalize_rejects_forbidden_component_outside_candidate_pool(tmp_path):
    fixture = _make_fixture(tmp_path)
    result = build_abo_exact_mini(**fixture.build_kwargs())
    duplicated = (result.output_dir / result.drafts[0].local_path).read_bytes()
    catalog, scope_path, scope_sha = _build_global_catalog(
        result, tmp_path, other_bytes=duplicated
    )

    with pytest.raises(ABOGlobalFinalizationError) as captured:
        finalize_abo_exact_candidates(
            candidate_bundle=_verified_candidate(fixture, result),
            global_catalog=catalog,
            global_scope_path=scope_path,
            expected_global_scope_sha256=scope_sha,
            output_path=tmp_path / "finalized.json",
            minimum_finalized_pairs=1,
        )

    assert captured.value.manifest.finalized_pair_count == 0


def test_global_finalize_rejects_query_query_shared_forbidden_component(tmp_path):
    fixture = _make_fixture(tmp_path, item_count=2, duplicate_second_query=True)
    result = build_abo_exact_mini(**fixture.build_kwargs())
    assert result.audit.provisional_pair_item_count == 2
    catalog, scope_path, scope_sha = _build_global_catalog(result, tmp_path)

    with pytest.raises(ABOGlobalFinalizationError) as captured:
        finalize_abo_exact_candidates(
            candidate_bundle=_verified_candidate(fixture, result),
            global_catalog=catalog,
            global_scope_path=scope_path,
            expected_global_scope_sha256=scope_sha,
            output_path=tmp_path / "finalized.json",
            minimum_finalized_pairs=2,
        )

    assert captured.value.manifest.finalized_pair_count == 1
    assert any(
        row.reason == "global_forbidden_leakage"
        for row in captured.value.manifest.pair_dispositions
    )


def test_global_finalize_rejects_gallery_gallery_shared_forbidden_component(tmp_path):
    fixture = _make_fixture(tmp_path, item_count=2, duplicate_second_gallery=True)
    result = build_abo_exact_mini(**fixture.build_kwargs())
    assert result.audit.provisional_pair_item_count == 2
    catalog, scope_path, scope_sha = _build_global_catalog(result, tmp_path)

    with pytest.raises(ABOGlobalFinalizationError) as captured:
        finalize_abo_exact_candidates(
            candidate_bundle=_verified_candidate(fixture, result),
            global_catalog=catalog,
            global_scope_path=scope_path,
            expected_global_scope_sha256=scope_sha,
            output_path=tmp_path / "finalized.json",
            minimum_finalized_pairs=2,
        )

    assert captured.value.manifest.finalized_pair_count == 1
    assert any(
        row.reason == "global_forbidden_leakage"
        for row in captured.value.manifest.pair_dispositions
    )


def test_candidate_asset_disappearance_and_extra_files_fail_bundle_loader(tmp_path):
    fixture = _make_fixture(tmp_path)
    source = load_verified_abo_source_bundle(**fixture.source_kwargs())
    result = build_abo_exact_mini(**fixture.build_kwargs("missing"))
    (result.output_dir / result.drafts[0].local_path).unlink()
    with pytest.raises(ABOProvenanceError, match="file set|missing|drifted"):
        load_verified_abo_candidate_bundle(
            result.output_dir,
            expected_bundle_manifest_sha256=result.bundle_manifest_sha256,
            source_bundle=source,
            source_approval=fixture.source_approval,
            review_ledger_path=fixture.review_path,
            expected_review_ledger_sha256=fixture.review_sha256,
        )

    result = build_abo_exact_mini(**fixture.build_kwargs("extra"))
    (result.output_dir / "unregistered.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ABOProvenanceError, match="file set"):
        load_verified_abo_candidate_bundle(
            result.output_dir,
            expected_bundle_manifest_sha256=result.bundle_manifest_sha256,
            source_bundle=source,
            source_approval=fixture.source_approval,
            review_ledger_path=fixture.review_path,
            expected_review_ledger_sha256=fixture.review_sha256,
        )


def test_candidate_coordinated_rehash_cannot_upgrade_forged_provenance(tmp_path):
    fixture = _make_fixture(tmp_path)
    result = build_abo_exact_mini(**fixture.build_kwargs())
    source = load_verified_abo_source_bundle(**fixture.source_kwargs())
    provenance_path = result.output_dir / "abo-provenance.jsonl"
    rows = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["reviewer_id"] = "forged-but-plausible-human"
    forged_provenance = b"".join(canonical_json_bytes(row) for row in rows)
    provenance_path.write_bytes(forged_provenance)

    manifest_path = result.output_dir / "candidate-bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged_sha = _sha(forged_provenance)
    manifest["provenance_sha256"] = forged_sha
    for descriptor in manifest["files"]:
        if descriptor["path"] == "abo-provenance.jsonl":
            descriptor["bytes"] = len(forged_provenance)
            descriptor["sha256"] = forged_sha
    unsigned = {
        key: value for key, value in manifest.items() if key != "bundle_self_sha256"
    }
    manifest["bundle_self_sha256"] = _sha(canonical_json_bytes(unsigned))
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ABOProvenanceError, match="provenance differs"):
        load_verified_abo_candidate_bundle(
            result.output_dir,
            expected_bundle_manifest_sha256=_sha(manifest_path.read_bytes()),
            source_bundle=source,
            source_approval=fixture.source_approval,
            review_ledger_path=fixture.review_path,
            expected_review_ledger_sha256=fixture.review_sha256,
        )


def test_candidate_loader_rejects_a_constructed_ungranted_source_handle(tmp_path):
    fixture = _make_fixture(tmp_path)
    result = build_abo_exact_mini(**fixture.build_kwargs())
    source = load_verified_abo_source_bundle(**fixture.source_kwargs())

    with pytest.raises(ABOProvenanceError, match="VerifiedABOSourceBundle"):
        load_verified_abo_candidate_bundle(
            result.output_dir,
            expected_bundle_manifest_sha256=result.bundle_manifest_sha256,
            source_bundle=replace(source, _verification_token=None),
            source_approval=fixture.source_approval,
            review_ledger_path=fixture.review_path,
            expected_review_ledger_sha256=fixture.review_sha256,
        )


def test_permissions_distinguish_local_and_remote_embedding(tmp_path):
    fixture = _make_fixture(tmp_path)
    permissions = fixture.lock.permissions

    assert permissions.local_noncommercial_research_allowed is True
    assert permissions.local_noncommercial_embedding_allowed is True
    assert permissions.remote_embedding_allowed is True
    assert permissions.cloud_upload_allowed is True
    assert permissions.redistribution_allowed is False
    assert permissions.public_demo_allowed is True
    assert "embedding_allowed" not in type(permissions).model_fields
    with pytest.raises(ValidationError):
        ABOPermissionMatrix(redistribution_allowed=True)


def test_source_toctou_and_ancestor_symlink_are_rejected(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path / "toctou")
    original = abo._build_candidate_bundle_manifest

    def mutate_then_build(**kwargs):
        raw_path = (
            fixture.raw_cache_root
            / fixture.receipt.image_metadata_artifact.cache_relative_path
        )
        raw_path.write_bytes(raw_path.read_bytes() + b"drift")
        return original(**kwargs)

    monkeypatch.setattr(abo, "_build_candidate_bundle_manifest", mutate_then_build)
    with pytest.raises(ABOProvenanceError, match="changed during ABO processing"):
        build_abo_exact_mini(**fixture.build_kwargs())

    fixture = _make_fixture(tmp_path / "symlink")
    real = fixture.raw_cache_root
    linked = tmp_path / "linked-cache"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")
    kwargs = fixture.source_kwargs()
    kwargs["raw_cache_root"] = linked
    with pytest.raises(ABOProvenanceError, match="reparse|ancestor"):
        load_verified_abo_source_bundle(**kwargs)


def test_noncanonical_receipt_and_coordinated_lock_change_keep_external_root(tmp_path):
    fixture = _make_fixture(tmp_path)
    pretty = json.dumps(fixture.receipt.model_dump(mode="json"), indent=2).encode()
    fixture.receipt_path.write_bytes(pretty)
    changed_lock = fixture.lock.model_copy(
        update={"acquisition_receipt_sha256": _sha(pretty)}
    )
    fixture.lock_path.write_bytes(canonical_json_bytes(changed_lock))

    kwargs = fixture.source_kwargs()
    kwargs["expected_source_lock_sha256"] = _sha(fixture.lock_path.read_bytes())
    kwargs["expected_acquisition_receipt_sha256"] = _sha(pretty)
    with pytest.raises(ABOProvenanceError, match="canonical JSON"):
        load_verified_abo_source_bundle(**kwargs)

    kwargs["expected_acquisition_receipt_sha256"] = fixture.receipt_sha256
    with pytest.raises(ABOProvenanceError, match="independent expected digest"):
        load_verified_abo_source_bundle(**kwargs)


def test_license_conflict_remains_conservative(tmp_path):
    fixture = _make_fixture(tmp_path)
    source = load_verified_abo_source_bundle(**fixture.source_kwargs())

    assert source.lock.direct_license_id == "CC-BY-4.0"
    assert source.lock.registry_license_id == "CC-BY-NC-4.0"
    assert source.lock.license_id == LICENSE_ID
    assert source.lock.license_review_status == "unresolved_conservative_by_nc"
    with pytest.raises(ValidationError):
        ABOSourceLock.model_validate(
            {**fixture.lock.model_dump(mode="python"), "license_id": "CC-BY-4.0"}
        )
