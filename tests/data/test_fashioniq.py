from datetime import UTC, datetime
from io import BytesIO
import json

from PIL import Image
import pytest

from skillchain.data.fashioniq import (
    FROZEN_METADATA_REVISION,
    FashionIQProvenanceError,
    build_fashioniq_image_adapter,
    download_inventory,
    finalize_repeated_failures,
    load_verified_fashioniq_adapter_bundle,
    read_inventory,
    transport_url,
    verify_finalized_acquisition,
)
from skillchain.data.source_lock import RequiredSourceLock, build_artifact_scope
from skillchain.data.source_review import (
    SourcePermissions,
    SourceReviewPolicy,
    SourceReviewRecord,
    SourceReviewRequirement,
    load_source_review_policy,
    load_verified_source_review_ledger,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


def _jpeg_bytes(color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="JPEG")
    return output.getvalue()


def _metadata(tmp_path, rows):
    root = tmp_path / f"fashion-iq-metadata-{FROZEN_METADATA_REVISION}"
    urls = root / "image_url"
    urls.mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        values = rows.get(category, [])
        (urls / f"asin2url.{category}.txt").write_text(
            "".join(f"{asin} \t {url}\n" for asin, url in values),
            encoding="utf-8",
        )
    (urls / "broken_links").mkdir()
    return root


def _formal_adapter_inputs(tmp_path):
    raw_root = tmp_path / "raw"
    fashioniq_root = raw_root / "fashioniq"
    metadata_root = fashioniq_root / f"fashion-iq-metadata-{FROZEN_METADATA_REVISION}"
    image_url_root = metadata_root / "image_url"
    image_url_root.mkdir(parents=True)
    inventories = {
        "dress": [
            ("ACCEPT1", "http://ecx.images-amazon.com/images/I/accept.jpg"),
            ("EXCLUDE1", "http://ecx.images-amazon.com/images/I/exclude.jpg"),
        ],
        "shirt": [],
        "toptee": [],
    }
    for category, rows in inventories.items():
        (image_url_root / f"asin2url.{category}.txt").write_text(
            "".join(f"{asin} {url}\n" for asin, url in rows),
            encoding="utf-8",
        )
    (image_url_root / "broken_links").mkdir()

    annotations_root = fashioniq_root / "fashion-iq-fixture-annotations"
    annotations_root.mkdir()
    (annotations_root / "captions.json").write_text(
        '[{"candidate":"ACCEPT1","target":"EXCLUDE1"}]\n',
        encoding="utf-8",
    )
    images_root = fashioniq_root / "images"
    (images_root / "dress").mkdir(parents=True)
    (images_root / "dress" / "ACCEPT1.jpg").write_bytes(_jpeg_bytes())

    state_root = fashioniq_root / "download-state"
    state_root.mkdir()
    exclusion = {
        "category": "dress",
        "asin": "EXCLUDE1",
        "source_url": "http://ecx.images-amazon.com/images/I/exclude.jpg",
        "decision": "exclude",
        "reason": "owner_directed_exclusion_after_two_identical_complete_retries",
    }
    (state_root / "exclusions.jsonl").write_text(
        json.dumps(exclusion, sort_keys=True) + "\n", encoding="utf-8"
    )
    (state_root / "summary.json").write_text(
        json.dumps({"status": "complete_with_exclusions"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (state_root / "exclusions-summary.json").write_text(
        json.dumps(
            {
                "inventory_total": 2,
                "accepted_images": 1,
                "excluded": 1,
                "status": "complete_with_exclusions",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (state_root / "failures.jsonl").write_text("{}\n", encoding="utf-8")

    scopes = tuple(
        sorted(
            (
                build_artifact_scope(
                    raw_root,
                    scope_id="accepted_images",
                    mode="recursive_tree",
                    root="fashioniq/images",
                ),
                build_artifact_scope(
                    raw_root,
                    scope_id="annotations",
                    mode="recursive_tree",
                    root="fashioniq/fashion-iq-fixture-annotations",
                ),
                build_artifact_scope(
                    raw_root,
                    scope_id="exclusion_ledger",
                    mode="explicit_files",
                    paths=(
                        "fashioniq/download-state/exclusions-summary.json",
                        "fashioniq/download-state/exclusions.jsonl",
                        "fashioniq/download-state/failures.jsonl",
                    ),
                ),
                build_artifact_scope(
                    raw_root,
                    scope_id="url_inventory",
                    mode="recursive_tree",
                    root=(f"fashioniq/fashion-iq-metadata-{FROZEN_METADATA_REVISION}"),
                ),
            ),
            key=lambda scope: scope.scope_id,
        )
    )
    revision = "annotations-fixture_metadata-fixture_exclusions-fixture"
    lock = RequiredSourceLock(
        source_id="fashioniq",
        source_revision=revision,
        lock_plan_sha256="a" * 64,
        artifact_scopes=scopes,
        acquisition_identities=(),
    )
    lock_path = tmp_path / "fashioniq.source-lock.json"
    lock_bytes = canonical_json_bytes(lock.model_dump(mode="json"))
    lock_path.write_bytes(lock_bytes)

    evidence = {
        "schema_version": 1,
        "source_id": "fashioniq",
        "license_id": "LicenseRef-FashionIQ-Academic-Research-Only-2020",
        "assessment_boundary": "fixture",
    }
    evidence_path = tmp_path / "fashioniq.license-evidence.json"
    evidence_bytes = canonical_json_bytes(evidence)
    evidence_path.write_bytes(evidence_bytes)
    return {
        "raw_root": raw_root,
        "lock_path": lock_path,
        "lock_sha256": sha256_bytes(lock_bytes),
        "revision": revision,
        "evidence_path": evidence_path,
        "evidence_sha256": sha256_bytes(evidence_bytes),
        "images_root": images_root,
    }


def _verified_review(
    tmp_path,
    inputs,
    *,
    source_revision=None,
    purposes=("capability_gold", "product_gallery", "tool_gold"),
    local_embedding_allowed=True,
    outbound_allowed=True,
    redistribution_allowed=False,
):
    portfolio_sha256 = "b" * 64
    required_permissions = (
        ("download_allowed", "local_embedding_allowed", "local_research_allowed")
        if local_embedding_allowed
        else ("download_allowed", "local_research_allowed")
    )
    policy = SourceReviewPolicy(
        policy_id="fashioniq-adapter-fixture",
        portfolio_sha256=portfolio_sha256,
        requirements=(
            SourceReviewRequirement(
                source_id="fashioniq",
                required=True,
                purposes=tuple(sorted(purposes)),
                required_permissions=required_permissions,
                pii_review="not_applicable",
            ),
        ),
    )
    policy_path = tmp_path / "policy.json"
    policy_bytes = canonical_json_bytes(policy.model_dump(mode="json"))
    policy_path.write_bytes(policy_bytes)
    loaded_policy = load_source_review_policy(
        policy_path,
        expected_policy_file_sha256=sha256_bytes(policy_bytes),
        expected_portfolio_sha256=portfolio_sha256,
    )
    record = SourceReviewRecord(
        source_id="fashioniq",
        source_revision=source_revision or inputs["revision"],
        source_lock_sha256=inputs["lock_sha256"],
        license_id="LicenseRef-FashionIQ-Academic-Research-Only-2020",
        license_evidence_sha256=inputs["evidence_sha256"],
        decision="approved",
        reviewer_id="fixture-owner",
        reviewed_at=datetime(2026, 7, 25, tzinfo=UTC),
        purposes=tuple(sorted(purposes)),
        permissions=SourcePermissions(
            download_allowed=True,
            local_research_allowed=True,
            local_embedding_allowed=local_embedding_allowed,
            remote_embedding_allowed=outbound_allowed,
            redistribution_allowed=redistribution_allowed,
            public_demo_allowed=outbound_allowed,
        ),
        pii_status="not_applicable",
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_bytes = canonical_jsonl_bytes((record.model_dump(mode="json"),))
    ledger_path.write_bytes(ledger_bytes)
    return load_verified_source_review_ledger(
        ledger_path,
        loaded_policy,
        policy_file_sha256=sha256_bytes(policy_bytes),
        expected_ledger_file_sha256=sha256_bytes(ledger_bytes),
    )


def test_inventory_is_strict_and_transport_uses_tls(tmp_path):
    root = _metadata(
        tmp_path,
        {
            "dress": [
                ("B0001", "http://ecx.images-amazon.com/images/I/example.jpg"),
            ]
        },
    )
    records = read_inventory(root)
    assert len(records) == 1
    assert transport_url(records[0].source_url) == (
        "https://ecx.images-amazon.com/images/I/example.jpg"
    )


def test_download_is_atomic_resumable_and_uses_official_replacement(tmp_path):
    root = _metadata(
        tmp_path,
        {
            "dress": [
                ("REMOTE1", "http://ecx.images-amazon.com/images/I/remote.jpg"),
                ("BROKEN1", "http://g-ecx.images-amazon.com/images/I/broken.jpg"),
            ]
        },
    )
    (root / "image_url" / "broken_links" / "BROKEN1.jpg").write_bytes(
        _jpeg_bytes("blue")
    )
    output = tmp_path / "images"
    state = tmp_path / "state"
    calls = []

    def fetcher(url):
        calls.append(url)
        return _jpeg_bytes()

    first = download_inventory(
        root, output, state, workers=2, progress_every=1, fetcher=fetcher
    )
    assert first["status"] == "complete"
    assert first["counts"] == {
        "downloaded": 1,
        "replacement": 1,
        "skipped": 0,
        "failed": 0,
        "excluded": 0,
    }
    assert calls == ["https://ecx.images-amazon.com/images/I/remote.jpg"]
    assert not list(output.rglob("*.part"))

    second = download_inventory(
        root,
        output,
        state,
        workers=2,
        progress_every=1,
        fetcher=lambda _: (_ for _ in ()).throw(AssertionError("must resume")),
    )
    assert second["counts"]["skipped"] == 2


def test_failure_manifest_is_replaced_on_successful_retry(tmp_path):
    root = _metadata(
        tmp_path,
        {
            "toptee": [
                ("RETRY1", "http://ecx.images-amazon.com/images/I/retry.jpg"),
            ]
        },
    )
    output = tmp_path / "images"
    state = tmp_path / "state"

    failed = download_inventory(
        root,
        output,
        state,
        workers=1,
        progress_every=1,
        fetcher=lambda _: b"not an image",
    )
    assert failed["status"] == "complete_with_failures"
    assert "RETRY1" in (state / "failures.jsonl").read_text(encoding="utf-8")

    recovered = download_inventory(
        root,
        output,
        state,
        workers=1,
        progress_every=1,
        fetcher=lambda _: _jpeg_bytes(),
    )
    assert recovered["status"] == "complete"
    assert (state / "failures.jsonl").read_text(encoding="utf-8") == ""


def test_explicit_exclusion_is_counted_and_not_fetched(tmp_path):
    root = _metadata(
        tmp_path,
        {
            "shirt": [
                ("EXCLUDE1", "http://ecx.images-amazon.com/images/I/exclude.jpg"),
            ]
        },
    )
    output = tmp_path / "images"
    state = tmp_path / "state"
    state.mkdir()
    (state / "exclusions.jsonl").write_text(
        json.dumps(
            {
                "category": "shirt",
                "asin": "EXCLUDE1",
                "source_url": "http://ecx.images-amazon.com/images/I/exclude.jpg",
                "decision": "exclude",
                "reason": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = download_inventory(
        root,
        output,
        state,
        workers=1,
        progress_every=1,
        fetcher=lambda _: (_ for _ in ()).throw(AssertionError("must exclude")),
    )

    assert report["status"] == "complete_with_exclusions"
    assert report["counts"]["excluded"] == 1
    assert report["completed_this_run"] == 1


def test_finalize_requires_identical_complete_failure_sets(tmp_path):
    root = _metadata(
        tmp_path,
        {
            "toptee": [
                ("FAIL1", "http://ecx.images-amazon.com/images/I/fail.jpg"),
            ]
        },
    )
    output = tmp_path / "images"
    state = tmp_path / "state"
    state.mkdir()
    failure = {
        "category": "toptee",
        "asin": "FAIL1",
        "source_url": "http://ecx.images-amazon.com/images/I/fail.jpg",
        "error": "HTTP 404",
    }
    line = json.dumps(failure) + "\n"
    (state / "failures.jsonl").write_text(line, encoding="utf-8")
    (state / "complete-run-fixture.failures.jsonl").write_text(line, encoding="utf-8")
    (state / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete_with_failures",
                "completed_this_run": 1,
                "total": 1,
                "counts": {
                    "downloaded": 0,
                    "replacement": 0,
                    "skipped": 0,
                    "failed": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    report = finalize_repeated_failures(root, output, state)

    assert report["excluded"] == 1
    final = json.loads((state / "summary.json").read_text(encoding="utf-8"))
    assert final["status"] == "complete_with_exclusions"
    assert final["counts"]["failed"] == 0
    assert final["counts"]["excluded"] == 1
    verified = verify_finalized_acquisition(root, output, state)
    assert verified["inventory_total"] == 1
    assert verified["accepted_images"] == 0
    assert verified["excluded"] == 1


def test_formal_adapter_binds_exact_approval_and_keeps_exclusions_out(tmp_path):
    inputs = _formal_adapter_inputs(tmp_path)
    review = _verified_review(tmp_path, inputs)
    output = tmp_path / "adapter-output"

    result = build_fashioniq_image_adapter(
        raw_root=inputs["raw_root"],
        source_lock_path=inputs["lock_path"],
        expected_source_lock_sha256=inputs["lock_sha256"],
        license_evidence_path=inputs["evidence_path"],
        expected_license_evidence_sha256=inputs["evidence_sha256"],
        source_review_ledger=review,
        output_dir=output,
    )

    assert [row.source_record_id for row in result.drafts] == ["dress:ACCEPT1"]
    assert result.drafts[0].local_path == "fashioniq/images/dress/ACCEPT1.jpg"
    assert result.drafts[0].cloud_upload_allowed is True
    assert result.drafts[0].public_demo_allowed is True
    assert {row.source_record_id: row.disposition for row in result.dispositions} == {
        "dress:ACCEPT1": "accepted_asset",
        "dress:EXCLUDE1": "excluded_locked",
    }
    assert result.manifest.source_lock_sha256 == inputs["lock_sha256"]
    assert result.manifest.license_evidence_sha256 == inputs["evidence_sha256"]
    assert result.manifest.excluded_locked == 1
    verified = load_verified_fashioniq_adapter_bundle(
        output, expected_manifest_sha256=result.manifest_sha256
    )
    assert verified.drafts == result.drafts
    assert verified.dispositions == result.dispositions


def test_formal_adapter_does_not_consume_unapproved_runtime_summary(tmp_path):
    inputs = _formal_adapter_inputs(tmp_path)
    review = _verified_review(tmp_path, inputs)
    summary_path = (
        inputs["raw_root"] / "fashioniq" / "download-state" / "summary.json"
    )
    summary_path.write_text("not approved adapter input\n", encoding="utf-8")

    result = build_fashioniq_image_adapter(
        raw_root=inputs["raw_root"],
        source_lock_path=inputs["lock_path"],
        expected_source_lock_sha256=inputs["lock_sha256"],
        license_evidence_path=inputs["evidence_path"],
        expected_license_evidence_sha256=inputs["evidence_sha256"],
        source_review_ledger=review,
        output_dir=tmp_path / "adapter-output",
    )

    assert result.manifest.accepted_assets == 1
    assert result.manifest.excluded_locked == 1


@pytest.mark.parametrize(
    ("review_kwargs", "match"),
    [
        (
            {"source_revision": "different-fashioniq-revision"},
            "source_revision",
        ),
        (
            {"purposes": ("product_gallery",)},
            "purposes",
        ),
        (
            {"local_embedding_allowed": False},
            "permissions",
        ),
        (
            {"outbound_allowed": False},
            "permissions",
        ),
        (
            {"redistribution_allowed": True},
            "forbids redistribution",
        ),
    ],
)
def test_formal_adapter_rejects_stale_or_under_authorized_review(
    tmp_path, review_kwargs, match
):
    inputs = _formal_adapter_inputs(tmp_path)
    review = _verified_review(tmp_path, inputs, **review_kwargs)
    output = tmp_path / "adapter-output"

    with pytest.raises(ValueError, match=match):
        build_fashioniq_image_adapter(
            raw_root=inputs["raw_root"],
            source_lock_path=inputs["lock_path"],
            expected_source_lock_sha256=inputs["lock_sha256"],
            license_evidence_path=inputs["evidence_path"],
            expected_license_evidence_sha256=inputs["evidence_sha256"],
            source_review_ledger=review,
            output_dir=output,
        )

    assert not output.exists()


def test_formal_adapter_rejects_raw_drift_and_bundle_tampering(tmp_path):
    inputs = _formal_adapter_inputs(tmp_path)
    review = _verified_review(tmp_path, inputs)
    accepted_path = inputs["images_root"] / "dress" / "ACCEPT1.jpg"
    accepted_path.write_bytes(_jpeg_bytes("blue"))

    with pytest.raises(ValueError, match="RAW scope differs from lock"):
        build_fashioniq_image_adapter(
            raw_root=inputs["raw_root"],
            source_lock_path=inputs["lock_path"],
            expected_source_lock_sha256=inputs["lock_sha256"],
            license_evidence_path=inputs["evidence_path"],
            expected_license_evidence_sha256=inputs["evidence_sha256"],
            source_review_ledger=review,
            output_dir=tmp_path / "drifted-output",
        )

    clean = _formal_adapter_inputs(tmp_path / "clean")
    clean_review = _verified_review(tmp_path / "clean", clean)
    result = build_fashioniq_image_adapter(
        raw_root=clean["raw_root"],
        source_lock_path=clean["lock_path"],
        expected_source_lock_sha256=clean["lock_sha256"],
        license_evidence_path=clean["evidence_path"],
        expected_license_evidence_sha256=clean["evidence_sha256"],
        source_review_ledger=clean_review,
        output_dir=tmp_path / "clean-output",
    )
    assets_path = result.output_dir / "dataset-assets.jsonl"
    assets_path.write_bytes(assets_path.read_bytes() + b"\n")
    with pytest.raises(FashionIQProvenanceError, match="payload drifted"):
        load_verified_fashioniq_adapter_bundle(
            result.output_dir,
            expected_manifest_sha256=result.manifest_sha256,
        )
