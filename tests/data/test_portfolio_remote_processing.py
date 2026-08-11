from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from skillchain.data import portfolio_remote_processing as remote_processing
from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
)
from skillchain.data.portfolio_remote_processing import (
    PortfolioRemoteProcessingAuthorization,
    PortfolioRemoteProcessingError,
    PortfolioProcessor,
    VerifiedPortfolioRemoteProcessingRuntime,
    load_portfolio_remote_processing_receipt,
    publish_remote_authorized_portfolio_catalog,
    require_verified_portfolio_remote_processing_runtime,
    verify_portfolio_remote_processing_runtime,
    verify_remote_authorized_query_bindings,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def _draft(
    *,
    source_id: str,
    source_revision: str,
    local_path: str,
    cloud_upload_allowed: bool | None,
) -> DatasetAssetDraft:
    return DatasetAssetDraft(
        source_dataset=source_id,
        source_revision=source_revision,
        source_record_id=f"{source_id}-record",
        transform_policy_version="identity-v1",
        local_path=local_path,
        license_id=f"LicenseRef-{source_id}",
        cloud_upload_allowed=cloud_upload_allowed,
        public_demo_allowed=False,
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    asset_root = tmp_path / "assets"
    _write_image(asset_root / "query_images/a.png", (100, 20, 30))
    _write_image(asset_root / "query_images/b.png", (10, 120, 30))
    drafts = (
        _draft(
            source_id="alpha",
            source_revision="alpha-v1",
            local_path="query_images/a.png",
            cloud_upload_allowed=False,
        ),
        _draft(
            source_id="beta",
            source_revision="beta-v1",
            local_path="query_images/b.png",
            cloud_upload_allowed=None,
        ),
    )
    assets = tuple(inventory_dataset_asset(item, asset_root) for item in drafts)
    base_catalog = tmp_path / "catalog-v1"
    publish_asset_catalog(
        assets,
        base_catalog,
        asset_root,
        coverage_roots=("query_images",),
    )
    loaded = load_asset_catalog(base_catalog, asset_root, verify_files=True)
    dataset_assets_path = tmp_path / "dataset-assets.jsonl"
    dataset_assets_content = b"".join(
        canonical_json_bytes(
            {
                "asset_id": asset.asset_id,
                "image_path": asset.local_path,
                "source_dataset": asset.source_dataset,
            }
        )
        for asset in loaded.assets
    )
    dataset_assets_path.write_bytes(dataset_assets_content)
    dataset_assets_sha256 = sha256_bytes(dataset_assets_content)
    source_artifacts = {
        "alpha": "a" * 64,
        "beta": "b" * 64,
    }
    selection = {
        "dataset_assets": {
            "path": dataset_assets_path.name,
            "rows": len(loaded.assets),
            "sha256": dataset_assets_sha256,
        },
        "selections": [
            {
                "cloud_upload_allowed": asset.cloud_upload_allowed,
                "image_path": asset.local_path,
                "image_sha256": asset.sha256,
                "public_demo_allowed": asset.public_demo_allowed,
                "source_artifact_sha256": source_artifacts[asset.source_dataset],
                "source_dataset": asset.source_dataset,
                "source_revision": asset.source_revision,
            }
            for asset in loaded.assets
        ],
    }
    selection_path = tmp_path / "selection.json"
    selection_content = canonical_json_bytes(selection)
    selection_path.write_bytes(selection_content)
    base_manifest_content = (base_catalog / "manifest.json").read_bytes()
    queries = [
        {
            "asset_id": asset.asset_id,
            "image_path": asset.local_path,
            "query_id": f"q-{index}",
        }
        for index, asset in enumerate(loaded.assets, start=1)
    ]
    queries_path = tmp_path / "queries.jsonl"
    queries_content = b"".join(canonical_json_bytes(item) for item in queries)
    queries_path.write_bytes(queries_content)
    plan_path = tmp_path / "dev_mini.json"
    plan_content = canonical_json_bytes(
        {
            "asset_catalog_sha256": loaded.catalog_sha256,
            "queries": queries,
            "scope": "dev_mini",
        }
    )
    plan_path.write_bytes(plan_content)
    authorization = {
        "authorization_id": "test-portfolio-remote-v1",
        "base_catalog_manifest_file_sha256": sha256_bytes(base_manifest_content),
        "base_catalog_sha256": loaded.catalog_sha256,
        "base_dataset_assets_sha256": dataset_assets_sha256,
        "base_selection_manifest_sha256": sha256_bytes(selection_content),
        "cloud_upload_allowed": True,
        "owner_statement": "Owner approves the exact selected test assets.",
        "plan_sha256": sha256_bytes(plan_content),
        "processor_scope": [
            "dashscope-kimi-feedback",
            "dashscope-kimi-judge",
            "dashscope-qwen-assistant",
        ],
        "public_demo_allowed": False,
        "query_artifact_sha256": sha256_bytes(queries_content),
        "redistribution_allowed": False,
        "remote_model_inference_allowed": True,
        "reviewed_at": "2026-07-28T00:00:00Z",
        "reviewer_id": "project-owner",
        "risk_boundary": "Test-only exact-scope authorization.",
        "schema_version": 1,
        "scope": "dev_mini-selected-query-assets",
        "source_decisions": [
            {
                "asset_count": 1,
                "cloud_upload_allowed": True,
                "remote_model_inference_allowed": True,
                "source_artifact_sha256": source_artifacts["alpha"],
                "source_id": "alpha",
                "source_revision": "alpha-v1",
            },
            {
                "asset_count": 1,
                "cloud_upload_allowed": True,
                "remote_model_inference_allowed": True,
                "source_artifact_sha256": source_artifacts["beta"],
                "source_id": "beta",
                "source_revision": "beta-v1",
            },
        ],
        "status": "owner-approved",
        "track": "portfolio",
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_content = canonical_json_bytes(authorization)
    authorization_path.write_bytes(authorization_content)
    return {
        "asset_root": asset_root,
        "authorization_path": authorization_path,
        "authorization_sha256": sha256_bytes(authorization_content),
        "base_catalog": base_catalog,
        "base_loaded": loaded,
        "dataset_assets_path": dataset_assets_path,
        "plan_path": plan_path,
        "queries_path": queries_path,
        "selection_path": selection_path,
    }


def _core_fixture(tmp_path: Path) -> dict[str, object]:
    fixture = _fixture(tmp_path)
    loaded = fixture["base_loaded"]
    asset_root = fixture["asset_root"]
    drafts = tuple(
        DatasetAssetDraft.model_validate(
            {
                name: (
                    1
                    if name == "schema_version"
                    else asset.model_dump(mode="json")[name]
                )
                for name in DatasetAssetDraft.model_fields
                if name in asset.model_dump(mode="json")
            },
            strict=True,
        )
        for asset in loaded.assets
    )
    dataset_assets_content = b"".join(
        canonical_json_bytes(item.model_dump(mode="json")) for item in drafts
    )
    dataset_assets_path = fixture["dataset_assets_path"]
    dataset_assets_path.write_bytes(dataset_assets_content)
    dataset_assets_sha256 = sha256_bytes(dataset_assets_content)
    selections = []
    for asset, draft in zip(loaded.assets, drafts, strict=True):
        selections.append(
            {
                "candidate_id": f"{asset.source_dataset}.core.test",
                "candidate_sha256": sha256_bytes(
                    canonical_json_bytes({"asset_id": asset.asset_id})
                ),
                "capability_bindings": [],
                "destination_path": asset.local_path,
                "draft": draft.model_dump(mode="json"),
                "expected_bytes": (asset_root / Path(asset.local_path)).stat().st_size,
                "expected_sha256": asset.sha256,
                "pool": "exact_match",
                "source_id": asset.source_dataset,
            }
        )
    selection = {
        "dataset_asset_count": len(selections),
        "dataset_assets_sha256": dataset_assets_sha256,
        "formal_eligible": False,
        "formal_status": "non_formal",
        "inventory_sha256": "c" * 64,
        "policy_version": "portfolio-core-asset-preparation-v1",
        "schema_version": 1,
        "selections": selections,
        "source_policy_sha256": "d" * 64,
        "state": "complete",
        "track": "portfolio",
    }
    selection_path = fixture["selection_path"]
    selection_content = canonical_json_bytes(selection)
    selection_path.write_bytes(selection_content)

    queries_path = fixture["queries_path"]
    queries_content = queries_path.read_bytes()
    queries = [json.loads(line) for line in queries_content.splitlines()]
    plan = {
        "asset_catalog_sha256": loaded.catalog_sha256,
        "queries": queries,
        "scope": "core",
    }
    plan_path = fixture["plan_path"]
    plan_content = canonical_json_bytes(plan)
    plan_path.write_bytes(plan_content)

    authorization = {
        "authorization_id": "test-portfolio-core-remote-v1",
        "authorized_asset_count": len(loaded.assets),
        "base_catalog_asset_file_sha256": loaded.manifest.assets.sha256,
        "base_catalog_components_file_sha256": (loaded.manifest.components.sha256),
        "base_catalog_manifest_file_sha256": sha256_bytes(
            (fixture["base_catalog"] / "manifest.json").read_bytes()
        ),
        "base_catalog_permission_counts": {
            "false": 1,
            "true": 0,
            "unknown": 1,
        },
        "base_catalog_sha256": loaded.catalog_sha256,
        "base_dataset_assets_sha256": dataset_assets_sha256,
        "base_selection_manifest_sha256": sha256_bytes(selection_content),
        "cloud_upload_allowed": True,
        "formal_eligible": False,
        "kind": "portfolio-core-remote-processing-owner-authorization",
        "owner_statement": "Owner approves the exact Core v9 catalog.",
        "plan_sha256": sha256_bytes(plan_content),
        "processor_scope": [
            "dashscope-kimi-feedback",
            "dashscope-kimi-judge",
            "dashscope-qwen-assistant",
        ],
        "public_demo_allowed": False,
        "query_artifact_sha256": sha256_bytes(queries_content),
        "query_count": len(queries),
        "query_referenced_unique_asset_count": len(loaded.assets),
        "redistribution_allowed": False,
        "remote_model_inference_allowed": True,
        "reviewed_at": "2026-08-06T23:59:12+08:00",
        "reviewer_id": "project-owner",
        "risk_boundary": "Private Portfolio processing only.",
        "runtime_overlay_status": ("pending-create-only-publication-and-preflight"),
        "schema_version": 1,
        "scope": "core-v9-all-catalog-assets",
        "source_decisions": [
            {
                "asset_count": 1,
                "source_id": "alpha",
                "source_revision": "alpha-v1",
            },
            {
                "asset_count": 1,
                "source_id": "beta",
                "source_revision": "beta-v1",
            },
        ],
        "status": "owner-approved",
        "track": "portfolio",
    }
    authorization_path = fixture["authorization_path"]
    authorization_content = canonical_json_bytes(authorization)
    authorization_path.write_bytes(authorization_content)
    fixture.update(
        {
            "authorization_path": authorization_path,
            "authorization_sha256": sha256_bytes(authorization_content),
            "authorization_value": authorization,
            "dataset_assets_path": dataset_assets_path,
            "plan_path": plan_path,
            "selection_path": selection_path,
        }
    )
    return fixture


def _publish(tmp_path: Path, fixture: dict[str, object]):
    return publish_remote_authorized_portfolio_catalog(
        authorization_file=fixture["authorization_path"],
        expected_authorization_sha256=fixture["authorization_sha256"],
        selection_manifest=fixture["selection_path"],
        base_catalog=fixture["base_catalog"],
        asset_root=fixture["asset_root"],
        output_catalog=tmp_path / "catalog-v2",
        receipt_output=tmp_path / "receipt.json",
    )


def _verify_runtime(
    fixture: dict[str, object],
    result,
    *,
    processor: PortfolioProcessor = "dashscope-qwen-assistant",
    verified_catalogs=None,
):
    return verify_portfolio_remote_processing_runtime(
        authorization_file=fixture["authorization_path"],
        expected_authorization_sha256=fixture["authorization_sha256"],
        receipt_file=result.receipt_path,
        expected_receipt_file_sha256=sha256_bytes(result.receipt_path.read_bytes()),
        selection_manifest=fixture["selection_path"],
        dataset_assets=fixture["dataset_assets_path"],
        base_catalog=fixture["base_catalog"],
        output_catalog=result.output_catalog,
        asset_root=fixture["asset_root"],
        plan_file=fixture["plan_path"],
        queries_file=fixture["queries_path"],
        processor=processor,
        _verified_catalogs=verified_catalogs,
    )


def test_permission_overlay_changes_only_cloud_flag(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _publish(tmp_path, fixture)
    before = fixture["base_loaded"]
    after = load_asset_catalog(
        result.output_catalog,
        fixture["asset_root"],
        verify_files=True,
    )

    assert all(item.cloud_upload_allowed is True for item in after.assets)
    assert after.manifest.unknown_cloud_permission_count == 0
    assert after.manifest.public_demo_allowed_count == 0
    assert before.components == after.components
    assert before.manifest.components.sha256 == after.manifest.components.sha256
    assert [item.asset_id for item in before.assets] == [
        item.asset_id for item in after.assets
    ]
    for old, new in zip(before.assets, after.assets, strict=True):
        assert old.model_dump(exclude={"cloud_upload_allowed"}) == new.model_dump(
            exclude={"cloud_upload_allowed"}
        )

    receipt_file_sha256 = hashlib.sha256(result.receipt_path.read_bytes()).hexdigest()
    receipt = load_portfolio_remote_processing_receipt(
        result.receipt_path,
        expected_sha256=receipt_file_sha256,
    )
    assert receipt.asset_count == 2
    assert receipt.before_permission_counts == {
        "false": 1,
        "true": 0,
        "unknown": 1,
    }
    assert receipt.after_permission_counts == {
        "false": 0,
        "true": 2,
        "unknown": 0,
    }


def test_core_v9_overlay_and_three_processor_preflights(
    tmp_path: Path,
) -> None:
    fixture = _core_fixture(tmp_path)
    result = _publish(tmp_path, fixture)
    authorization = PortfolioRemoteProcessingAuthorization.model_validate_json(
        fixture["authorization_path"].read_bytes(),
        strict=True,
    )
    receipt = load_portfolio_remote_processing_receipt(
        result.receipt_path,
        expected_sha256=sha256_bytes(result.receipt_path.read_bytes()),
    )
    before = fixture["base_loaded"]
    after = load_asset_catalog(
        result.output_catalog,
        fixture["asset_root"],
        verify_files=True,
    )

    assert authorization.scope == "core-v9-all-catalog-assets"
    assert authorization.authorized_asset_count == 2
    assert receipt.asset_count == 2
    assert receipt.public_demo_allowed_count == 0
    assert receipt.after_permission_counts == {
        "false": 0,
        "true": 2,
        "unknown": 0,
    }
    assert before.components == after.components
    for old, new in zip(before.assets, after.assets, strict=True):
        assert old.model_dump(exclude={"cloud_upload_allowed"}) == new.model_dump(
            exclude={"cloud_upload_allowed"}
        )
    for processor in (
        "dashscope-kimi-feedback",
        "dashscope-kimi-judge",
        "dashscope-qwen-assistant",
    ):
        runtime = _verify_runtime(
            fixture,
            result,
            processor=processor,
        )
        assert runtime.authorization.scope == "core-v9-all-catalog-assets"
        assert runtime.processor == processor
        assert (
            require_verified_portfolio_remote_processing_runtime(
                runtime,
                processor=processor,
                catalog_sha256=result.output_catalog_sha256,
            )
            is runtime
        )
    with pytest.raises(FileExistsError):
        _publish(tmp_path, fixture)


def test_runtime_preflight_reuses_only_exact_preverified_catalogs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture = _core_fixture(tmp_path)
    result = _publish(tmp_path, fixture)
    base = fixture["base_loaded"]
    output = load_asset_catalog(
        result.output_catalog,
        fixture["asset_root"],
        verify_files=True,
    )

    def reject_duplicate_catalog_scan(*_args, **_kwargs):
        raise AssertionError("preverified catalogs must not be rescanned")

    monkeypatch.setattr(
        remote_processing,
        "load_asset_catalog",
        reject_duplicate_catalog_scan,
    )
    runtime = _verify_runtime(
        fixture,
        result,
        verified_catalogs=(base, output),
    )

    assert runtime.catalog is output
    with pytest.raises(
        PortfolioRemoteProcessingError,
        match="differs from runtime paths",
    ):
        _verify_runtime(
            fixture,
            result,
            verified_catalogs=(base, base),
        )


def test_core_v9_authorization_rejects_catalog_count_drift(
    tmp_path: Path,
) -> None:
    fixture = _core_fixture(tmp_path)
    authorization_path = fixture["authorization_path"]
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["authorized_asset_count"] = 3
    authorization["base_catalog_permission_counts"] = {
        "false": 2,
        "true": 0,
        "unknown": 1,
    }
    authorization["source_decisions"][0]["asset_count"] = 2
    authorization_path.write_bytes(canonical_json_bytes(authorization))
    fixture["authorization_sha256"] = sha256_bytes(authorization_path.read_bytes())

    with pytest.raises(
        PortfolioRemoteProcessingError,
        match="catalog asset count differs from owner authorization",
    ):
        _publish(tmp_path, fixture)


def test_core_v9_owner_contract_supports_1022_assets() -> None:
    source_counts = {
        "abo": 340,
        "cord": 35,
        "fashioniq": 130,
        "inaturalist": 122,
        "isia_food500": 210,
        "rpc": 125,
        "sroie": 30,
        "wikimedia_commons_documents": 30,
    }
    value = {
        "authorization_id": "portfolio-core-test-v1",
        "authorized_asset_count": 1022,
        "base_catalog_asset_file_sha256": "a" * 64,
        "base_catalog_components_file_sha256": "b" * 64,
        "base_catalog_manifest_file_sha256": "c" * 64,
        "base_catalog_permission_counts": {
            "false": 595,
            "true": 0,
            "unknown": 427,
        },
        "base_catalog_sha256": "d" * 64,
        "base_dataset_assets_sha256": "e" * 64,
        "base_selection_manifest_sha256": "f" * 64,
        "cloud_upload_allowed": True,
        "formal_eligible": False,
        "kind": "portfolio-core-remote-processing-owner-authorization",
        "owner_statement": "Owner approves the exact Core v9 catalog.",
        "plan_sha256": "1" * 64,
        "processor_scope": [
            "dashscope-kimi-feedback",
            "dashscope-kimi-judge",
            "dashscope-qwen-assistant",
        ],
        "public_demo_allowed": False,
        "query_artifact_sha256": "2" * 64,
        "query_count": 1500,
        "query_referenced_unique_asset_count": 867,
        "redistribution_allowed": False,
        "remote_model_inference_allowed": True,
        "reviewed_at": "2026-08-06T23:59:12+08:00",
        "reviewer_id": "project-owner",
        "risk_boundary": "Private Portfolio processing only.",
        "runtime_overlay_status": ("pending-create-only-publication-and-preflight"),
        "schema_version": 1,
        "scope": "core-v9-all-catalog-assets",
        "source_decisions": [
            {
                "asset_count": count,
                "source_id": source_id,
                "source_revision": f"{source_id}-revision",
            }
            for source_id, count in source_counts.items()
        ],
        "status": "owner-approved",
        "track": "portfolio",
    }

    authorization = PortfolioRemoteProcessingAuthorization.model_validate_json(
        canonical_json_bytes(value),
        strict=True,
    )
    assert authorization.authorized_asset_count == 1022
    assert (
        sum(decision.asset_count for decision in authorization.source_decisions) == 1022
    )


def test_core_v9_role_swap_requires_forward_owner_authorization() -> None:
    """v1 remains historical; only v2 can authorize the swapped processors."""

    source_counts = {
        "abo": 340,
        "cord": 35,
        "fashioniq": 130,
        "inaturalist": 122,
        "isia_food500": 210,
        "rpc": 125,
        "sroie": 30,
        "wikimedia_commons_documents": 30,
    }
    common = {
        "authorization_id": "portfolio-core-role-swap-test-v2",
        "authorized_asset_count": 1022,
        "base_catalog_asset_file_sha256": "a" * 64,
        "base_catalog_components_file_sha256": "b" * 64,
        "base_catalog_manifest_file_sha256": "c" * 64,
        "base_catalog_permission_counts": {
            "false": 595,
            "true": 0,
            "unknown": 427,
        },
        "base_catalog_sha256": "d" * 64,
        "base_dataset_assets_sha256": "e" * 64,
        "base_selection_manifest_sha256": "f" * 64,
        "cloud_upload_allowed": True,
        "formal_eligible": False,
        "kind": "portfolio-core-remote-processing-owner-authorization",
        "owner_statement": "Owner approves the exact role-swapped Core scope.",
        "plan_sha256": "1" * 64,
        "processor_scope": [
            "aifast-gemini-judge",
            "dashscope-kimi-feedback",
            "dashscope-qwen-assistant",
        ],
        "public_demo_allowed": False,
        "query_artifact_sha256": "2" * 64,
        "query_count": 1500,
        "query_referenced_unique_asset_count": 867,
        "redistribution_allowed": False,
        "remote_model_inference_allowed": True,
        "reviewed_at": "2026-08-09T18:00:00+08:00",
        "reviewer_id": "project-owner",
        "risk_boundary": "Private Portfolio processing only.",
        "runtime_overlay_status": ("pending-create-only-publication-and-preflight"),
        "scope": "core-v9-all-catalog-assets",
        "source_decisions": [
            {
                "asset_count": count,
                "source_id": source_id,
                "source_revision": f"{source_id}-revision",
            }
            for source_id, count in source_counts.items()
        ],
        "status": "owner-approved",
        "track": "portfolio",
    }

    with pytest.raises(
        ValueError,
        match="schema-versioned owner-approved processor set",
    ):
        PortfolioRemoteProcessingAuthorization.model_validate_json(
            canonical_json_bytes({**common, "schema_version": 1}), strict=True
        )

    authorization = PortfolioRemoteProcessingAuthorization.model_validate_json(
        canonical_json_bytes({**common, "schema_version": 2}), strict=True
    )
    assert authorization.processor_scope == (
        "aifast-gemini-judge",
        "dashscope-kimi-feedback",
        "dashscope-qwen-assistant",
    )


def test_permission_overlay_rejects_unpinned_authorization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["authorization_sha256"] = "0" * 64
    with pytest.raises(
        PortfolioRemoteProcessingError,
        match="authorization SHA-256 mismatch",
    ):
        _publish(tmp_path, fixture)


def test_permission_overlay_rejects_source_scope_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["selection_path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["selections"][0]["source_artifact_sha256"] = "c" * 64
    path.write_bytes(canonical_json_bytes(value))
    authorization_path = fixture["authorization_path"]
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["base_selection_manifest_sha256"] = sha256_bytes(path.read_bytes())
    authorization_path.write_bytes(canonical_json_bytes(authorization))
    fixture["authorization_sha256"] = sha256_bytes(authorization_path.read_bytes())
    with pytest.raises(
        PortfolioRemoteProcessingError,
        match="selection source binding differs",
    ):
        _publish(tmp_path, fixture)


def test_permission_overlay_is_create_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _publish(tmp_path, fixture)
    with pytest.raises(FileExistsError):
        _publish(tmp_path, fixture)


def test_query_bindings_resolve_against_remote_catalog(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _publish(tmp_path, fixture)

    verified = verify_remote_authorized_query_bindings(
        catalog_dir=result.output_catalog,
        asset_root=fixture["asset_root"],
        queries_path=fixture["queries_path"],
        expected_query_artifact_sha256=sha256_bytes(
            fixture["queries_path"].read_bytes()
        ),
    )
    assert verified.query_count == 2
    assert verified.unique_asset_count == 2
    assert verified.catalog_sha256 == result.output_catalog_sha256


def test_runtime_preflight_reverifies_complete_permission_chain(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _publish(tmp_path, fixture)
    runtime = _verify_runtime(fixture, result)

    assert runtime.catalog.catalog_sha256 == result.output_catalog_sha256
    assert runtime.query_artifact_sha256 == sha256_bytes(
        fixture["queries_path"].read_bytes()
    )
    assert (
        require_verified_portfolio_remote_processing_runtime(
            runtime,
            processor="dashscope-qwen-assistant",
            catalog_sha256=result.output_catalog_sha256,
        )
        is runtime
    )
    feedback_runtime = _verify_runtime(
        fixture,
        result,
        processor="dashscope-kimi-feedback",
    )
    assert feedback_runtime.processor == "dashscope-kimi-feedback"


def test_runtime_preflight_rejects_dataset_assets_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _publish(tmp_path, fixture)
    fixture["dataset_assets_path"].write_bytes(
        fixture["dataset_assets_path"].read_bytes()
        + canonical_json_bytes({"unexpected": True})
    )

    with pytest.raises(
        PortfolioRemoteProcessingError,
        match="dataset-assets file SHA-256 mismatch",
    ):
        _verify_runtime(fixture, result)


@pytest.mark.parametrize(
    ("path_key", "message"),
    (
        ("plan_path", "plan SHA-256 mismatch"),
        ("queries_path", "query artifact SHA-256 mismatch"),
    ),
)
def test_runtime_preflight_rejects_plan_or_query_drift(
    tmp_path: Path,
    path_key: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    result = _publish(tmp_path, fixture)
    path = fixture[path_key]
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(PortfolioRemoteProcessingError, match=message):
        _verify_runtime(fixture, result)


def test_runtime_preflight_rejects_processor_outside_owner_scope(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authorization_path = fixture["authorization_path"]
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["processor_scope"] = ["dashscope-qwen-assistant"]
    authorization_path.write_bytes(canonical_json_bytes(authorization))
    fixture["authorization_sha256"] = sha256_bytes(authorization_path.read_bytes())
    result = _publish(tmp_path, fixture)

    with pytest.raises(
        PortfolioRemoteProcessingError,
        match="processor is outside the owner-authorized scope",
    ):
        _verify_runtime(
            fixture,
            result,
            processor="dashscope-kimi-judge",
        )


def test_runtime_handle_cannot_be_forged(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _publish(tmp_path, fixture)
    authorization = PortfolioRemoteProcessingAuthorization.model_validate_json(
        fixture["authorization_path"].read_bytes(),
        strict=True,
    )
    receipt = load_portfolio_remote_processing_receipt(
        result.receipt_path,
        expected_sha256=sha256_bytes(result.receipt_path.read_bytes()),
    )
    forged = VerifiedPortfolioRemoteProcessingRuntime(
        authorization=authorization,
        receipt=receipt,
        processor="dashscope-qwen-assistant",
        catalog=load_asset_catalog(
            result.output_catalog,
            fixture["asset_root"],
            verify_files=True,
        ),
        authorization_file_sha256=fixture["authorization_sha256"],
        receipt_file_sha256=sha256_bytes(result.receipt_path.read_bytes()),
        dataset_assets_sha256=sha256_bytes(fixture["dataset_assets_path"].read_bytes()),
        plan_sha256=sha256_bytes(fixture["plan_path"].read_bytes()),
        query_artifact_sha256=sha256_bytes(fixture["queries_path"].read_bytes()),
    )

    with pytest.raises(
        PortfolioRemoteProcessingError,
        match="verified Portfolio runtime preflight",
    ):
        require_verified_portfolio_remote_processing_runtime(
            forged,
            processor="dashscope-qwen-assistant",
            catalog_sha256=result.output_catalog_sha256,
        )
