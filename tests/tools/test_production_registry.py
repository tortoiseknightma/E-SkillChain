from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import skillchain.tools.production_registry as production
from skillchain.tools.production_registry import (
    FormalRegistryConfigurationError,
    build_formal_registry_runtime_candidate,
    load_authority_issued_formal_registry,
    load_formal_registry_artifact_lock,
    load_formal_registry_runtime_lock,
    publish_formal_registry_runtime_candidate,
)
from skillchain.tools.registry import (
    MVP_TOOL_NAMES,
    MVP_TOOL_NAMES_V2,
    MVPToolServices,
    build_mvp_registry,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _artifact_lock_payload(registry_sha256: str) -> dict:
    digest = "1" * 64
    return {
        "asset_catalog_path": "catalog/assets",
        "asset_catalog_sha256": digest,
        "asset_root_path": "assets",
        "detector_manifest": {
            "path": "models/detector/manifest.json",
            "sha256": digest,
        },
        "document_safety_catalog_path": "catalog/document-safety",
        "document_safety_catalog_sha256": digest,
        "document_safety_review_ledger_sha256": digest,
        "embedding_model": "qwen3-vl-embedding",
        "embedding_provider": "dashscope",
        "kb_catalog": {"path": "catalog/kb", "sha256": digest},
        "kb_index": {"path": "index/kb", "sha256": digest},
        "ocr_manifest": {
            "path": "models/ocr/manifest.json",
            "sha256": digest,
        },
        "policy_version": "formal-registry-artifacts-v2",
        "product_index": {"path": "index/products", "sha256": digest},
        "product_query_artifact": {
            "path": "queries/split_assignment.jsonl",
            "sha256": digest,
        },
        "schema_version": 2,
        "tool_spec_registry_sha256": registry_sha256,
    }


def _write_artifact_lock(
    path: Path,
    registry_sha256: str,
) -> str:
    content = canonical_json_bytes(_artifact_lock_payload(registry_sha256))
    path.write_bytes(content)
    return sha256_bytes(content)


def _write_real_fixture_artifact_lock(
    path: Path,
    artifact_root: Path,
    formal_fixture,
) -> str:
    product_index = formal_fixture.product_search.index
    fixture_root = product_index.root.parent
    query_path = fixture_root / "queries.jsonl"
    kb_bundle = formal_fixture.kb_lookup.bundle
    detector_artifact = formal_fixture.object_detection.artifact
    ocr_artifact = formal_fixture.document_ocr.artifact
    safety = formal_fixture.safety_catalog

    def relative(value: Path) -> str:
        return value.relative_to(artifact_root).as_posix()

    payload = {
        "asset_catalog_path": relative(formal_fixture.asset_catalog.root),
        "asset_catalog_sha256": safety.manifest.asset_catalog_sha256,
        "asset_root_path": relative(formal_fixture.asset_catalog.asset_root),
        "detector_manifest": {
            "path": relative(detector_artifact.manifest_path),
            "sha256": detector_artifact.manifest.manifest_sha256,
        },
        "document_safety_catalog_path": relative(safety.root),
        "document_safety_catalog_sha256": safety.catalog_sha256,
        "document_safety_review_ledger_sha256": (safety.manifest.review_ledger_sha256),
        "embedding_model": "qwen3-vl-embedding",
        "embedding_provider": "dashscope",
        "kb_catalog": {
            "path": relative(kb_bundle.catalog.root),
            "sha256": kb_bundle.catalog.manifest.catalog_sha256,
        },
        "kb_index": {
            "path": relative(kb_bundle.root),
            "sha256": kb_bundle.manifest.bundle_sha256,
        },
        "ocr_manifest": {
            "path": relative(ocr_artifact.manifest_path),
            "sha256": ocr_artifact.manifest.manifest_sha256,
        },
        "policy_version": "formal-registry-artifacts-v2",
        "product_index": {
            "path": relative(product_index.root),
            "sha256": product_index.external_manifest_sha256,
        },
        "product_query_artifact": {
            "path": relative(query_path),
            "sha256": sha256_bytes(query_path.read_bytes()),
        },
        "schema_version": 2,
        "tool_spec_registry_sha256": formal_fixture.registry.registry_sha256,
    }
    content = canonical_json_bytes(payload)
    path.write_bytes(content)
    return sha256_bytes(content)


def _patch_service_loader(monkeypatch, formal_fixture) -> None:
    services = SimpleNamespace(
        asset_catalog=formal_fixture.asset_catalog,
        document_safety_catalog=formal_fixture.safety_catalog,
        product_search=formal_fixture.product_search,
        kb_lookup=formal_fixture.kb_lookup,
        object_detection=formal_fixture.object_detection,
        document_ocr=formal_fixture.document_ocr,
    )
    monkeypatch.setattr(
        production,
        "_load_locked_services",
        lambda _lock, _root, *, api_key: services,
    )


def _build_v2_registry(formal_fixture):
    return build_mvp_registry(
        MVPToolServices(
            product_search=formal_fixture.product_search,
            kb_lookup=formal_fixture.kb_lookup,
            object_detection=formal_fixture.object_detection,
            document_ocr=formal_fixture.document_ocr,
            safety_approval_for=formal_fixture.safety_catalog.approval_for,
        ),
        include_multi_product=True,
    )


def _set_formal_registry_environment(
    monkeypatch,
    *,
    artifact_lock: Path,
    runtime_lock: Path,
    artifact_root: Path,
    artifact_digest: str,
    runtime_digest: str,
) -> None:
    monkeypatch.setenv(
        "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_LOCK",
        str(artifact_lock),
    )
    monkeypatch.setenv(
        "SKILLCHAIN_FORMAL_REGISTRY_RUNTIME_LOCK",
        str(runtime_lock),
    )
    monkeypatch.setenv(
        "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_ROOT",
        str(artifact_root),
    )
    monkeypatch.setenv(
        "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_LOCK_SHA256",
        artifact_digest,
    )
    monkeypatch.setenv(
        "SKILLCHAIN_FORMAL_REGISTRY_RUNTIME_LOCK_SHA256",
        runtime_digest,
    )


def test_artifact_lock_requires_external_digest_and_canonical_paths(
    tmp_path: Path,
    canonical_registry_factory,
):
    formal = canonical_registry_factory(name="production-lock-schema")
    path = tmp_path / "artifact-lock.json"
    digest = _write_artifact_lock(path, formal.registry.registry_sha256)

    lock, observed = load_formal_registry_artifact_lock(
        path,
        expected_lock_file_sha256=digest,
    )
    assert observed == digest
    assert lock.product_index.path == "index/products"

    with pytest.raises(
        FormalRegistryConfigurationError,
        match="external expected",
    ):
        load_formal_registry_artifact_lock(
            path,
            expected_lock_file_sha256="0" * 64,
        )

    payload = _artifact_lock_payload(formal.registry.registry_sha256)
    payload["product_index"]["path"] = "../index/products"
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(FormalRegistryConfigurationError, match="invalid"):
        load_formal_registry_artifact_lock(
            path,
            expected_lock_file_sha256=sha256_bytes(path.read_bytes()),
        )


def test_local_embedding_track_requires_its_own_locked_manifest(
    tmp_path: Path,
    canonical_registry_factory,
):
    formal = canonical_registry_factory(name="production-local-embedding-lock")
    path = tmp_path / "artifact-lock.json"
    payload = _artifact_lock_payload(formal.registry.registry_sha256)
    payload["embedding_provider"] = "open_clip"
    payload["embedding_model"] = (
        "open-clip:xlm-roberta-large-ViT-H-14:frozen_laion5b_s13b_b90k"
    )
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(FormalRegistryConfigurationError, match="invalid"):
        load_formal_registry_artifact_lock(
            path,
            expected_lock_file_sha256=sha256_bytes(path.read_bytes()),
        )

    payload["embedding_manifest"] = {
        "path": "models/embedding/manifest.json",
        "sha256": "1" * 64,
    }
    path.write_bytes(canonical_json_bytes(payload))
    lock, _digest = load_formal_registry_artifact_lock(
        path,
        expected_lock_file_sha256=sha256_bytes(path.read_bytes()),
    )
    assert lock.embedding_manifest is not None
    assert lock.embedding_provider == "open_clip"


def test_candidate_contains_authority_snapshot_but_exposes_no_live_registry(
    tmp_path: Path,
    canonical_registry_factory,
    monkeypatch,
):
    formal = canonical_registry_factory(name="production-candidate")
    _patch_service_loader(monkeypatch, formal)
    artifact_lock = tmp_path / "artifact-lock.json"
    artifact_digest = _write_artifact_lock(
        artifact_lock,
        formal.registry.registry_sha256,
    )

    candidate = build_formal_registry_runtime_candidate(
        artifact_lock,
        tmp_path,
        expected_artifact_lock_file_sha256=artifact_digest,
    )

    assert candidate.artifact_lock_file_sha256 == artifact_digest
    assert candidate.schema_version == 1
    assert candidate.policy_version == "formal-registry-runtime-v1"
    assert {item.name for item in candidate.tool_runtime_bindings} == MVP_TOOL_NAMES


def test_v2_candidate_and_loader_bind_all_eight_tools(
    tmp_path: Path,
    canonical_registry_factory,
    monkeypatch,
):
    formal = canonical_registry_factory(name="production-v2-candidate")
    registry_v2 = _build_v2_registry(formal)
    registry_v2.require_formal_runtime()
    _patch_service_loader(monkeypatch, formal)
    artifact_lock = tmp_path / "artifact-lock-v2.json"
    artifact_digest = _write_artifact_lock(
        artifact_lock,
        registry_v2.registry_sha256,
    )

    candidate = build_formal_registry_runtime_candidate(
        artifact_lock,
        tmp_path,
        expected_artifact_lock_file_sha256=artifact_digest,
    )

    assert candidate.schema_version == 2
    assert candidate.policy_version == "formal-registry-runtime-v2"
    assert {item.name for item in candidate.tool_runtime_bindings} == (
        MVP_TOOL_NAMES_V2
    )
    evidence = {item.name: item.sha256 for item in candidate.evidence_runtime_bindings}
    assert evidence["multi_product_search"] is not None

    runtime_lock = tmp_path / "runtime-lock-v2.json"
    runtime_lock.write_bytes(canonical_json_bytes(candidate.model_dump(mode="json")))
    runtime_digest = sha256_bytes(runtime_lock.read_bytes())
    loaded, observed = load_formal_registry_runtime_lock(
        runtime_lock,
        expected_lock_file_sha256=runtime_digest,
    )
    assert observed == runtime_digest
    assert loaded == candidate

    issued = load_authority_issued_formal_registry(
        artifact_lock,
        runtime_lock,
        tmp_path,
        expected_artifact_lock_file_sha256=artifact_digest,
        expected_runtime_lock_file_sha256=runtime_digest,
    )
    assert issued.registry.manifest.schema_version == 2
    assert issued.registry.manifest.tool_count == 8
    assert issued.snapshot == candidate.snapshot()

    _set_formal_registry_environment(
        monkeypatch,
        artifact_lock=artifact_lock,
        runtime_lock=runtime_lock,
        artifact_root=tmp_path,
        artifact_digest=artifact_digest,
        runtime_digest=runtime_digest,
    )
    context = production.build_formal_runtime_context_from_env()
    assert context.registry.manifest.schema_version == 2
    assert context.multi_product_executor is None


def test_runtime_lock_generation_fields_cannot_be_mixed(
    tmp_path: Path,
    canonical_registry_factory,
    monkeypatch,
):
    formal = canonical_registry_factory(name="production-v2-lock-mismatch")
    registry_v2 = _build_v2_registry(formal)
    _patch_service_loader(monkeypatch, formal)
    artifact_lock = tmp_path / "artifact-lock-v2.json"
    artifact_digest = _write_artifact_lock(
        artifact_lock,
        registry_v2.registry_sha256,
    )
    candidate = build_formal_registry_runtime_candidate(
        artifact_lock,
        tmp_path,
        expected_artifact_lock_file_sha256=artifact_digest,
    )
    payload = candidate.model_dump(mode="json")
    payload["schema_version"] = 1
    payload["policy_version"] = "formal-registry-runtime-v1"
    runtime_lock = tmp_path / "mixed-runtime-lock.json"
    runtime_lock.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(FormalRegistryConfigurationError, match="invalid"):
        load_formal_registry_runtime_lock(
            runtime_lock,
            expected_lock_file_sha256=sha256_bytes(runtime_lock.read_bytes()),
        )


def test_real_loader_revalidates_every_fixture_artifact_and_runtime(
    tmp_path: Path,
    canonical_registry_factory,
):
    formal = canonical_registry_factory(name="production-real-loader")
    artifact_lock = tmp_path / "real-artifact-lock.json"
    artifact_digest = _write_real_fixture_artifact_lock(
        artifact_lock,
        tmp_path,
        formal,
    )

    candidate = build_formal_registry_runtime_candidate(
        artifact_lock,
        tmp_path,
        expected_artifact_lock_file_sha256=artifact_digest,
        api_key="fixture-key",
    )

    assert candidate.registry_sha256 == formal.registry.registry_sha256
    assert len(candidate.tool_runtime_bindings) == 7
    assert formal.object_detection.backend._model is not None
    assert formal.document_ocr.backend._engine is not None


def test_verified_loader_rebuilds_and_matches_both_external_locks(
    tmp_path: Path,
    canonical_registry_factory,
    monkeypatch,
):
    formal = canonical_registry_factory(name="production-verified")
    _patch_service_loader(monkeypatch, formal)
    artifact_lock = tmp_path / "artifact-lock.json"
    artifact_digest = _write_artifact_lock(
        artifact_lock,
        formal.registry.registry_sha256,
    )
    runtime_lock = tmp_path / "runtime-lock.json"
    published = publish_formal_registry_runtime_candidate(
        artifact_lock,
        tmp_path,
        runtime_lock,
        expected_artifact_lock_file_sha256=artifact_digest,
    )
    runtime_digest = sha256_bytes(runtime_lock.read_bytes())

    issued = load_authority_issued_formal_registry(
        artifact_lock,
        runtime_lock,
        tmp_path,
        expected_artifact_lock_file_sha256=artifact_digest,
        expected_runtime_lock_file_sha256=runtime_digest,
    )

    assert issued.registry.formal_runtime_ready is True
    assert issued.snapshot == published.snapshot()
    assert issued.runtime_lock_file_sha256 == runtime_digest
    assert issued.multi_product.product_search is issued.product_search

    _set_formal_registry_environment(
        monkeypatch,
        artifact_lock=artifact_lock,
        runtime_lock=runtime_lock,
        artifact_root=tmp_path,
        artifact_digest=artifact_digest,
        runtime_digest=runtime_digest,
    )
    context = production.build_formal_runtime_context_from_env()
    assert context.registry.manifest.schema_version == 1
    assert context.multi_product_executor is not None
    assert context.multi_product_executor.product_search is formal.product_search


def test_runtime_lock_rejects_coordinated_rehash_and_runtime_tampering(
    tmp_path: Path,
    canonical_registry_factory,
    monkeypatch,
):
    formal = canonical_registry_factory(name="production-runtime-tamper")
    _patch_service_loader(monkeypatch, formal)
    artifact_lock = tmp_path / "artifact-lock.json"
    artifact_digest = _write_artifact_lock(
        artifact_lock,
        formal.registry.registry_sha256,
    )
    runtime_lock = tmp_path / "runtime-lock.json"
    publish_formal_registry_runtime_candidate(
        artifact_lock,
        tmp_path,
        runtime_lock,
        expected_artifact_lock_file_sha256=artifact_digest,
    )

    artifact_payload = json.loads(artifact_lock.read_text(encoding="utf-8"))
    artifact_payload["product_query_artifact"]["sha256"] = "2" * 64
    artifact_lock.write_bytes(canonical_json_bytes(artifact_payload))
    changed_artifact_digest = sha256_bytes(artifact_lock.read_bytes())
    with pytest.raises(
        FormalRegistryConfigurationError,
        match="does not bind",
    ):
        load_authority_issued_formal_registry(
            artifact_lock,
            runtime_lock,
            tmp_path,
            expected_artifact_lock_file_sha256=changed_artifact_digest,
            expected_runtime_lock_file_sha256=sha256_bytes(runtime_lock.read_bytes()),
        )

    artifact_digest = _write_artifact_lock(
        artifact_lock,
        formal.registry.registry_sha256,
    )
    runtime_payload = json.loads(runtime_lock.read_text(encoding="utf-8"))
    runtime_payload["registry_runtime_sha256"] = "3" * 64
    runtime_lock.write_bytes(canonical_json_bytes(runtime_payload))
    with pytest.raises(
        FormalRegistryConfigurationError,
        match="does not match",
    ):
        load_authority_issued_formal_registry(
            artifact_lock,
            runtime_lock,
            tmp_path,
            expected_artifact_lock_file_sha256=artifact_digest,
            expected_runtime_lock_file_sha256=sha256_bytes(runtime_lock.read_bytes()),
        )


def test_runtime_candidate_publish_is_create_only(
    tmp_path: Path,
    canonical_registry_factory,
    monkeypatch,
):
    formal = canonical_registry_factory(name="production-create-only")
    _patch_service_loader(monkeypatch, formal)
    artifact_lock = tmp_path / "artifact-lock.json"
    artifact_digest = _write_artifact_lock(
        artifact_lock,
        formal.registry.registry_sha256,
    )
    destination = tmp_path / "runtime-lock.json"

    publish_formal_registry_runtime_candidate(
        artifact_lock,
        tmp_path,
        destination,
        expected_artifact_lock_file_sha256=artifact_digest,
    )
    with pytest.raises(FileExistsError):
        publish_formal_registry_runtime_candidate(
            artifact_lock,
            tmp_path,
            destination,
            expected_artifact_lock_file_sha256=artifact_digest,
        )


def test_environment_factory_fails_closed_without_configuration(monkeypatch):
    for name in (
        "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_LOCK",
        "SKILLCHAIN_FORMAL_REGISTRY_RUNTIME_LOCK",
        "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_ROOT",
        "SKILLCHAIN_FORMAL_REGISTRY_ARTIFACT_LOCK_SHA256",
        "SKILLCHAIN_FORMAL_REGISTRY_RUNTIME_LOCK_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        FormalRegistryConfigurationError,
        match="environment is incomplete",
    ):
        production.build_formal_runtime_context_from_env()
