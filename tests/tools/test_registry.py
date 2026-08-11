from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from skillchain.tools.product_search import ProductSearchService
from skillchain.tools.multi_product import (
    MultiProductResult,
    MultiProductSearchService,
)
from skillchain.tools.registry import (
    AssetToolInput,
    DiagnosticToolRegistry,
    EncyclopediaLookupInput,
    MVP_TOOL_NAMES,
    MVP_TOOL_NAMES_V2,
    RecipeLookupInput,
    RegistryError,
    StyleSimilarSearchInput,
    TextProductSearchInput,
    ToolCallError,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    build_mvp_registry_spec,
    load_registry_manifest,
    publish_registry_manifest,
    require_formal_registry_runtime,
)
from skillchain.tools.serialization import canonical_json_bytes


class FakeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    value: str


def _registry(
    *,
    calls: list[tuple[str, dict]] | None = None,
    invalid_output_for: str | None = None,
    runtime_seed: str | None = None,
) -> ToolRegistry:
    calls = calls if calls is not None else []

    def definition(name: str) -> ToolDefinition:
        if name == "style_similar_search":
            input_model = StyleSimilarSearchInput
            binding = "query_asset_and_text_exact"
        elif name in {
            "image_product_search",
            "object_detect",
            "document_ocr",
        }:
            input_model = AssetToolInput
            binding = "query_asset"
        elif name == "text_product_search":
            input_model = TextProductSearchInput
            binding = "query_text_exact"
        elif name == "encyclopedia_lookup":
            input_model = EncyclopediaLookupInput
            binding = "query_derived_text"
        else:
            input_model = RecipeLookupInput
            binding = "query_derived_text"

        def handler(value, context, *, tool_name=name):
            dumped = value.model_dump(mode="json")
            calls.append((tool_name, dumped))
            if invalid_output_for == tool_name:
                return {"value": 123}
            return {"value": f"{context.query_id}:{tool_name}"}

        return ToolDefinition(
            name=name,
            tool_version="2.3.0" if name == "style_similar_search" else "1.0.0",
            description=f"Deterministic {name}",
            input_model=input_model,
            output_type=FakeOutput,
            input_binding=binding,
            network_policy=(
                "runtime_bound_embedding"
                if name
                in {
                    "image_product_search",
                    "text_product_search",
                    "style_similar_search",
                }
                else "offline"
            ),
            output_trust=(
                "untrusted_document_text"
                if name == "document_ocr"
                else "retrieval_evidence"
            ),
            handler=handler,
            runtime_binding_sha256=(
                hashlib.sha256(f"{runtime_seed}:{name}".encode()).hexdigest()
                if runtime_seed is not None
                else None
            ),
        )

    return ToolRegistry([definition(name) for name in sorted(MVP_TOOL_NAMES)])


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    image = tmp_path / "query.png"
    image.write_bytes(b"image")
    return ToolExecutionContext(
        query_id="q-1",
        query_asset_id="asset-1",
        query_text="blue dress",
        resolve_asset=lambda asset_id: image,
    )


def test_registry_requires_exactly_seven_names_and_has_stable_specs():
    first = _registry()
    second = _registry()

    assert {spec.name for spec in first.specs()} == MVP_TOOL_NAMES
    assert first.registry_sha256 == second.registry_sha256
    assert first.manifest == second.manifest
    assert all(
        spec.input_json_schema["additionalProperties"] is False
        for spec in first.specs()
    )

    definitions = []
    with pytest.raises(RegistryError, match="exactly seven"):
        ToolRegistry(definitions)


def test_v2_registry_spec_adds_only_private_multi_product_composite():
    legacy = build_mvp_registry_spec()
    composite = build_mvp_registry_spec(include_multi_product=True)

    assert legacy.manifest.schema_version == 1
    assert legacy.manifest.tool_count == 7
    assert {spec.name for spec in legacy.specs()} == MVP_TOOL_NAMES
    assert composite.manifest.schema_version == 2
    assert composite.manifest.tool_count == 8
    assert {spec.name for spec in composite.specs()} == MVP_TOOL_NAMES_V2
    assert composite.registry_sha256 != legacy.registry_sha256

    style_spec = next(
        item for item in composite.specs() if item.name == "style_similar_search"
    )
    assert style_spec.tool_version == "2.3.0"
    assert "cross-category coordination" in style_spec.description
    assert style_spec.input_binding == "query_asset_and_text_exact"
    assert style_spec.input_json_schema["additionalProperties"] is False
    assert set(style_spec.input_json_schema["properties"]) == {
        "asset_id",
        "query",
    }
    assert set(style_spec.input_json_schema["required"]) == {"asset_id", "query"}

    spec = next(
        item for item in composite.specs() if item.name == "multi_product_search"
    )
    assert spec.tool_version == "1.0.0"
    assert spec.input_binding == "query_asset"
    assert spec.network_policy == "runtime_bound_embedding"
    assert spec.output_trust == "mixed_prediction_and_retrieval_evidence"
    assert spec.input_json_schema["additionalProperties"] is False
    assert set(spec.input_json_schema["properties"]) == {"asset_id"}
    assert set(spec.input_json_schema["required"]) == {"asset_id"}
    assert "crop_path" not in json.dumps(spec.input_json_schema)
    assert "bbox" not in json.dumps(spec.input_json_schema)


def test_runtime_binding_distinguishes_identical_specs_with_different_handlers():
    first = _registry(runtime_seed="runtime-a")
    second = _registry(runtime_seed="runtime-b")

    assert first.registry_sha256 == second.registry_sha256
    assert first.registry_runtime_sha256 != second.registry_runtime_sha256
    assert first.runtime_binding_sha256("image_product_search") != (
        second.runtime_binding_sha256("image_product_search")
    )


def test_registry_validates_arguments_before_handler_and_binds_query_context(
    context: ToolExecutionContext,
):
    calls: list[tuple[str, dict]] = []
    registry = _registry(calls=calls)

    with pytest.raises(ToolCallError) as extra:
        registry.call_tool(
            "object_detect", {"asset_id": "asset-1", "extra": True}, context
        )
    assert extra.value.code == "invalid_arguments"
    with pytest.raises(ToolCallError) as wrong_asset:
        registry.call_tool("object_detect", {"asset_id": "asset-2"}, context)
    assert wrong_asset.value.code == "context_violation"
    with pytest.raises(ToolCallError) as wrong_text:
        registry.call_tool("text_product_search", {"query": "other query"}, context)
    assert wrong_text.value.code == "context_violation"
    assert calls == []

    result = registry.call_tool("text_product_search", {"query": "blue dress"}, context)
    assert result == {"value": "q-1:text_product_search"}
    assert calls == [("text_product_search", {"query": "blue dress"})]


def test_style_23_requires_explicit_authoritative_asset_and_query_before_handler(
    context: ToolExecutionContext,
):
    calls: list[tuple[str, dict]] = []
    registry = _registry(calls=calls)

    with pytest.raises(ToolCallError) as missing_query:
        registry.call_tool(
            "style_similar_search",
            {"asset_id": context.query_asset_id},
            context,
        )
    assert missing_query.value.code == "invalid_arguments"

    with pytest.raises(ToolCallError) as wrong_query:
        registry.call_tool(
            "style_similar_search",
            {
                "asset_id": context.query_asset_id,
                "query": "different query",
            },
            context,
        )
    assert wrong_query.value.code == "context_violation"

    with pytest.raises(ToolCallError) as wrong_asset:
        registry.call_tool(
            "style_similar_search",
            {
                "asset_id": "asset-not-authoritative",
                "query": context.query_text,
            },
            context,
        )
    assert wrong_asset.value.code == "context_violation"
    assert calls == []

    invocation = registry.invoke(
        "style_similar_search",
        {
            "asset_id": context.query_asset_id,
            "query": context.query_text,
        },
        context,
    )
    assert invocation.arguments == {
        "asset_id": context.query_asset_id,
        "query": context.query_text,
    }
    assert calls == [
        (
            "style_similar_search",
            {
                "asset_id": context.query_asset_id,
                "query": context.query_text,
            },
        )
    ]


def test_registry_returns_auditable_hashes_and_canonical_json(
    context: ToolExecutionContext,
):
    registry = _registry()
    invocation = registry.invoke("encyclopedia_lookup", {"entity": "dress"}, context)

    assert invocation.arguments == {"entity": "dress"}
    assert len(invocation.arguments_sha256) == 64
    assert len(invocation.output_sha256) == 64
    assert invocation.output_bytes == b'{"value":"q-1:encyclopedia_lookup"}\n'
    assert (
        registry.call_tool_json("encyclopedia_lookup", {"entity": "dress"}, context)
        == '{"value":"q-1:encyclopedia_lookup"}\n'
    )


def test_registry_rejects_unknown_tools_bad_json_and_invalid_output(
    context: ToolExecutionContext,
):
    registry = _registry(invalid_output_for="recipe_lookup")

    with pytest.raises(ToolCallError) as unknown:
        registry.call_tool("missing", {}, context)
    assert unknown.value.code == "unknown_tool"
    with pytest.raises(ToolCallError) as nonfinite:
        registry.call_tool("recipe_lookup", {"dish": math.nan}, context)
    assert nonfinite.value.code == "invalid_arguments"
    with pytest.raises(ToolCallError) as invalid_output:
        registry.call_tool("recipe_lookup", {"dish": "braised pork"}, context)
    assert invalid_output.value.code == "invalid_output"


def test_remote_asset_resolution_requires_explicit_catalog_permission(
    context: ToolExecutionContext,
):
    with pytest.raises(ToolCallError) as denied:
        context.remote_asset_path(context.query_asset_id)
    assert denied.value.code == "permission_denied"

    allowed = ToolExecutionContext(
        query_id=context.query_id,
        query_asset_id=context.query_asset_id,
        query_text=context.query_text,
        resolve_asset=context.resolve_asset,
        query_cloud_upload_allowed=True,
    )
    assert allowed.remote_asset_path(allowed.query_asset_id).is_file()


def test_registry_manifest_is_create_only_and_detects_tampering(
    tmp_path: Path,
):
    registry = _registry()
    path = tmp_path / "registry.json"

    publish_registry_manifest(registry, path)
    assert load_registry_manifest(path, expected_registry=registry) == registry.manifest
    with pytest.raises(FileExistsError):
        publish_registry_manifest(registry, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["tools"][0]["description"] = "tampered"
    path.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(RegistryError, match="schema"):
        load_registry_manifest(path)


def test_registry_manifest_rejects_duplicate_keys_and_symlink(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key"):
        load_registry_manifest(duplicate)

    if not hasattr(os, "symlink"):
        return
    target = tmp_path / "target.json"
    publish_registry_manifest(_registry(), target)
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not permitted")
    with pytest.raises(Exception, match="non-symlink"):
        load_registry_manifest(link)


def test_diagnostic_registry_cannot_gain_formal_authority_by_instance_assignment():
    registry = build_mvp_registry_spec()

    assert type(registry) is DiagnosticToolRegistry
    assert registry.formal_runtime_ready is False
    with pytest.raises(AttributeError):
        registry._canonical_mvp_builder = True  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        registry._formal_runtime_guard = lambda: None  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        registry.formal_runtime_snapshot = lambda: None  # type: ignore[method-assign]
    with pytest.raises(RegistryError, match="reviewed concrete services"):
        require_formal_registry_runtime(registry)


def test_canonical_builder_issues_controlled_formal_runtime_handle(
    canonical_registry_factory,
):
    value = canonical_registry_factory(name="canonical-positive")

    handle = require_formal_registry_runtime(value.registry)
    snapshot = handle.snapshot()
    handle.verify(snapshot)
    assert value.registry.formal_runtime_ready is True
    assert snapshot.registry_runtime_sha256 == value.registry.registry_runtime_sha256


def _canonical_query_context(
    value,
    *,
    authoritative_asset_id: str | None = None,
    cloud_upload_allowed: bool | None = None,
) -> ToolExecutionContext:
    query_asset = next(
        asset
        for asset in value.asset_catalog.assets
        if asset.source_record_id == "query-only"
    )
    asset_id = authoritative_asset_id or query_asset.asset_id

    def resolve(candidate_asset_id: str) -> Path:
        resolution = value.asset_catalog.resolve_asset_id(candidate_asset_id)
        return value.asset_catalog.asset_root / resolution.local_path

    return ToolExecutionContext(
        query_id="fixture-query",
        query_asset_id=asset_id,
        query_text="canonical registry fixture query",
        resolve_asset=resolve,
        query_cloud_upload_allowed=cloud_upload_allowed,
    )


def test_v2_formal_registry_invokes_composite_and_binds_transitive_runtime(
    canonical_registry_factory,
):
    value = canonical_registry_factory(
        name="canonical-v2-positive",
        include_multi_product=True,
    )
    context = _canonical_query_context(value, cloud_upload_allowed=True)
    expected_runtime = MultiProductSearchService(
        value.object_detection,
        value.product_search,
    ).formal_runtime_binding_sha256

    handle = require_formal_registry_runtime(value.registry)
    snapshot = handle.snapshot()
    invocation = value.registry.invoke(
        "multi_product_search",
        {"asset_id": context.query_asset_id},
        context,
    )

    assert value.registry.manifest.schema_version == 2
    assert value.registry.manifest.tool_count == 8
    assert invocation.tool_name == "multi_product_search"
    assert invocation.spec_sha256 == next(
        spec.spec_sha256
        for spec in value.registry.specs()
        if spec.name == "multi_product_search"
    )
    assert isinstance(invocation.output, dict)
    result = MultiProductResult.model_validate_json(
        canonical_json_bytes(invocation.output)
    )
    assert result.input_binding.asset_id == context.query_asset_id
    assert len(result.objects) == 1
    assert result.artifact_binding == value.product_search.artifact_binding
    assert (
        snapshot.tool_runtime_sha256("multi_product_search")
        == expected_runtime
        == snapshot.evidence_runtime_sha256("multi_product_search")
    )
    value.registry.verify_formal_evidence("multi_product_search", result)


def test_v2_composite_rejects_non_authoritative_and_private_crop_arguments(
    canonical_registry_factory,
):
    value = canonical_registry_factory(
        name="canonical-v2-context-gates",
        include_multi_product=True,
    )
    context = _canonical_query_context(value, cloud_upload_allowed=True)

    with pytest.raises(ToolCallError) as wrong_asset:
        value.registry.invoke(
            "multi_product_search",
            {"asset_id": "asset-not-authoritative"},
            context,
        )
    assert wrong_asset.value.code == "context_violation"

    for private_arguments in (
        {"asset_id": context.query_asset_id, "crop_path": "private/crop.png"},
        {
            "asset_id": context.query_asset_id,
            "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
        },
    ):
        with pytest.raises(ToolCallError) as private:
            value.registry.invoke(
                "multi_product_search",
                private_arguments,
                context,
            )
        assert private.value.code == "invalid_arguments"


def test_v2_composite_requires_cloud_upload_permission_before_remote_execution(
    canonical_registry_factory,
):
    value = canonical_registry_factory(
        name="canonical-v2-cloud-gate",
        include_multi_product=True,
    )
    denied_context = _canonical_query_context(
        value,
        cloud_upload_allowed=False,
    )

    with pytest.raises(ToolCallError) as denied:
        value.registry.invoke(
            "multi_product_search",
            {"asset_id": denied_context.query_asset_id},
            denied_context,
        )
    assert denied.value.code == "permission_denied"


def test_formal_handle_rejects_reviewed_service_class_monkeypatch(
    canonical_registry_factory,
    monkeypatch,
):
    value = canonical_registry_factory(name="service-class-monkeypatch")

    monkeypatch.setattr(
        ProductSearchService,
        "trace_text_product_search",
        lambda *_args, **_kwargs: [],
    )

    assert value.registry.formal_runtime_ready is False
    with pytest.raises(RegistryError, match="class implementation changed"):
        require_formal_registry_runtime(value.registry)


def test_formal_handle_rejects_service_instance_monkeypatch(
    canonical_registry_factory,
):
    value = canonical_registry_factory(name="service-instance-monkeypatch")
    value.product_search.trace_text_product_search = lambda _query: []

    assert value.registry.formal_runtime_ready is False
    with pytest.raises(RegistryError, match="service graph changed"):
        require_formal_registry_runtime(value.registry)


def test_formal_handle_rejects_registry_class_monkeypatch(
    canonical_registry_factory,
    monkeypatch,
):
    value = canonical_registry_factory(name="registry-class-monkeypatch")

    monkeypatch.setattr(
        ToolRegistry,
        "formal_runtime_snapshot",
        lambda _registry: None,
    )

    assert value.registry.formal_runtime_ready is False
    with pytest.raises(RegistryError, match="class implementation changed"):
        require_formal_registry_runtime(value.registry)


@pytest.mark.parametrize(
    ("service_name", "loader_name"),
    (
        ("object_detection", "_load_model"),
        ("document_ocr", "_load_engine"),
    ),
)
def test_formal_handle_rejects_model_loader_instance_monkeypatch(
    canonical_registry_factory,
    service_name,
    loader_name,
):
    value = canonical_registry_factory(name=f"{service_name}-loader-monkeypatch")
    service = getattr(value, service_name)
    service.backend.__dict__[loader_name] = lambda *_args, **_kwargs: object()

    assert value.registry.formal_runtime_ready is False
    with pytest.raises(RegistryError, match="service graph changed"):
        require_formal_registry_runtime(value.registry)


def test_formal_handle_rejects_dashscope_session_state_mutation(
    canonical_registry_factory,
):
    value = canonical_registry_factory(name="dashscope-session-mutation")
    client = value.product_search.backend._backend
    client._session.verify = False

    assert value.registry.formal_runtime_ready is False
    with pytest.raises(RegistryError, match="live runtime binding cannot be resolved"):
        require_formal_registry_runtime(value.registry).snapshot()


def test_formal_handle_rejects_dashscope_private_request_monkeypatch(
    canonical_registry_factory,
):
    value = canonical_registry_factory(name="dashscope-request-monkeypatch")
    client = value.product_search.backend._backend
    client.__dict__["_request_embeddings"] = lambda _contents: object()

    assert value.registry.formal_runtime_ready is False
    with pytest.raises(RegistryError, match="service graph changed"):
        require_formal_registry_runtime(value.registry)
