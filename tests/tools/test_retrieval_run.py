from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from PIL import Image
import pytest
from pydantic import BaseModel, ConfigDict

from skillchain.data.asset_catalog import (
    DatasetAssetDraft,
    NearDuplicatePolicy,
    inventory_dataset_asset,
    load_asset_catalog,
    publish_asset_catalog,
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
from skillchain.tools import retrieval_run as run_module
from skillchain.tools.contracts import ProductSearchTrace, RetrievalArtifactBinding
from skillchain.tools.retrieval_run import (
    ProvisionalRetrievalRun,
    RegistryRetrievalExecutor,
    RetrievalRequest,
    RetrievalRunError,
    create_provisional_retrieval_run,
    create_registry_retrieval_run,
    load_provisional_retrieval_run,
    load_verified_retrieval_run,
    require_verified_retrieval_run,
)
from skillchain.tools.registry import (
    AssetToolInput,
    EncyclopediaLookupInput,
    MVP_TOOL_NAMES,
    RecipeLookupInput,
    TextProductSearchInput,
    ToolDefinition,
    ToolCallError,
    ToolRegistry,
)
from skillchain.tools.product_search import ProductSearchService

TOOL_SPEC_SHA256 = "a" * 64
TOOL_NAME = "image_product_search"
EXECUTOR_ID = "fixture-executor-v1"


class _RegistryHit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    artifact_binding: RetrievalArtifactBinding
    query_echo: str


def _retrieval_registry(binding: RetrievalArtifactBinding) -> ToolRegistry:
    definitions = []
    for name in sorted(MVP_TOOL_NAMES):
        if name in {
            "image_product_search",
            "style_similar_search",
            "object_detect",
            "document_ocr",
        }:
            input_model = AssetToolInput
            input_binding = "query_asset"
        elif name == "text_product_search":
            input_model = TextProductSearchInput
            input_binding = "query_text_exact"
        elif name == "encyclopedia_lookup":
            input_model = EncyclopediaLookupInput
            input_binding = "query_derived_text"
        else:
            input_model = RecipeLookupInput
            input_binding = "query_derived_text"

        def handler(value, context, *, tool_name=name):
            if tool_name == "image_product_search":
                path = context.asset_path(value.asset_id)
                return ProductSearchTrace(
                    tool_name="image_product_search",
                    input_kind="image",
                    query_input_sha256=sha256_bytes(path.read_bytes()),
                    query_asset_id=value.asset_id,
                    query_vector_sha256="d" * 64,
                    artifact_binding=binding,
                    hits=(),
                )
            return [
                _RegistryHit(
                    artifact_binding=binding,
                    query_echo=context.query_id,
                )
            ]

        definitions.append(
            ToolDefinition(
                name=name,
                tool_version="1.0.0",
                description=f"fixture {name}",
                input_model=input_model,
                output_type=(
                    ProductSearchTrace
                    if name == "image_product_search"
                    else list[_RegistryHit]
                ),
                input_binding=input_binding,
                network_policy="offline",
                output_trust="retrieval_evidence",
                handler=handler,
                runtime_binding_sha256=(
                    "c" * 64 if name == "image_product_search" else None
                ),
            )
        )
    return ToolRegistry(definitions)


def _write_image(path: Path, *, pattern: int) -> None:
    image = Image.new("RGB", (64, 64), "white")
    pixels = image.load()
    assert pixels is not None
    for y in range(64):
        for x in range(64):
            if (
                (pattern == 0 and x < y)
                or (pattern == 1 and x + y < 50)
                or (pattern == 2 and (x // 7 + y // 5) % 2)
            ):
                pixels[x, y] = (0, 0, 0)
    image.save(path, format="PNG")


def _query(catalog, asset_id: str, query_id: str) -> Query:
    resolution = catalog.resolve_asset_id(asset_id)
    intent = "multi_product"
    capability = capability_for_intent(intent)
    text = f"retrieve products for {query_id}"
    return Query(
        schema_version=2,
        taxonomy_version=TAXONOMY_VERSION,
        task_spec_version=TASK_SPEC_VERSION,
        query_id=query_id,
        asset_id=asset_id,
        image_path=resolution.local_path,
        leakage_group_id=resolution.leakage_group_id,
        template_family="retrieval-run-fixture-v1",
        generator_batch_id=f"batch-{query_id}",
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
                annotator_id="retrieval-run-test",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


@pytest.fixture
def evidence(tmp_path: Path):
    asset_root = tmp_path / "assets"
    images = asset_root / "images"
    images.mkdir(parents=True)
    _write_image(images / "first.png", pattern=0)
    _write_image(images / "second.png", pattern=1)
    drafts = (
        DatasetAssetDraft(
            source_dataset="retrieval-fixture",
            source_revision="fixture-v1",
            source_record_id="first",
            local_path="images/first.png",
            product_id="product-first",
            license_id="test-only",
        ),
        DatasetAssetDraft(
            source_dataset="retrieval-fixture",
            source_revision="fixture-v1",
            source_record_id="second",
            local_path="images/second.png",
            product_id="product-second",
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
    # Deliberately not lexical order: the exact authoritative artifact order wins.
    queries = (
        _query(catalog, by_record["second"].asset_id, "q-zulu"),
        _query(catalog, by_record["first"].asset_id, "q-alpha"),
    )
    query_path = tmp_path / "queries.jsonl"
    query_path.write_bytes(canonical_jsonl_bytes(queries))
    binding = RetrievalArtifactBinding(
        mode="verified",
        index_integrity_sha256="1" * 64,
        eligibility_sha256="2" * 64,
        query_artifact_sha256=sha256_bytes(query_path.read_bytes()),
        gallery_artifact_sha256="3" * 64,
        asset_catalog_sha256=catalog.catalog_sha256,
        products_parquet_sha256="4" * 64,
        leakage_policy_version=catalog.leakage_policy_version,
    )
    return {
        "asset_root": asset_root,
        "catalog": catalog,
        "catalog_dir": catalog_dir,
        "queries": queries,
        "query_path": query_path,
        "binding": binding,
        "output": tmp_path / "run",
    }


def _hit(binding: RetrievalArtifactBinding, query_id: str) -> dict:
    return {
        "artifact_binding": binding.model_dump(mode="json"),
        "query_echo": query_id,
        "rank": 1,
        "score": 0.5,
    }


def _create(evidence, executor, *, output: Path | None = None):
    return create_provisional_retrieval_run(
        evidence["query_path"],
        output or evidence["output"],
        evidence["catalog"],
        evidence["binding"],
        executor,
        run_id="formal-run-001",
        tool_name=TOOL_NAME,
        tool_spec_sha256=TOOL_SPEC_SHA256,
        executor_id=EXECUTOR_ID,
    )


def _load(evidence, *, output: Path | None = None):
    return load_provisional_retrieval_run(
        output or evidence["output"],
        evidence["query_path"],
        evidence["catalog"],
        artifact_binding=evidence["binding"],
        tool_name=TOOL_NAME,
        tool_spec_sha256=TOOL_SPEC_SHA256,
        executor_id=EXECUTOR_ID,
    )


def _resign(value: dict, field: str) -> None:
    unsigned = dict(value)
    unsigned.pop(field, None)
    value[field] = sha256_bytes(canonical_json_bytes(unsigned))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_bundle(
    root: Path,
    *,
    executions: list[dict] | None = None,
    calls: list[dict] | None = None,
    raw_spec: bytes | None = None,
) -> None:
    if executions is not None:
        for execution in executions:
            _resign(execution, "execution_sha256")
        (root / "query-executions.jsonl").write_bytes(canonical_jsonl_bytes(executions))
    if calls is not None:
        for call in calls:
            _resign(call, "call_sha256")
        (root / "tool-calls.jsonl").write_bytes(canonical_jsonl_bytes(calls))
    if raw_spec is not None:
        (root / "spec.json").write_bytes(raw_spec)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_executions = _read_jsonl(root / "query-executions.jsonl")
    current_calls = _read_jsonl(root / "tool-calls.jsonl")
    for filename, rows in (
        ("spec.json", 1),
        ("query-executions.jsonl", len(current_executions)),
        ("tool-calls.jsonl", len(current_calls)),
    ):
        content = (root / filename).read_bytes()
        manifest["artifacts"][filename] = {
            "bytes": len(content),
            "path": filename,
            "rows": rows,
            "sha256": sha256_bytes(content),
        }
    manifest["call_count"] = len(current_calls)
    _resign(manifest, "manifest_sha256")
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def test_create_run_uses_authoritative_query_order_and_resolved_assets(evidence):
    requests: list[RetrievalRequest] = []

    def executor(request: RetrievalRequest):
        requests.append(request)
        return [_hit(evidence["binding"], request.query_id)]

    verified = _create(evidence, executor)

    assert isinstance(verified, ProvisionalRetrievalRun)
    assert [query.query_id for query in verified.queries] == ["q-zulu", "q-alpha"]
    assert [execution.query_id for execution in verified.executions] == [
        "q-zulu",
        "q-alpha",
    ]
    assert [request.query_id for request in requests] == ["q-zulu", "q-alpha"]
    for request, query in zip(requests, evidence["queries"], strict=True):
        expected = (
            evidence["catalog"]
            .asset_root.joinpath(*query.image_path.split("/"))
            .resolve()
        )
        assert request.image_path == expected
        assert "image_path" not in request.arguments
        assert request.arguments["query_asset"]["asset_id"] == query.asset_id
    assert verified.manifest.query_count == 2
    assert verified.manifest.success_count == 2
    assert verified.manifest.error_count == 0
    assert set(path.name for path in evidence["output"].iterdir()) == {
        "manifest.json",
        "query-executions.jsonl",
        "spec.json",
        "tool-calls.jsonl",
    }


def test_registry_executor_binds_live_spec_runtime_and_authoritative_asset(evidence):
    registry = _retrieval_registry(evidence["binding"])

    class FormalProductRuntime(ProductSearchService):
        formal_runtime_binding_sha256 = "c" * 64

        def __init__(self):
            pass

    executor = RegistryRetrievalExecutor(
        registry=registry,
        tool_name="image_product_search",
        product_search_service=FormalProductRuntime(),
    )

    verified = create_registry_retrieval_run(
        evidence["query_path"],
        evidence["output"],
        evidence["catalog"],
        evidence["binding"],
        executor,
        run_id="registry-run-001",
    )

    assert verified.spec.tool_spec_sha256 == executor.tool_spec_sha256
    assert verified.spec.executor_id == executor.executor_id
    assert verified.spec.executor_id.startswith("registry.v1.")
    assert [call.output["query_asset_id"] for call in verified.calls] == [
        evidence["queries"][0].asset_id,
        evidence["queries"][1].asset_id,
    ]
    assert {call.output["query_vector_sha256"] for call in verified.calls} == {"d" * 64}
    assert [call.output["query_input_sha256"] for call in verified.calls] == [
        evidence["catalog"].resolve_asset_id(query.asset_id).asset.sha256
        for query in evidence["queries"]
    ]
    reloaded = load_verified_retrieval_run(
        evidence["output"],
        evidence["query_path"],
        evidence["catalog"],
        artifact_binding=evidence["binding"],
        executor=executor,
    )
    assert require_verified_retrieval_run(reloaded) is reloaded


def test_empty_results_keep_a_top_level_verified_binding(evidence):
    verified = _create(evidence, lambda request: [])

    assert [call.output for call in verified.calls] == [[], []]
    assert {call.status for call in verified.calls} == {"ok"}
    assert {call.artifact_binding for call in verified.calls} == {evidence["binding"]}
    assert all(
        call.output_sha256 == sha256_bytes(canonical_json_bytes([]))
        for call in verified.calls
    )


def test_executor_errors_are_hashed_and_do_not_remove_queries(evidence):
    def executor(request: RetrievalRequest):
        if request.query_id == "q-zulu":
            raise RuntimeError("backend unavailable")
        return [_hit(evidence["binding"], request.query_id)]

    verified = _create(evidence, executor)

    assert [execution.status for execution in verified.executions] == ["error", "ok"]
    assert verified.manifest.success_count == 1
    assert verified.manifest.error_count == 1
    error_call = verified.calls[0]
    assert error_call.output == {
        "error": {
            "code": "tool_execution_failed",
            "message": "tool execution failed",
        }
    }
    assert error_call.output_sha256 == sha256_bytes(
        canonical_json_bytes(error_call.output)
    )


def test_error_artifact_uses_only_allowlisted_code_and_fixed_message(evidence):
    def executor(_request: RetrievalRequest):
        raise ToolCallError("invalid_output", "secret path C:/private/data")

    run = _create(evidence, executor)

    assert run.calls[0].output == {
        "error": {
            "code": "invalid_output",
            "message": "tool output is invalid",
        }
    }
    serialized = canonical_json_bytes(run.calls[0].output)
    assert b"private" not in serialized
    assert b"ToolCallError" not in serialized


def test_wrong_or_missing_hit_binding_aborts_publication(evidence):
    wrong = evidence["binding"].model_copy(update={"index_integrity_sha256": "9" * 64})

    with pytest.raises(RetrievalRunError, match="artifact_binding mismatch"):
        _create(evidence, lambda request: [_hit(wrong, request.query_id)])
    assert not evidence["output"].exists()

    with pytest.raises(RetrievalRunError, match="missing artifact_binding"):
        _create(evidence, lambda request: [{"rank": 1}])
    assert not evidence["output"].exists()


def test_run_destination_is_create_only(evidence):
    _create(evidence, lambda request: [])
    before = (evidence["output"] / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        _create(evidence, lambda request: [])

    assert (evidence["output"] / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-execution", "complete authoritative query order"),
        ("duplicate-execution", "complete authoritative query order"),
        ("extra-execution", "complete authoritative query order"),
        ("missing-call", "call references mismatch"),
        ("duplicate-call", "duplicate retrieval call_id"),
        ("orphan-call", "orphan retrieval call"),
        ("bad-ordinal", "ordinals are not contiguous"),
    ],
)
def test_loader_rejects_incomplete_or_incoherent_run_graph(evidence, mutation, message):
    _create(evidence, lambda request: [])
    executions = _read_jsonl(evidence["output"] / "query-executions.jsonl")
    calls = _read_jsonl(evidence["output"] / "tool-calls.jsonl")

    if mutation == "missing-execution":
        executions.pop()
    elif mutation == "duplicate-execution":
        executions.append(dict(executions[0]))
    elif mutation == "extra-execution":
        extra = dict(executions[0])
        extra["query_id"] = "q-extra"
        extra["query_ordinal"] = 2
        executions.append(extra)
    elif mutation == "missing-call":
        calls.pop()
    elif mutation == "duplicate-call":
        calls.append(dict(calls[0]))
    elif mutation == "orphan-call":
        calls[0]["query_id"] = "q-orphan"
    elif mutation == "bad-ordinal":
        calls[0]["ordinal"] = 1
    _rewrite_bundle(evidence["output"], executions=executions, calls=calls)

    with pytest.raises(RetrievalRunError, match=message):
        _load(evidence)


def test_loader_rejects_non_authoritative_path_like_arguments(evidence):
    _create(evidence, lambda request: [])
    calls = _read_jsonl(evidence["output"] / "tool-calls.jsonl")
    executions = _read_jsonl(evidence["output"] / "query-executions.jsonl")
    spec = json.loads((evidence["output"] / "spec.json").read_text(encoding="utf-8"))
    calls[0]["arguments"]["image_path"] = "gallery/attacker.png"
    calls[0]["arguments_sha256"] = sha256_bytes(
        canonical_json_bytes(calls[0]["arguments"])
    )
    calls[0]["call_id"] = run_module._call_id(
        spec["spec_sha256"],
        calls[0]["query_id"],
        calls[0]["ordinal"],
        calls[0]["tool_name"],
        calls[0]["arguments_sha256"],
    )
    executions[0]["call_ids"] = [calls[0]["call_id"]]
    _rewrite_bundle(evidence["output"], executions=executions, calls=calls)

    with pytest.raises(RetrievalRunError, match="not authoritative"):
        _load(evidence)


def test_loader_rechecks_argument_output_and_artifact_binding_hashes(evidence):
    _create(evidence, lambda request: [_hit(evidence["binding"], request.query_id)])
    calls = _read_jsonl(evidence["output"] / "tool-calls.jsonl")
    calls[0]["output"][0]["score"] = 0.75
    # Deliberately retain output_sha256 while re-signing the containing call.
    _rewrite_bundle(evidence["output"], calls=calls)
    with pytest.raises(RetrievalRunError, match="output hash mismatch"):
        _load(evidence)

    other_output = evidence["output"].with_name("binding-run")
    _create(evidence, lambda request: [], output=other_output)
    calls = _read_jsonl(other_output / "tool-calls.jsonl")
    calls[0]["artifact_binding"]["index_integrity_sha256"] = "9" * 64
    _rewrite_bundle(other_output, calls=calls)
    with pytest.raises(RetrievalRunError, match="call artifact binding mismatch"):
        _load(evidence, output=other_output)


def test_loader_rejects_duplicate_json_keys_and_noncanonical_json(evidence):
    _create(evidence, lambda request: [])
    spec_path = evidence["output"] / "spec.json"
    original = spec_path.read_bytes()
    duplicate = b'{"schema_version":1,' + original[1:]
    _rewrite_bundle(evidence["output"], raw_spec=duplicate)
    with pytest.raises(RetrievalRunError, match="duplicate key"):
        _load(evidence)

    other_output = evidence["output"].with_name("other-run")
    _create(evidence, lambda request: [], output=other_output)
    spec = json.loads((other_output / "spec.json").read_text(encoding="utf-8"))
    noncanonical = (json.dumps(spec, ensure_ascii=False, indent=2) + "\n").encode()
    _rewrite_bundle(other_output, raw_spec=noncanonical)
    with pytest.raises(RetrievalRunError, match="canonical JSON"):
        _load(evidence, output=other_output)


def test_loader_rejects_nonfinite_json_numbers(evidence):
    _create(evidence, lambda request: [])
    calls = _read_jsonl(evidence["output"] / "tool-calls.jsonl")
    calls[0]["output"] = [float("nan")]
    calls[0]["output_sha256"] = sha256_bytes(canonical_json_bytes(calls[0]["output"]))
    _rewrite_bundle(evidence["output"], calls=calls)

    with pytest.raises(RetrievalRunError, match="non-finite"):
        _load(evidence)


def test_loader_rejects_symlinked_bundle_artifact(evidence):
    _create(evidence, lambda request: [])
    calls_path = evidence["output"] / "tool-calls.jsonl"
    target = evidence["output"] / "tool-calls-copy.jsonl"
    shutil.copyfile(calls_path, target)
    calls_path.unlink()
    try:
        os.symlink(target, calls_path)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RetrievalRunError, match="symbolic link"):
        _load(evidence)


def test_authoritative_artifact_changes_during_execution_abort_publication(evidence):
    def executor(request: RetrievalRequest):
        if request.query_ordinal == 0:
            evidence["query_path"].write_bytes(
                evidence["query_path"].read_bytes() + b"\n"
            )
        return []

    with pytest.raises(ValueError, match="quer|authoritative"):
        _create(evidence, executor)
    assert not evidence["output"].exists()


def test_loaded_run_fails_after_query_asset_bytes_change(evidence):
    _create(evidence, lambda request: [])
    first_asset = evidence["queries"][0]
    path = evidence["asset_root"].joinpath(*first_asset.image_path.split("/"))
    _write_image(path, pattern=2)

    with pytest.raises(ValueError, match="SHA|sha|catalog"):
        _load(evidence)


def test_evaluator_gate_accepts_only_loader_verified_objects(evidence):
    provisional = _create(evidence, lambda request: [])

    with pytest.raises(TypeError, match="VerifiedRetrievalRun"):
        require_verified_retrieval_run(provisional)
    with pytest.raises(TypeError, match="VerifiedRetrievalRun"):
        require_verified_retrieval_run([])
    with pytest.raises(TypeError, match="VerifiedRetrievalRun"):
        require_verified_retrieval_run(evidence["output"])
