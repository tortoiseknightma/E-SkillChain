"""Public, read-only loader for the completed Static opt800 GCS-v2 corpus.

The matrix analyzer historically owned this verification inline.  S1 Feedback
needs the same trusted rows (including the Assistant response, receipt, and
scorer sidecar), so this module exposes one reusable fail-closed boundary.
It never writes artifacts and never calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

from skillchain.evaluation.assistant_runs import (
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    AssistantRequestSnapshot,
)
from skillchain.evaluation.packets import AssistantResult
from skillchain.evaluation.portfolio_core_inputs import VerifiedPortfolioCoreInputs
from skillchain.evaluation.portfolio_gcs import (
    GCS_CAPABILITY_ORDER,
    GCS_V2_POLICY_SHA256,
    GCSPopulationV2,
    GCSQueryScoreV2,
    build_gcs_population_v2,
    portfolio_gcs_oracles_v2,
    score_portfolio_gcs_v2,
)
from skillchain.evaluation.portfolio_gcs_evidence import (
    PublicScorerEvidenceV2,
    classify_style_query_v2,
    require_public_scorer_evidence_v2,
)
from skillchain.evaluation.portfolio_launch import (
    LoadedPortfolioLaunchPackage,
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (
    VerifiedPortfolioStaticOptRuntime,
    load_verified_portfolio_static_opt_runtime_evidence,
    validate_portfolio_static_opt_evidence_control,
)
from skillchain.schemas import Query
from skillchain.synthesis.portfolio_core_r3_overlay import (
    R3RepairManifest,
    R3RepairPlan,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes
from skillchain.task_spec import (
    MVP_TASK_SPEC_V1_PATH,
    TaskSpecification,
    load_mvp_task_specification_v1,
)
from skillchain.tools.serialization import (
    parse_canonical_json,
    read_stable_regular_file,
)


STATIC_GCS_CORPUS_POLICY_VERSION = "portfolio-static-opt800-gcs-corpus-v1"
STATIC_GCS_QUERY_COUNT = 800
STATIC_GCS_SHARD_COUNT = 32
STATIC_GCS_BATCH_SIZE = 25
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_CORPUS_MARKER = object()


class StaticGCSCorpusError(ValueError):
    """A launch, checkpoint, sidecar, or frozen source failed verification."""


@dataclass(frozen=True)
class StaticGCSCorpusStrata:
    source_dataset: str
    repair_status: str
    boundary_status: str
    style_submode: str | None


@dataclass(frozen=True)
class VerifiedStaticGCSRow:
    query_ordinal: int
    batch_id: str
    shard_id: str
    instance_id: str
    query: Query
    request: AssistantRequestSnapshot
    response: AssistantBackendResponse
    result: AssistantResult
    receipt: AssistantExecutionReceipt
    sidecar: PublicScorerEvidenceV2
    score: GCSQueryScoreV2
    atomic_component_id: str
    leakage_group_id: str
    checkpoint_path: Path
    checkpoint_file_sha256: str
    checkpoint_row_sha256: str
    image_sha256: str
    strata: StaticGCSCorpusStrata


@dataclass(frozen=True)
class VerifiedStaticGCSCorpus:
    """Deeply verified in-memory handle for the exact completed opt800 run."""

    execution_root: Path
    artifact_repository_root: Path
    control_file_sha256: str
    control_sha256: str
    launch_root: Path
    runtime_root: Path
    launch: LoadedPortfolioLaunchPackage
    runtime: VerifiedPortfolioStaticOptRuntime
    core_inputs: VerifiedPortfolioCoreInputs
    task_spec: TaskSpecification
    population: GCSPopulationV2
    rows: tuple[VerifiedStaticGCSRow, ...]
    checkpoint_set_sha256: str
    sidecar_set_sha256: str
    corpus_sha256: str
    _marker: object

    def row_by_query_id(self) -> dict[str, VerifiedStaticGCSRow]:
        return {row.query.query_id: row for row in self.rows}


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StaticGCSCorpusError(f"{label} must be a lowercase SHA-256")
    return value


def _hash_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _resolve_bound_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StaticGCSCorpusError(f"{label} path is invalid")
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.absolute().resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise StaticGCSCorpusError(f"{label} escapes the artifact repository root")
    return resolved


def _load_execution_control(
    execution_root: Path,
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, object], str]:
    expected_file_sha256 = _require_sha256(
        expected_file_sha256, "execution-control file SHA-256"
    )
    path = execution_root / "execution-control.json"
    content = read_stable_regular_file(
        path,
        label="Static GCS execution control",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise StaticGCSCorpusError("execution-control file SHA-256 mismatch")
    try:
        raw = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StaticGCSCorpusError("execution control is not JSON") from error
    if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
        raise StaticGCSCorpusError("execution control is not one canonical object")
    supplied = _require_sha256(raw.get("control_sha256"), "control self hash")
    unsigned = dict(raw)
    unsigned.pop("control_sha256", None)
    if supplied != _hash_json(unsigned):
        raise StaticGCSCorpusError("execution control self hash mismatch")
    blinding_sha256 = raw.get("blinding_key_sha256")
    if blinding_sha256 is not None:
        key = read_stable_regular_file(
            execution_root / "blinding-key.bin",
            label="Static GCS blinding key",
            max_bytes=64,
        )
        if len(key) != 32 or sha256_bytes(key) != blinding_sha256:
            raise StaticGCSCorpusError("execution blinding key mismatch")
    return raw, supplied


def _load_task_specification(
    runtime: VerifiedPortfolioStaticOptRuntime,
) -> TaskSpecification:
    lock = runtime.runtime_lock
    semantic_input = runtime.semantic_authoring_input
    raw = parse_canonical_json(
        semantic_input.task_specification.content,
        label="Static runtime embedded TaskSpec",
    )
    if not isinstance(raw, dict):
        raise StaticGCSCorpusError("embedded TaskSpec is not an object")
    task_spec = TaskSpecification.model_validate(raw, strict=True)
    active = load_mvp_task_specification_v1()
    active_file = read_stable_regular_file(
        MVP_TASK_SPEC_V1_PATH,
        label="active GCS-v2 TaskSpec",
        max_bytes=4 * 1024 * 1024,
    )
    if (
        task_spec != active
        or semantic_input.task_specification.version != task_spec.task_spec_version
        or semantic_input.task_specification.identity_sha256
        != task_spec.task_spec_sha256
        or semantic_input.task_specification.content
        != canonical_json_bytes(task_spec.model_dump(mode="json"))
        or lock.get("task_spec_version") != task_spec.task_spec_version
        or lock.get("task_spec_sha256") != task_spec.task_spec_sha256
        or lock.get("task_spec_file_sha256") != sha256_bytes(active_file)
        or lock.get("semantic_authoring_input_file_sha256")
        != runtime.semantic_authoring_input_file_sha256
        or lock.get("semantic_authoring_input_sha256") != semantic_input.input_sha256
        or runtime.refresh_receipt.get("new_semantic_authoring_input_file_sha256")
        != runtime.semantic_authoring_input_file_sha256
        or runtime.refresh_receipt.get("new_semantic_authoring_input_sha256")
        != semantic_input.input_sha256
        or lock.get("static_contract_refresh_receipt_file_sha256")
        != runtime.refresh_receipt_file_sha256
        or lock.get("static_contract_refresh_receipt_sha256")
        != runtime.refresh_receipt.get("receipt_sha256")
        or lock.get("core_runtime_sources_receipt_file_sha256")
        != runtime.core_source_receipt_file_sha256
        or lock.get("core_runtime_sources_receipt_sha256")
        != runtime.core_source_receipt.get("receipt_sha256")
        or lock.get("runtime_data_sha256")
        != runtime.core_source_receipt.get("runtime_data_sha256")
    ):
        raise StaticGCSCorpusError("Static runtime TaskSpec binding drifted")
    return task_spec


def _load_repair_query_ids(
    core_inputs: VerifiedPortfolioCoreInputs,
) -> frozenset[str]:
    manifest_path = core_inputs.files.materialization_manifest_path.absolute()
    manifest_content = read_stable_regular_file(
        manifest_path,
        label="Core r3 materialization manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if sha256_bytes(manifest_content) != (
        core_inputs.files.expected_materialization_manifest_file_sha256
    ):
        raise StaticGCSCorpusError("Core r3 materialization manifest drifted")
    manifest = R3RepairManifest.model_validate_json(manifest_content, strict=True)
    if manifest.canonical_bytes() != manifest_content:
        raise StaticGCSCorpusError("Core r3 materialization manifest is not canonical")
    matches = tuple(
        item for item in manifest.files if item.relative_path == "repair-plan.json"
    )
    if len(matches) != 1:
        raise StaticGCSCorpusError("Core r3 manifest does not bind one repair plan")
    descriptor = matches[0]
    repair_content = read_stable_regular_file(
        manifest_path.parent / descriptor.relative_path,
        label="Core r3 repair plan",
        max_bytes=2 * 1024 * 1024,
    )
    if (
        len(repair_content) != descriptor.bytes
        or sha256_bytes(repair_content) != descriptor.sha256
    ):
        raise StaticGCSCorpusError("Core r3 repair plan differs from its manifest")
    repair_plan = R3RepairPlan.model_validate_json(repair_content, strict=True)
    if (
        repair_plan.canonical_bytes() != repair_content
        or repair_plan.repair_plan_sha256 != manifest.repair_plan_sha256
    ):
        raise StaticGCSCorpusError("Core r3 repair plan identity drifted")
    query_ids = frozenset(
        query_id for batch in repair_plan.batches for query_id in batch.query_ids
    )
    if len(query_ids) != manifest.replacement_query_count:
        raise StaticGCSCorpusError("Core r3 repair population is incomplete")
    return query_ids


def _request(value: object) -> AssistantRequestSnapshot:
    return AssistantRequestSnapshot.model_validate_json(
        canonical_json_bytes(value), strict=True
    )


def _checkpoint_models(
    raw: Mapping[str, object],
) -> tuple[
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    PublicScorerEvidenceV2,
]:
    response = AssistantBackendResponse.model_validate_json(
        canonical_json_bytes(raw.get("response")), strict=True
    )
    receipt = AssistantExecutionReceipt.model_validate_json(
        canonical_json_bytes(raw.get("receipt")), strict=True
    )
    sidecar = require_public_scorer_evidence_v2(raw.get("public_scorer_evidence"))
    response_sha256 = _hash_json(response.model_dump(mode="json"))
    if receipt.response_sha256 != response_sha256:
        raise StaticGCSCorpusError("checkpoint receipt/response mismatch")
    return response, receipt, sidecar


def _assistant_result(
    launch: LoadedPortfolioLaunchPackage,
    request: AssistantRequestSnapshot,
    response: AssistantBackendResponse,
) -> AssistantResult:
    routed = response.selected_capability is not None
    public_error_code = (
        "runtime_error"
        if response.error_code is not None
        and response.error_code.startswith("provider_pre_response_")
        else response.error_code
    )
    return AssistantResult(
        run_id=launch.plan.matrix_run_id,
        query_id=request.query.query_id,
        config=request.config,
        response_text=response.response_text,
        visible_cards=response.visible_cards,
        visible_tool_evidence=response.visible_tool_evidence,
        tool_trace=response.tool_trace,
        selected_capability=response.selected_capability,
        skill_slug=response.skill_slug,
        bank_sha256=request.treatment.bank_sha256 if routed else None,
        route_trace_sha256=response.route_trace_sha256,
        query_artifact_sha256=launch.plan.query_artifact_sha256,
        split_manifest_sha256=launch.plan.portfolio_plan_sha256,
        registry_sha256=response.registry_sha256,
        registry_runtime_sha256=response.registry_runtime_sha256,
        backbone_provider=response.backbone_provider,
        backbone_model=response.backbone_model,
        backbone_request_id=response.backbone_request_id,
        usage=response.usage,
        latency_ms=response.latency_ms,
        error_code=public_error_code,
    )


def _corpus_identity_payload(
    *,
    control_sha256: str,
    launch_sha256: str,
    runtime_sha256: str,
    population_sha256: str,
    checkpoint_set_sha256: str,
    sidecar_set_sha256: str,
    parent_bank_sha256: str,
) -> dict[str, object]:
    return {
        "policy_version": STATIC_GCS_CORPUS_POLICY_VERSION,
        "control_sha256": control_sha256,
        "launch_plan_sha256": launch_sha256,
        "runtime_lock_sha256": runtime_sha256,
        "population_mapping_sha256": population_sha256,
        "checkpoint_set_sha256": checkpoint_set_sha256,
        "sidecar_set_sha256": sidecar_set_sha256,
        "parent_static_bank_sha256": parent_bank_sha256,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
    }


def load_verified_static_gcs_corpus(
    execution_root: str | Path,
    *,
    expected_control_file_sha256: str,
    artifact_repository_root: str | Path,
) -> VerifiedStaticGCSCorpus:
    """Load and rescore all 800 schema-v2 checkpoints without provider calls."""

    repository_root = Path(artifact_repository_root).absolute().resolve(strict=True)
    execution = Path(execution_root)
    execution = execution if execution.is_absolute() else repository_root / execution
    execution = execution.absolute().resolve(strict=True)
    if execution != repository_root and repository_root not in execution.parents:
        raise StaticGCSCorpusError("execution root escapes artifact repository root")
    control, control_sha256 = _load_execution_control(
        execution,
        expected_file_sha256=expected_control_file_sha256,
    )
    launch_root = _resolve_bound_path(
        repository_root, control.get("launch_root"), label="launch root"
    )
    runtime_root = _resolve_bound_path(
        repository_root, control.get("runtime_root"), label="runtime root"
    )
    launch = load_portfolio_launch_package(
        launch_root,
        expected_plan_file_sha256=_require_sha256(
            control.get("launch_plan_file_sha256"), "launch-plan file SHA-256"
        ),
    )
    runtime = load_verified_portfolio_static_opt_runtime_evidence(
        runtime_root,
        expected_runtime_lock_file_sha256=_require_sha256(
            control.get("runtime_lock_file_sha256"), "runtime-lock file SHA-256"
        ),
    )
    try:
        validate_portfolio_static_opt_evidence_control(control, runtime)
    except (TypeError, ValueError) as error:
        raise StaticGCSCorpusError(str(error)) from error
    if (
        control.get("launch_plan_sha256") != launch.plan.launch_plan_sha256
        or control.get("matrix_run_id") != launch.plan.matrix_run_id
        or control.get("instance_count") != STATIC_GCS_QUERY_COUNT
        or control.get("shard_count") != STATIC_GCS_SHARD_COUNT
        or launch.plan.query_count != STATIC_GCS_QUERY_COUNT
        or launch.plan.instance_count != STATIC_GCS_QUERY_COUNT
        or launch.plan.shard_count != STATIC_GCS_SHARD_COUNT
        or launch.plan.config_order != ("llm_static",)
    ):
        raise StaticGCSCorpusError("execution control differs from Static launch")

    core_inputs = reconstruct_verified_portfolio_core_inputs(launch.plan)
    queries = tuple(query for query in core_inputs.queries if query.split == "opt_pool")
    launch_query_ids = {
        query_id for shard in launch.plan.shards for query_id in shard.query_ids
    }
    if (
        len(queries) != STATIC_GCS_QUERY_COUNT
        or {query.query_id for query in queries} != launch_query_ids
    ):
        raise StaticGCSCorpusError("verified Core opt_pool differs from launch")
    task_spec = _load_task_specification(runtime)
    population = build_gcs_population_v2(queries)
    binding_by_query = {item.query_id: item for item in population.bindings}
    query_by_id = {query.query_id: query for query in queries}
    member_by_query = {member.query_id: member for member in launch.instances}
    shard_by_id = {shard.shard_id: shard for shard in launch.plan.shards}
    if (
        len(member_by_query) != STATIC_GCS_QUERY_COUNT
        or len(shard_by_id) != STATIC_GCS_SHARD_COUNT
        or any(
            shard.config != "llm_static"
            or shard.query_count != STATIC_GCS_BATCH_SIZE
            or len(shard.query_ids) != STATIC_GCS_BATCH_SIZE
            for shard in shard_by_id.values()
        )
    ):
        raise StaticGCSCorpusError("Static launch geometry is invalid")
    catalog = core_inputs.runtime_asset_catalog()
    repair_query_ids = _load_repair_query_ids(core_inputs)
    oracles = portfolio_gcs_oracles_v2()
    rows: list[VerifiedStaticGCSRow] = []
    checkpoint_inventory: list[dict[str, object]] = []
    sidecar_hashes: list[str] = []

    for query_id, member in sorted(
        member_by_query.items(), key=lambda item: item[1].query_ordinal
    ):
        query = query_by_id.get(query_id)
        shard = shard_by_id.get(member.shard_id)
        if query is None or shard is None or query_id not in shard.query_ids:
            raise StaticGCSCorpusError("launch member query/shard identity drifted")
        checkpoint_path = (execution / member.assistant_output_relpath).resolve(
            strict=True
        )
        if execution not in checkpoint_path.parents:
            raise StaticGCSCorpusError("Assistant checkpoint escapes execution root")
        content = read_stable_regular_file(
            checkpoint_path,
            label=f"Static GCS checkpoint {query_id}",
            max_bytes=16 * 1024 * 1024,
        )
        try:
            raw = json.loads(content)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise StaticGCSCorpusError("Assistant checkpoint is not JSON") from error
        if not isinstance(raw, dict) or canonical_json_bytes(raw) != content:
            raise StaticGCSCorpusError("Assistant checkpoint is not canonical JSON")
        supplied_row_sha256 = _require_sha256(
            raw.get("row_sha256"), "Assistant checkpoint row SHA-256"
        )
        unsigned = dict(raw)
        unsigned.pop("row_sha256", None)
        if supplied_row_sha256 != _hash_json(unsigned):
            raise StaticGCSCorpusError("Assistant checkpoint self hash mismatch")
        response, receipt, sidecar = _checkpoint_models(raw)
        request = _request(raw.get("request"))
        result = _assistant_result(launch, request, response)
        if (
            raw.get("kind") != "portfolio-assistant-checkpoint"
            or raw.get("schema_version") != 2
            or raw.get("instance_sha256") != member.instance_sha256
            or raw.get("query_ordinal") != member.query_ordinal
            or request.matrix_run_id != launch.plan.matrix_run_id
            or request.config != "llm_static"
            or request.query.query_id != query_id
            or receipt.request_sha256 != request.request_sha256
            or sidecar.matrix_run_id != launch.plan.matrix_run_id
            or sidecar.instance_id != member.instance_sha256
            or sidecar.request_sha256 != request.request_sha256
            or sidecar.query_id != query_id
            or sidecar.config != "llm_static"
            or sidecar.query_artifact_sha256 != launch.plan.query_artifact_sha256
        ):
            raise StaticGCSCorpusError("checkpoint identity differs from launch")
        if (execution / member.final_output_relpath).exists():
            raise StaticGCSCorpusError(
                "Static opt corpus unexpectedly has Final output"
            )
        score = score_portfolio_gcs_v2(
            query,
            result,
            receipt,
            sidecar,
            task_spec,
            oracles,
            population=population,
        )
        asset = catalog.resolve_asset_id(query.asset_id).asset
        style_submode = (
            classify_style_query_v2(query.text).style_submode
            if query.canonical_capability == "product.style_recommendation"
            else None
        )
        checkpoint_file_sha256 = sha256_bytes(content)
        rows.append(
            VerifiedStaticGCSRow(
                query_ordinal=member.query_ordinal,
                batch_id=shard.accepted_batch_id,
                shard_id=shard.shard_id,
                instance_id=member.instance_sha256,
                query=query,
                request=request,
                response=response,
                result=result,
                receipt=receipt,
                sidecar=sidecar,
                score=score,
                atomic_component_id=binding_by_query[query_id].component_id,
                leakage_group_id=query.leakage_group_id,
                checkpoint_path=checkpoint_path,
                checkpoint_file_sha256=checkpoint_file_sha256,
                checkpoint_row_sha256=supplied_row_sha256,
                image_sha256=asset.sha256,
                strata=StaticGCSCorpusStrata(
                    source_dataset=asset.source_dataset,
                    repair_status=(
                        "r3_language_repaired"
                        if query_id in repair_query_ids
                        else "r3_carry_forward"
                    ),
                    boundary_status=(
                        "boundary" if query.is_boundary else "non_boundary"
                    ),
                    style_submode=style_submode,
                ),
            )
        )
        checkpoint_inventory.append(
            {
                "query_id": query_id,
                "assistant_output_relpath": member.assistant_output_relpath,
                "file_sha256": checkpoint_file_sha256,
                "row_sha256": supplied_row_sha256,
            }
        )
        sidecar_hashes.append(sidecar.evidence_sha256)
        if (
            read_stable_regular_file(
                checkpoint_path,
                label=f"Static GCS checkpoint recheck {query_id}",
                max_bytes=16 * 1024 * 1024,
            )
            != content
        ):
            raise StaticGCSCorpusError("Assistant checkpoint changed during load")

    ordered_rows = tuple(rows)
    if (
        len(ordered_rows) != STATIC_GCS_QUERY_COUNT
        or len({row.query.query_id for row in ordered_rows}) != STATIC_GCS_QUERY_COUNT
        or {row.score.evaluated_capability for row in ordered_rows}
        != set(GCS_CAPABILITY_ORDER)
    ):
        raise StaticGCSCorpusError("Static GCS corpus is incomplete")
    checkpoint_set_sha256 = _hash_json(checkpoint_inventory)
    sidecar_set_sha256 = _hash_json(sidecar_hashes)
    corpus_sha256 = _hash_json(
        _corpus_identity_payload(
            control_sha256=control_sha256,
            launch_sha256=launch.plan.launch_plan_sha256,
            runtime_sha256=str(runtime.runtime_lock["runtime_lock_sha256"]),
            population_sha256=population.population_mapping_sha256,
            checkpoint_set_sha256=checkpoint_set_sha256,
            sidecar_set_sha256=sidecar_set_sha256,
            parent_bank_sha256=runtime.bank.bank_sha256,
        )
    )
    return VerifiedStaticGCSCorpus(
        execution_root=execution,
        artifact_repository_root=repository_root,
        control_file_sha256=_require_sha256(
            expected_control_file_sha256, "execution-control file SHA-256"
        ),
        control_sha256=control_sha256,
        launch_root=launch_root,
        runtime_root=runtime_root,
        launch=launch,
        runtime=runtime,
        core_inputs=core_inputs,
        task_spec=task_spec,
        population=population,
        rows=ordered_rows,
        checkpoint_set_sha256=checkpoint_set_sha256,
        sidecar_set_sha256=sidecar_set_sha256,
        corpus_sha256=corpus_sha256,
        _marker=_VERIFIED_CORPUS_MARKER,
    )


def require_verified_static_gcs_corpus(
    value: VerifiedStaticGCSCorpus,
) -> VerifiedStaticGCSCorpus:
    """Reject forged handles and freshly reverify every source artifact."""

    if (
        type(value) is not VerifiedStaticGCSCorpus
        or value._marker is not _VERIFIED_CORPUS_MARKER
    ):
        raise TypeError("S1 Feedback requires a verified Static GCS corpus")
    rebuilt = load_verified_static_gcs_corpus(
        value.execution_root,
        expected_control_file_sha256=value.control_file_sha256,
        artifact_repository_root=value.artifact_repository_root,
    )
    if rebuilt.corpus_sha256 != value.corpus_sha256:
        raise StaticGCSCorpusError("verified Static GCS corpus changed on disk")
    return rebuilt


def accept_loaded_verified_static_gcs_corpus(
    value: VerifiedStaticGCSCorpus,
) -> VerifiedStaticGCSCorpus:
    """Accept a handle returned by this process's deep loader without rescanning."""

    if (
        type(value) is not VerifiedStaticGCSCorpus
        or value._marker is not _VERIFIED_CORPUS_MARKER
    ):
        raise TypeError("S1 Feedback requires a verified Static GCS corpus")
    return value


__all__ = [
    "STATIC_GCS_CORPUS_POLICY_VERSION",
    "StaticGCSCorpusError",
    "StaticGCSCorpusStrata",
    "VerifiedStaticGCSCorpus",
    "VerifiedStaticGCSRow",
    "accept_loaded_verified_static_gcs_corpus",
    "load_verified_static_gcs_corpus",
    "require_verified_static_gcs_corpus",
]
