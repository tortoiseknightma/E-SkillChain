"""Read-only, fail-closed analysis of a completed Portfolio matrix.

The analyzer performs no model calls and never writes into the execution,
launch, or runtime roots.  Legacy five-config selections retain the deep
cross-config audit/replay path.  A Static opt rollout instead verifies every
schema-v2 Assistant checkpoint and embedded scorer sidecar, then scores the
fixed population directly under the frozen six-capability GCS v2 policy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
from itertools import combinations
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import tempfile
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.audit_portfolio_batch import (  # noqa: E402
    _load_external_frozen_source,
    _load_launch_profile_inputs,
    audit_portfolio_batch,
)
from scripts.run_portfolio_shard import (  # noqa: E402
    _assistant_result,
    _assistant_row,
    _load_control,
)
from skillchain.evaluation.assistant_runs import (  # noqa: E402
    AssistantRequestSnapshot,
)
from skillchain.evaluation.final_runtime import (  # noqa: E402
    load_final_judge_evaluation_result,
)
from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    load_query_attempt_receipts,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    MAIN_CONFIG_ORDER,
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_gcs import (  # noqa: E402
    GCS_CAPABILITY_ORDER,
    GCS_COMPONENTS,
    GCS_V2_POLICY_SHA256,
    GCS_V2_POLICY_VERSION,
    GCSConfigSummaryV2,
    GCSQueryScoreV2,
    build_gcs_population_v2,
    portfolio_gcs_oracles_v2,
    score_portfolio_gcs_v2,
    summarize_gcs_v2,
)
from skillchain.evaluation.portfolio_gcs_evidence import (  # noqa: E402
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    classify_style_query_v2,
)
from skillchain.evaluation.portfolio_treatment_io import (  # noqa: E402
    load_verified_portfolio_treatment_runtime,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    load_verified_portfolio_static_opt_runtime,
)
from skillchain.evaluation.portfolio_treatments import (  # noqa: E402
    calculate_portfolio_skill_adherence,
)
from skillchain.synthesis.store import atomic_create_file  # noqa: E402
from skillchain.synthesis.portfolio_core_r3_overlay import (  # noqa: E402
    R3RepairManifest,
    R3RepairPlan,
)
from skillchain.task_spec import (  # noqa: E402
    MVP_TASK_SPEC_V1_PATH,
    TaskSpecification,
    load_mvp_task_specification_v1,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


CAPABILITIES = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_ITERATIONS = 2_000
_STATIC_OPT_EXECUTION_SCOPE = "static_opt_rollout"
_GCS_V2_CHECKPOINT_SCHEMA_VERSION = 2
_STATIC_GCS_STRATUM_ORDER = (
    "capability",
    "source",
    "repair",
    "boundary",
    "style_submode",
)


@dataclass(frozen=True)
class MatrixRow:
    query_ordinal: int
    query_id: str
    batch_id: str
    partition: str
    leakage_group_id: str
    canonical_capability: str
    config: str
    bank_sha256: str | None
    j_project: float
    selected_capability: str | None
    route_correct: bool
    route_acceptable: bool
    structural_adherence: float | None
    gate_aligned_adherence: float | None
    assistant_hard_error: bool
    evaluator_anomaly: bool
    assistant_status: str
    assistant_error_code: str | None
    assistant_receipt_outcome: str
    assistant_model_call_count: int
    assistant_finish_reasons: tuple[str, ...]
    tool_call_count: int
    tool_error_count: int
    tool_error_codes: tuple[str, ...]
    tool_statuses: tuple[str, ...]
    tool_recovery_status: str
    card_requirement: str
    visible_card_count: int
    card_policy_compliant: bool
    card_guard_adjusted: bool | None
    judge_status: str
    judge_error_code: str | None
    judge_provider: str | None
    judge_model: str | None
    judge_finish_reason: str | None
    attempt_receipt_count: int
    attempt_failure_stages: tuple[str, ...]
    attempt_failure_subtypes: tuple[str, ...]
    attempt_recovery_status: str
    attempt_circuit_open_count: int
    attempt_captured_provider_response_count: int
    attempt_captured_input_tokens: int
    attempt_captured_output_tokens: int


@dataclass(frozen=True)
class _PartitionLayout:
    profile: str
    partition_order: tuple[str, ...]
    partition_by_query: Mapping[str, str]
    expected_query_counts: Mapping[str, int]
    aggregate_name: str
    primary_effect_scope: str
    use_policy: Mapping[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument(
        "--batch-id",
        help="Analyze one complete cross-config batch instead of the launch matrix.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _json_projection(value: object) -> object:
    """Project immutable Python containers into canonical JSON containers."""

    if isinstance(value, Mapping):
        return {str(key): _json_projection(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_projection(item) for item in value]
    return value


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _require_external_output(
    output_dir: Path,
    *,
    execution_root: Path,
    launch_root: Path,
    runtime_root: Path,
) -> None:
    absolute = output_dir.absolute().resolve(strict=False)
    for label, root in (
        ("execution", execution_root),
        ("launch", launch_root),
        ("runtime", runtime_root),
    ):
        if _is_within(absolute, root.absolute().resolve(strict=True)):
            raise ValueError(f"output directory must be outside the {label} root")
    if absolute.exists():
        raise ValueError("output directory already exists")


def _batch_ids(plan: object) -> tuple[str, ...]:
    values: list[str] = []
    for shard in getattr(plan, "shards"):
        batch_id = str(getattr(shard, "accepted_batch_id"))
        if batch_id not in values:
            values.append(batch_id)
    return tuple(values)


def _select_batch_ids(plan: object, requested: str | None) -> tuple[str, ...]:
    available = _batch_ids(plan)
    if not available:
        raise ValueError("launch contains no analyzable batches")
    if requested is not None:
        if requested not in available:
            raise ValueError(f"unknown batch ID: {requested}")
        selected = (requested,)
    else:
        selected = available
    expected_shards = 5 * len(selected)
    selected_shards = tuple(
        item
        for item in getattr(plan, "shards")
        if str(getattr(item, "accepted_batch_id")) in selected
    )
    if len(selected_shards) != expected_shards:
        raise ValueError("selected batches do not contain exactly five shards each")
    for batch_id in selected:
        batch_shards = tuple(
            item
            for item in selected_shards
            if str(getattr(item, "accepted_batch_id")) == batch_id
        )
        configs = tuple(str(getattr(item, "config")) for item in batch_shards)
        if sorted(configs) != sorted(MAIN_CONFIG_ORDER):
            raise ValueError(f"batch {batch_id} does not contain the five configs")
        query_sets = {
            tuple(str(value) for value in getattr(item, "query_ids"))
            for item in batch_shards
        }
        if len(query_sets) != 1:
            raise ValueError(f"batch {batch_id} does not share one paired query set")
    if requested is None:
        reference_queries = {
            batch_id: next(
                tuple(str(value) for value in getattr(item, "query_ids"))
                for item in selected_shards
                if str(getattr(item, "accepted_batch_id")) == batch_id
            )
            for batch_id in selected
        }
        flattened = tuple(
            query_id
            for batch_id in selected
            for query_id in reference_queries[batch_id]
        )
        if (
            len(set(flattened)) != len(flattened)
            or len(flattened) != int(getattr(plan, "query_count"))
            or len(selected_shards) != int(getattr(plan, "shard_count"))
            or sum(int(getattr(item, "query_count")) for item in selected_shards)
            != int(getattr(plan, "instance_count"))
        ):
            raise ValueError("complete-matrix selection differs from launch geometry")
    return selected


def _require_static_gcs_v2_contract(
    control: Mapping[str, object], plan: object
) -> None:
    """Recognize the one Assistant-only execution contract accepted by v2."""

    declarations = (
        control.get("execution_scope"),
        control.get("assistant_checkpoint_schema_version"),
        control.get("gcs_policy_version"),
        control.get("gcs_scorer_evidence_policy_version"),
        getattr(plan, "execution_mode", None),
        tuple(getattr(plan, "config_order", ())),
        tuple(getattr(plan, "selected_splits", ())),
    )
    expected = (
        _STATIC_OPT_EXECUTION_SCOPE,
        _GCS_V2_CHECKPOINT_SCHEMA_VERSION,
        GCS_V2_POLICY_VERSION,
        GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
        _STATIC_OPT_EXECUTION_SCOPE,
        ("llm_static",),
        ("opt_pool",),
    )
    if declarations != expected:
        raise ValueError("Static opt analysis lacks the exact GCS v2 contract")
    if (
        getattr(plan, "kind", None) != "portfolio-core-static-opt-800x1-launch-plan"
        or getattr(plan, "dataset_profile", None) != "core"
        or int(getattr(plan, "query_count", 0)) != 800
        or int(getattr(plan, "instance_count", 0)) != 800
        or int(getattr(plan, "shard_count", 0)) != 32
    ):
        raise ValueError("Static opt GCS v2 launch geometry is not 800x1")
    aliases = control.get("execution_artifact_aliases", [])
    if aliases not in (None, []):
        raise ValueError("Static opt GCS v2 cannot consume artifact aliases")


def _select_static_gcs_batch_ids(
    plan: object, requested: str | None
) -> tuple[str, ...]:
    available = _batch_ids(plan)
    if len(available) != 32:
        raise ValueError("Static opt launch must contain 32 complete batches")
    if requested is not None:
        if requested not in available:
            raise ValueError(f"unknown batch ID: {requested}")
        selected = (requested,)
    else:
        selected = available

    selected_shards = tuple(
        item
        for item in getattr(plan, "shards")
        if str(getattr(item, "accepted_batch_id")) in selected
    )
    if len(selected_shards) != len(selected):
        raise ValueError("Static opt selection must contain one shard per batch")
    flattened: list[str] = []
    for batch_id in selected:
        batch_shards = tuple(
            item
            for item in selected_shards
            if str(getattr(item, "accepted_batch_id")) == batch_id
        )
        if (
            len(batch_shards) != 1
            or str(getattr(batch_shards[0], "config")) != "llm_static"
            or int(getattr(batch_shards[0], "query_count")) != 25
            or len(tuple(getattr(batch_shards[0], "query_ids"))) != 25
        ):
            raise ValueError(f"Static opt batch is not one 25-query shard: {batch_id}")
        flattened.extend(str(item) for item in batch_shards[0].query_ids)
    if len(flattened) != 25 * len(selected) or len(set(flattened)) != len(flattened):
        raise ValueError("Static opt selection contains duplicate or missing queries")
    if requested is None and (
        len(flattened) != int(getattr(plan, "query_count"))
        or len(selected_shards) != int(getattr(plan, "shard_count"))
        or sum(int(getattr(item, "query_count")) for item in selected_shards)
        != int(getattr(plan, "instance_count"))
    ):
        raise ValueError("Static opt selection differs from launch geometry")
    return selected


def _load_static_runtime_task_specification(
    runtime: object,
) -> tuple[TaskSpecification, dict[str, object]]:
    """Recover TaskSpec from the verified runtime's typed semantic input."""

    lock = getattr(runtime, "runtime_lock")
    semantic_input = getattr(runtime, "semantic_authoring_input")
    raw = parse_canonical_json(
        semantic_input.task_specification.content,
        label="Static runtime embedded TaskSpec",
    )
    if not isinstance(raw, dict):
        raise ValueError("Static runtime embedded TaskSpec is not an object")
    task_spec = TaskSpecification.model_validate(raw, strict=True)
    active_task_spec = load_mvp_task_specification_v1()
    task_spec_file = read_stable_regular_file(
        MVP_TASK_SPEC_V1_PATH,
        label="GCS v2 active TaskSpec",
        max_bytes=4 * 1024 * 1024,
    )
    task_spec_file_sha256 = sha256_bytes(task_spec_file)
    if (
        task_spec != active_task_spec
        or semantic_input.task_specification.version != task_spec.task_spec_version
        or semantic_input.task_specification.identity_sha256
        != task_spec.task_spec_sha256
        or semantic_input.task_specification.content
        != canonical_json_bytes(task_spec.model_dump(mode="json"))
        or lock.get("task_spec_version") != task_spec.task_spec_version
        or lock.get("task_spec_sha256") != task_spec.task_spec_sha256
        or lock.get("task_spec_file_sha256") != task_spec_file_sha256
        or lock.get("semantic_authoring_input_file_sha256")
        != getattr(runtime, "semantic_authoring_input_file_sha256")
        or lock.get("semantic_authoring_input_sha256") != semantic_input.input_sha256
        or runtime.refresh_receipt.get("new_semantic_authoring_input_file_sha256")
        != runtime.semantic_authoring_input_file_sha256
        or runtime.refresh_receipt.get("new_semantic_authoring_input_sha256")
        != semantic_input.input_sha256
        or lock.get("static_contract_refresh_receipt_file_sha256")
        != getattr(runtime, "refresh_receipt_file_sha256")
        or lock.get("static_contract_refresh_receipt_sha256")
        != runtime.refresh_receipt.get("receipt_sha256")
        or lock.get("core_runtime_sources_receipt_file_sha256")
        != getattr(runtime, "core_source_receipt_file_sha256")
        or lock.get("core_runtime_sources_receipt_sha256")
        != runtime.core_source_receipt.get("receipt_sha256")
        or lock.get("runtime_data_sha256")
        != runtime.core_source_receipt.get("runtime_data_sha256")
    ):
        raise ValueError("Static runtime TaskSpec or refresh evidence binding drifted")
    return task_spec, {
        "task_spec_version": task_spec.task_spec_version,
        "task_spec_sha256": task_spec.task_spec_sha256,
        "task_spec_file_sha256": task_spec_file_sha256,
        "semantic_authoring_input_file_sha256": (
            runtime.semantic_authoring_input_file_sha256
        ),
        "semantic_authoring_input_sha256": semantic_input.input_sha256,
        "static_contract_refresh_receipt_file_sha256": (
            runtime.refresh_receipt_file_sha256
        ),
        "static_contract_refresh_receipt_sha256": runtime.refresh_receipt[
            "receipt_sha256"
        ],
        "core_runtime_sources_receipt_file_sha256": (
            runtime.core_source_receipt_file_sha256
        ),
        "core_runtime_sources_receipt_sha256": runtime.core_source_receipt[
            "receipt_sha256"
        ],
        "runtime_data_sha256": runtime.core_source_receipt["runtime_data_sha256"],
    }


def _assistant_request_from_checkpoint(value: object) -> AssistantRequestSnapshot:
    """Reparse canonical JSON while preserving strict tuple coercion semantics."""

    # Checkpoints are JSON, so immutable tuple fields necessarily arrive as
    # arrays.  ``model_validate(..., strict=True)`` rejects those arrays before
    # the model can restore its frozen tuple representation.  The JSON-aware
    # strict entrypoint is the contract used by the response/receipt loader too.
    return AssistantRequestSnapshot.model_validate_json(
        canonical_json_bytes(value), strict=True
    )


def _gcs_config_csv_row(summary: GCSConfigSummaryV2) -> dict[str, object]:
    statuses = dict(summary.style_support_status_counts)
    return {
        "config": summary.config,
        "scope": summary.scope,
        "query_count": summary.query_count,
        "component_count": summary.component_count,
        "population_mapping_sha256": summary.population_mapping_sha256,
        "query_micro_rate": summary.query_micro_rate,
        "headline_macro_rate": summary.headline_macro_rate,
        "hard_error_rate": summary.hard_error_rate,
        "oracle_coverage_complete": summary.oracle_coverage_complete,
        "headline_available": summary.headline_available,
        "style_candidates_count": statuses["candidates"],
        "style_no_result_count": statuses["no_result"],
        "style_unsupported_count": statuses["unsupported"],
        "style_unresolved_count": statuses["unresolved"],
        "component_failure_counts": summary.component_failure_counts,
        "failure_reason_counts": summary.failure_reason_counts,
    }


def _gcs_capability_csv_rows(
    summary: GCSConfigSummaryV2,
) -> list[dict[str, object]]:
    return [
        {
            "config": summary.config,
            "scope": summary.scope,
            "capability_id": item.capability_id,
            "query_count": item.query_count,
            "success_count": item.success_count,
            "gcs_rate": item.gcs_rate,
            "hard_error_count": item.hard_error_count,
            "hard_error_rate": item.hard_error_rate,
            "oracle_coverage_complete": item.oracle_coverage_complete,
            "component_failure_counts": item.component_failure_counts,
        }
        for item in summary.capabilities
    ]


def _static_not_applicable_artifacts(
    *, scope: str, population_mapping_sha256: str, query_count: int
) -> tuple[dict[str, object], dict[str, object]]:
    common = {
        "schema_version": 1,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "scope": scope,
        "population_mapping_sha256": population_mapping_sha256,
        "query_count": query_count,
        "status": "not_applicable",
        "reason_code": "single_config_static_opt_rollout",
        "model_calls_performed": 0,
    }
    bootstrap = {
        **common,
        "kind": "portfolio-gcs-paired-bootstrap",
        "available_configs": ["llm_static"],
        "required_config_count": 2,
    }
    gate = {
        **common,
        "kind": "portfolio-gcs-system-gain-gate",
        "baseline_config": "llm_static",
        "treatment_config": None,
        "required_treatment_config": "full",
    }
    return bootstrap, gate


def _external_analysis_sources(
    *,
    control: Mapping[str, object],
    launch: object,
    selected_batch_ids: Sequence[str],
    execution_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Resolve verified external and rollback-alias physical artifact sources."""

    sources: dict[str, object] = {}
    aliases = control.get("execution_artifact_aliases", [])
    if not isinstance(aliases, list):
        raise ValueError("execution artifact aliases must be a list")
    if aliases and execution_root is None:
        raise ValueError("artifact alias analysis requires the execution root")
    for batch_id in selected_batch_ids:
        batch_shards = tuple(
            item
            for item in launch.plan.shards
            if str(item.accepted_batch_id) == batch_id
        )
        by_config = {str(item.config): item for item in batch_shards}
        for alias in aliases:
            if not isinstance(alias, dict):
                raise ValueError("execution artifact alias must be an object")
            target = by_config.get(str(alias.get("target_config")))
            source_shard = by_config.get(str(alias.get("source_config")))
            if target is None:
                continue
            if (
                source_shard is None
                or alias.get("provider_model_call_count") != 0
                or source_shard.query_ids != target.query_ids
            ):
                raise ValueError("execution artifact alias source is invalid")
            assert execution_root is not None
            receipt_path = (
                execution_root / target.output_relpath / "artifact-alias.json"
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            unsigned = dict(receipt)
            supplied = unsigned.pop("alias_receipt_sha256", None)
            if (
                supplied != sha256_bytes(canonical_json_bytes(unsigned))
                or receipt.get("target_shard_id") != target.shard_id
                or receipt.get("source_shard_id") != source_shard.shard_id
                or receipt.get("treatment_alias_sha256") != alias.get("alias_sha256")
                or receipt.get("provider_model_call_count") != 0
            ):
                raise ValueError("execution artifact alias receipt is invalid")
            source_members = tuple(
                item
                for item in launch.instances
                if item.shard_id == source_shard.shard_id
            )
            sources[target.shard_id] = SimpleNamespace(
                execution_root=execution_root,
                shard=source_shard,
                members=source_members,
                external=False,
                artifact_alias=receipt,
            )

    descriptor = control.get("external_frozen_shard")
    if descriptor is None:
        return sources, None
    if not isinstance(descriptor, dict):
        raise ValueError("external_frozen_shard must be one object")
    accepted_batch_id = descriptor.get("accepted_batch_id")
    if tuple(selected_batch_ids) != (accepted_batch_id,):
        raise ValueError(
            "external shard repair analysis requires one explicit matching batch"
        )

    target_shards = tuple(
        item
        for item in launch.plan.shards
        if str(item.accepted_batch_id) == accepted_batch_id
    )
    target_noskill = next(
        (item for item in target_shards if str(item.config) == "noskill"),
        None,
    )
    if target_noskill is None:
        raise ValueError("external shard repair launch lacks target NoSkill")
    target_members = tuple(
        item for item in launch.instances if item.shard_id == target_noskill.shard_id
    )
    source, evidence, compatibility_violations = _load_external_frozen_source(
        control=control,
        target_launch=launch,
        target_shards=target_shards,
        target_members=target_members,
    )
    if source is None or evidence is None:
        raise ValueError("external frozen NoSkill source was not resolved")
    if compatibility_violations:
        raise ValueError(
            "external frozen NoSkill compatibility failed: "
            + "; ".join(compatibility_violations)
        )
    target_shard_id = str(target_noskill.shard_id)
    if evidence.get("target_shard_id") != target_shard_id:
        raise ValueError("external frozen NoSkill evidence targets another shard")
    sources[target_shard_id] = source
    return sources, evidence


def _artifact_projection(
    *,
    execution_root: Path,
    target_shard: object,
    target_members: Sequence[object],
    external_sources_by_shard: Mapping[str, object],
) -> tuple[Path, object, dict[str, object]]:
    """Map one logical launch shard to its verified physical artifact source."""

    target_shard_id = str(getattr(target_shard, "shard_id"))
    source = external_sources_by_shard.get(target_shard_id)
    if source is None:
        physical_root = execution_root
        physical_shard = target_shard
        physical_members = tuple(target_members)
    else:
        external = bool(getattr(source, "external", False))
        artifact_alias = getattr(source, "artifact_alias", None)
        if external and str(getattr(target_shard, "config")) != "noskill":
            raise ValueError("only NoSkill may use an external artifact source")
        if not external and not isinstance(artifact_alias, dict):
            raise ValueError("local physical source lacks an artifact alias")
        physical_root = Path(getattr(source, "execution_root"))
        physical_shard = getattr(source, "shard")
        physical_members = tuple(getattr(source, "members"))
        if external and str(getattr(physical_shard, "config")) != "noskill":
            raise ValueError("external artifact source is not one NoSkill shard")

    members_by_query = {
        str(getattr(member, "query_id")): member for member in physical_members
    }
    target_query_ids = {str(getattr(member, "query_id")) for member in target_members}
    if (
        len(members_by_query) != len(physical_members)
        or set(members_by_query) != target_query_ids
    ):
        raise ValueError("physical artifact source query membership differs")
    return physical_root, physical_shard, members_by_query


def _require_usable_audit(
    audit: Mapping[str, object],
    *,
    batch_id: str,
    expected_profile: str | None = None,
    expected_split: str | None = None,
) -> None:
    query_count = audit.get("query_count")
    if (
        audit.get("batch_id") != batch_id
        or audit.get("status") not in {"passed_clean", "review_required"}
        or type(query_count) is not int
        or query_count <= 0
        or audit.get("row_count") != query_count * len(MAIN_CONFIG_ORDER)
        or audit.get("blockers") != []
        or audit.get("model_calls_performed") != 0
        or (
            expected_profile is not None
            and audit.get("dataset_profile") != expected_profile
        )
        or (expected_split is not None and audit.get("batch_split") != expected_split)
    ):
        raise ValueError(f"batch audit failed closed: {batch_id}")


def _core_batch_split(
    plan: object,
    *,
    batch_id: str,
    layout: _PartitionLayout,
) -> str:
    query_ids = {
        str(query_id)
        for shard in getattr(plan, "shards")
        if str(getattr(shard, "accepted_batch_id")) == batch_id
        for query_id in getattr(shard, "query_ids")
    }
    splits = {layout.partition_by_query.get(query_id) for query_id in query_ids}
    if None in splits or len(splits) != 1:
        raise ValueError(f"Core batch is not split-atomic: {batch_id}")
    return str(next(iter(splits)))


def _validate_partition_contract(
    *,
    optimization_query_ids: Sequence[str],
    evaluation_query_ids: Sequence[str],
    optimization_leakage_group_ids: Sequence[str],
    evaluation_leakage_group_ids: Sequence[str],
) -> None:
    optimization = tuple(optimization_query_ids)
    evaluation = tuple(evaluation_query_ids)
    if (
        len(optimization) != 25
        or len(evaluation) != 175
        or len(set(optimization)) != 25
        or len(set(evaluation)) != 175
        or set(optimization) & set(evaluation)
        or len(set(optimization + evaluation)) != 200
    ):
        raise ValueError("treatment chain does not bind a disjoint 25/175 split")
    optimization_groups = set(optimization_leakage_group_ids)
    evaluation_groups = set(evaluation_leakage_group_ids)
    if (
        not optimization_groups
        or not evaluation_groups
        or optimization_groups & evaluation_groups
    ):
        raise ValueError("optimization/evaluation leakage clusters overlap")


def _build_partition_layout(
    *,
    plan: object,
    manifest: object,
    profile_inputs: object,
) -> _PartitionLayout:
    """Bind row partitions to the corpus profile, never to filename guesses."""

    profile = str(getattr(profile_inputs, "profile"))
    if profile == "core":
        selected_splits = tuple(str(item) for item in getattr(plan, "selected_splits"))
        if not selected_splits or selected_splits != tuple(
            getattr(profile_inputs, "selected_splits")
        ):
            raise ValueError("Core analysis split selection differs from launch")
        population_counts = dict(getattr(profile_inputs, "population_split_counts"))
        expected = {
            split: int(population_counts.get(split, 0)) for split in selected_splits
        }
        if any(count <= 0 for count in expected.values()) or sum(
            expected.values()
        ) != int(getattr(plan, "query_count")):
            raise ValueError("Core split populations differ from launch query count")
        split_by_query = dict(getattr(profile_inputs, "split_by_query"))
        selected_query_ids = {
            str(query_id)
            for shard in getattr(plan, "shards")
            for query_id in getattr(shard, "query_ids")
        }
        partition_by_query = {
            query_id: split_by_query[query_id]
            for query_id in selected_query_ids
            if query_id in split_by_query
        }
        if set(partition_by_query) != selected_query_ids or set(
            partition_by_query.values()
        ) - set(selected_splits):
            raise ValueError("Core launch queries are outside the selected splits")
        use_policy = {
            "primary_effect_scope": (
                "test_frozen" if "test_frozen" in selected_splits else "not_selected"
            ),
            "dev_mini": "smoke_and_diagnostic_evidence",
            "opt_pool": "training_and_candidate_generation_not_acceptance",
            "val": "candidate_acceptance_and_rollback_evidence",
            "test_frozen": "sealed_headline_effect_only",
            "all_selected": "descriptive_aggregate_over_selected_splits",
        }
        return _PartitionLayout(
            profile="core",
            partition_order=selected_splits,
            partition_by_query=partition_by_query,
            expected_query_counts=expected,
            aggregate_name="all_selected",
            primary_effect_scope=use_policy["primary_effect_scope"],
            use_policy=use_policy,
        )

    if profile != "dev_mini":
        raise ValueError(f"unsupported analysis profile: {profile}")
    _validate_partition_contract(
        optimization_query_ids=manifest.optimization_query_ids,
        evaluation_query_ids=manifest.evaluation_query_ids,
        optimization_leakage_group_ids=manifest.optimization_leakage_group_ids,
        evaluation_leakage_group_ids=manifest.evaluation_leakage_group_ids,
    )
    partition_by_query = {
        **{query_id: "optimization25" for query_id in manifest.optimization_query_ids},
        **{query_id: "evaluation175" for query_id in manifest.evaluation_query_ids},
    }
    return _PartitionLayout(
        profile="dev_mini",
        partition_order=("optimization25", "evaluation175"),
        partition_by_query=partition_by_query,
        expected_query_counts={
            "optimization25": len(manifest.optimization_query_ids),
            "evaluation175": len(manifest.evaluation_query_ids),
        },
        aggregate_name="all200",
        primary_effect_scope="evaluation175",
        use_policy={
            "primary_effect_scope": "evaluation175",
            "optimization25": "tuning_and_diagnostic_evidence_not_headline_effect",
            "all200": "descriptive_aggregate_not_disjoint_holdout_effect",
        },
    )


def _adherence_contract_banks(chain: object) -> dict[str, object]:
    output_banks = dict(getattr(chain, "output_banks"))
    by_sha = {bank.bank_sha256: bank for bank in output_banks.values()}
    contracts: dict[str, object] = {"llm_static": output_banks["llm_static"]}
    source_chain = getattr(chain, "source_chain", None)
    compatibility_rebind = getattr(chain, "compatibility_rebind", None)
    if (source_chain is None) != (compatibility_rebind is None):
        raise ValueError("incomplete compatibility-rebind adherence lineage")

    parent_by_config = {
        "s1": "llm_static",
        "s1s2": "s1",
        "full": "s1s2",
    }
    for config, parent_config in parent_by_config.items():
        report = getattr(chain, "gate_reports")[config]
        if source_chain is None:
            bank = by_sha.get(report.adherence_contract_bank_sha256)
        else:
            source_report = getattr(source_chain, "gate_reports")[config]
            source_bank = getattr(source_chain, "output_banks")[parent_config]
            rebound_bank = output_banks[parent_config]
            matching_bindings = [
                item
                for item in compatibility_rebind.bank_bindings
                if item.config == parent_config and item.role == "output"
            ]
            bank = None
            if (
                report.adherence_contract_bank_sha256
                == source_report.adherence_contract_bank_sha256
                == source_bank.bank_sha256
                and len(matching_bindings) == 1
                and matching_bindings[0].source_bank_sha256 == source_bank.bank_sha256
                and matching_bindings[0].rebound_bank_sha256 == rebound_bank.bank_sha256
            ):
                bank = rebound_bank
        if bank is None:
            raise ValueError(f"{config} gate adherence contract Bank is unavailable")
        contracts[config] = bank
    return contracts


def _attempt_inventory(
    *,
    shard_root: Path,
    members: Sequence[object],
) -> dict[str, tuple[object, ...]]:
    observed_paths: set[Path] = set()
    by_query: dict[str, tuple[object, ...]] = {}
    for member in members:
        receipts = load_query_attempt_receipts(
            shard_root,
            query_ordinal=int(getattr(member, "query_ordinal")),
            query_id=str(getattr(member, "query_id")),
            instance_sha256=str(getattr(member, "instance_sha256")),
        )
        by_query[str(getattr(member, "query_id"))] = receipts
        for receipt in receipts:
            observed_paths.add(
                shard_root
                / "attempt-receipts"
                / (
                    f"{receipt.query_ordinal:04d}-attempt-"
                    f"{receipt.attempt_index:02d}.json"
                )
            )
    directory = shard_root / "attempt-receipts"
    entries = tuple(directory.iterdir()) if directory.exists() else ()
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError("attempt-receipt directory contains a non-regular entry")
    actual_paths = set(entries)
    if actual_paths != observed_paths:
        raise ValueError("attempt-receipt directory contains unbound or unsafe files")
    return by_query


def _extract_rows(
    *,
    execution_root: Path,
    launch: object,
    selected_batch_ids: Sequence[str],
    private_by_id: Mapping[str, object],
    partition_by_query: Mapping[str, str],
    output_banks: Mapping[str, object],
    adherence_banks: Mapping[str, object],
    external_sources_by_shard: Mapping[str, object] | None = None,
) -> list[MatrixRow]:
    selected = set(selected_batch_ids)
    external_sources = external_sources_by_shard or {}
    rows: list[MatrixRow] = []
    shards = tuple(
        item for item in launch.plan.shards if str(item.accepted_batch_id) in selected
    )
    for shard in shards:
        members = tuple(
            sorted(
                (item for item in launch.instances if item.shard_id == shard.shard_id),
                key=lambda item: item.query_ordinal,
            )
        )
        if len(members) != 25:
            raise ValueError(f"shard {shard.shard_id} does not contain 25 members")
        physical_root, physical_shard, physical_members_by_query = _artifact_projection(
            execution_root=execution_root,
            target_shard=shard,
            target_members=members,
            external_sources_by_shard=external_sources,
        )
        shard_root = physical_root / physical_shard.output_relpath
        attempts_by_query = _attempt_inventory(
            shard_root=shard_root,
            members=tuple(physical_members_by_query.values()),
        )
        for member in members:
            query_id = str(member.query_id)
            physical_member = physical_members_by_query[query_id]
            private = private_by_id.get(query_id)
            if private is None:
                raise ValueError(f"missing frozen private query: {query_id}")
            partition = partition_by_query.get(query_id)
            if partition is None:
                raise ValueError(
                    f"query is outside the launch partition map: {query_id}"
                )

            assistant_path = physical_root / physical_member.assistant_output_relpath
            response, receipt = _assistant_row(assistant_path)
            final_path = physical_root / physical_member.final_output_relpath
            assistant_error_code = response.error_code
            final_raw = None
            if response.error_code is not None:
                judge_status = "not_invoked_assistant_error"
                judge_error_code = None
                judge_provider = None
                judge_model = None
                judge_finish_reason = None
                card_guard_adjusted = None
                j_project = 0.0
            else:
                final_raw = json.loads(final_path.read_text(encoding="utf-8"))
                if final_raw.get("kind") == "portfolio-final-fixed-zero":
                    # A typed fail-closed packet sanitizer result is a valid
                    # Assistant hard-error row, not a visual Judge receipt.
                    assistant_error_code = str(final_raw.get("assistant_error_code"))
                    judge_status = "not_invoked_assistant_error"
                    judge_error_code = None
                    judge_provider = None
                    judge_model = None
                    judge_finish_reason = None
                    card_guard_adjusted = None
                    j_project = 0.0
                else:
                    final = load_final_judge_evaluation_result(final_path)
                    judge_status = final.outcome.status
                    judge_error_code = final.outcome.error_code
                    judge_provider = final.provider
                    judge_model = final.model
                    judge_finish_reason = final.finish_reason
                    card_guard_adjusted = final.card_requirement_guard_adjusted
                    j_project = float(final.outcome.scores.j_project)
            if not math.isfinite(j_project) or not 0.0 <= j_project <= 100.0:
                raise ValueError(f"invalid J_project score: {query_id}/{shard.config}")

            assistant_hard_error = assistant_error_code is not None
            evaluator_anomaly = not assistant_hard_error and judge_status != "scored"
            bank = output_banks.get(shard.config)
            adherence_bank = adherence_banks.get(shard.config)
            if shard.config == "noskill":
                if bank is not None or adherence_bank is not None:
                    raise ValueError("NoSkill unexpectedly has an adherence Bank")
                bank_sha256 = None
                structural_adherence = None
                gate_aligned_adherence = None
            else:
                if bank is None or adherence_bank is None:
                    raise ValueError(f"missing adherence Bank: {shard.config}")
                bank_sha256 = bank.bank_sha256
                structural_adherence = calculate_portfolio_skill_adherence(
                    bank=bank,
                    selected_capability=response.selected_capability,
                    skill_slug=response.skill_slug,
                    response_text=response.response_text,
                    visible_cards=response.visible_cards,
                    hard_error=assistant_hard_error,
                )
                gate_aligned_adherence = calculate_portfolio_skill_adherence(
                    bank=adherence_bank,
                    selected_capability=response.selected_capability,
                    skill_slug=response.skill_slug,
                    response_text=response.response_text,
                    visible_cards=response.visible_cards,
                    hard_error=assistant_hard_error,
                    require_skill_slug_match=False,
                )

            tool_errors = tuple(
                item for item in response.tool_trace if item.status != "success"
            )
            if tool_errors:
                tool_recovery_status = (
                    "recovered" if response.error_code is None else "unresolved"
                )
            else:
                tool_recovery_status = "none"
            attempts = attempts_by_query[query_id]
            attempt_recovery_status = "recovered" if attempts else "none"
            selected_capability = response.selected_capability
            acceptable = tuple(getattr(private, "acceptable_capabilities"))
            requires_card = bool(getattr(private, "requires_card"))
            visible_card_count = len(response.visible_cards)
            rows.append(
                MatrixRow(
                    query_ordinal=int(member.query_ordinal),
                    query_id=query_id,
                    batch_id=str(shard.accepted_batch_id),
                    partition=partition,
                    leakage_group_id=str(getattr(private, "leakage_group_id")),
                    canonical_capability=str(getattr(private, "canonical_capability")),
                    config=str(shard.config),
                    bank_sha256=bank_sha256,
                    j_project=j_project,
                    selected_capability=selected_capability,
                    route_correct=(
                        selected_capability
                        == str(getattr(private, "canonical_capability"))
                    ),
                    route_acceptable=selected_capability in acceptable,
                    structural_adherence=structural_adherence,
                    gate_aligned_adherence=gate_aligned_adherence,
                    assistant_hard_error=assistant_hard_error,
                    evaluator_anomaly=evaluator_anomaly,
                    assistant_status=(
                        "success" if response.error_code is None else "error"
                    ),
                    assistant_error_code=assistant_error_code,
                    assistant_receipt_outcome=receipt.outcome,
                    assistant_model_call_count=len(receipt.model_calls),
                    assistant_finish_reasons=tuple(
                        item.finish_reason for item in receipt.model_calls
                    ),
                    tool_call_count=len(response.tool_trace),
                    tool_error_count=len(tool_errors),
                    tool_error_codes=tuple(
                        str(item.error_code) for item in tool_errors
                    ),
                    tool_statuses=tuple(
                        f"{item.tool_name}:{item.status}:{item.error_code or ''}"
                        for item in response.tool_trace
                    ),
                    tool_recovery_status=tool_recovery_status,
                    card_requirement="required" if requires_card else "forbidden",
                    visible_card_count=visible_card_count,
                    card_policy_compliant=(
                        (requires_card and visible_card_count > 0)
                        or (not requires_card and visible_card_count == 0)
                    ),
                    card_guard_adjusted=card_guard_adjusted,
                    judge_status=judge_status,
                    judge_error_code=judge_error_code,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                    judge_finish_reason=judge_finish_reason,
                    attempt_receipt_count=len(attempts),
                    attempt_failure_stages=tuple(
                        item.failure_stage for item in attempts
                    ),
                    attempt_failure_subtypes=tuple(
                        item.failure_subtype for item in attempts
                    ),
                    attempt_recovery_status=attempt_recovery_status,
                    attempt_circuit_open_count=sum(
                        item.circuit_open for item in attempts
                    ),
                    attempt_captured_provider_response_count=sum(
                        item.captured_provider_response_count for item in attempts
                    ),
                    attempt_captured_input_tokens=sum(
                        item.captured_input_tokens for item in attempts
                    ),
                    attempt_captured_output_tokens=sum(
                        item.captured_output_tokens for item in attempts
                    ),
                )
            )
    config_ordinal = {config: index for index, config in enumerate(MAIN_CONFIG_ORDER)}
    return sorted(
        rows,
        key=lambda item: (item.query_ordinal, config_ordinal[item.config]),
    )


def _validate_rectangular(rows: Sequence[MatrixRow]) -> None:
    by_query: dict[str, list[MatrixRow]] = defaultdict(list)
    for row in rows:
        by_query[row.query_id].append(row)
    if not by_query:
        raise ValueError("analysis selection is empty")
    for query_id, query_rows in by_query.items():
        configs = tuple(item.config for item in query_rows)
        if sorted(configs) != sorted(MAIN_CONFIG_ORDER):
            raise ValueError(f"query lacks one exact five-config pair set: {query_id}")
        invariant_values = {
            (
                item.query_ordinal,
                item.batch_id,
                item.partition,
                item.leakage_group_id,
                item.canonical_capability,
            )
            for item in query_rows
        }
        if len(invariant_values) != 1:
            raise ValueError(f"paired query metadata drifted: {query_id}")
    if len(rows) != len(by_query) * len(MAIN_CONFIG_ORDER):
        raise ValueError("analysis rows are not rectangular")


def _macro_f1(rows: Sequence[MatrixRow]) -> dict[str, object]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    confusion: dict[str, Counter[str]] = {
        capability: Counter() for capability in CAPABILITIES
    }
    for row in rows:
        prediction = (
            row.selected_capability
            if row.selected_capability in CAPABILITIES
            else "__missing_or_invalid__"
        )
        confusion[row.canonical_capability][prediction] += 1
    for capability in CAPABILITIES:
        tp = sum(
            row.canonical_capability == capability
            and row.selected_capability == capability
            for row in rows
        )
        fp = sum(
            row.canonical_capability != capability
            and row.selected_capability == capability
            for row in rows
        )
        fn = sum(
            row.canonical_capability == capability
            and row.selected_capability != capability
            for row in rows
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[capability] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "macro_f1": sum(f1_values) / len(CAPABILITIES),
        "classes": per_class,
        "confusion": {
            key: dict(sorted(value.items())) for key, value in sorted(confusion.items())
        },
        "missing_or_invalid_prediction_count": sum(
            row.selected_capability not in CAPABILITIES for row in rows
        ),
        "denominator": len(rows),
    }


def _mean_optional(values: Sequence[float | None]) -> tuple[float | None, int]:
    present = tuple(value for value in values if value is not None)
    if present and len(present) != len(values):
        raise ValueError("adherence projection mixes applicable and N/A rows")
    return (sum(present) / len(present), len(present)) if present else (None, 0)


def _counter(values: Iterable[str | None]) -> dict[str, int]:
    counts = Counter("none" if item is None else item for item in values)
    return dict(sorted(counts.items()))


def _config_summary(rows: Sequence[MatrixRow], scope: str) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for config in MAIN_CONFIG_ORDER:
        selected = tuple(item for item in rows if item.config == config)
        if not selected:
            raise ValueError(f"scope {scope} lacks config {config}")
        structural_mean, structural_n = _mean_optional(
            tuple(item.structural_adherence for item in selected)
        )
        gate_mean, gate_n = _mean_optional(
            tuple(item.gate_aligned_adherence for item in selected)
        )
        routing = None if config == "noskill" else _macro_f1(selected)
        summaries.append(
            {
                "scope": scope,
                "config": config,
                "query_count": len(selected),
                "j_denominator_all_rows": len(selected),
                "mean_j_project_all_rows": sum(item.j_project for item in selected)
                / len(selected),
                "median_j_project_all_rows": statistics.median(
                    item.j_project for item in selected
                ),
                "routing_applicability": (
                    "not_applicable" if routing is None else "applicable"
                ),
                "route_accuracy_missing_is_wrong": (
                    None
                    if routing is None
                    else sum(item.route_correct for item in selected) / len(selected)
                ),
                "route_acceptable_rate_missing_is_wrong": (
                    None
                    if routing is None
                    else sum(item.route_acceptable for item in selected) / len(selected)
                ),
                "route_macro_f1_6class_missing_is_wrong": (
                    None if routing is None else routing["macro_f1"]
                ),
                "route_missing_or_invalid_prediction_count": (
                    None
                    if routing is None
                    else routing["missing_or_invalid_prediction_count"]
                ),
                "route_classes": None if routing is None else routing["classes"],
                "route_confusion": (None if routing is None else routing["confusion"]),
                "structural_adherence_mean": structural_mean,
                "structural_adherence_denominator": structural_n,
                "gate_aligned_adherence_mean": gate_mean,
                "gate_aligned_adherence_denominator": gate_n,
                "hard_error_semantics": "assistant_execution_only",
                "hard_error_count": sum(item.assistant_hard_error for item in selected),
                "assistant_hard_error_count": sum(
                    item.assistant_hard_error for item in selected
                ),
                "evaluator_anomaly_count": sum(
                    item.evaluator_anomaly for item in selected
                ),
                "assistant_status_counts": _counter(
                    item.assistant_status for item in selected
                ),
                "assistant_error_code_counts": _counter(
                    item.assistant_error_code for item in selected
                ),
                "judge_status_counts": _counter(item.judge_status for item in selected),
                "judge_error_code_counts": _counter(
                    item.judge_error_code for item in selected
                ),
                "tool_error_count": sum(item.tool_error_count for item in selected),
                "tool_error_row_count": sum(
                    item.tool_error_count > 0 for item in selected
                ),
                "tool_recovery_status_counts": _counter(
                    item.tool_recovery_status for item in selected
                ),
                "recovered_tool_error_row_count": sum(
                    item.tool_recovery_status == "recovered" for item in selected
                ),
                "unresolved_tool_error_row_count": sum(
                    item.tool_recovery_status == "unresolved" for item in selected
                ),
                "card_policy_violation_count": sum(
                    not item.card_policy_compliant for item in selected
                ),
                "card_guard_adjusted_count": sum(
                    item.card_guard_adjusted is True for item in selected
                ),
                "attempt_receipt_count": sum(
                    item.attempt_receipt_count for item in selected
                ),
                "attempt_receipt_row_count": sum(
                    item.attempt_receipt_count > 0 for item in selected
                ),
                "recovered_attempt_receipt_count": sum(
                    item.attempt_receipt_count
                    for item in selected
                    if item.attempt_recovery_status == "recovered"
                ),
                "unresolved_attempt_receipt_count": sum(
                    item.attempt_receipt_count
                    for item in selected
                    if item.attempt_recovery_status == "unresolved"
                ),
                "attempt_recovery_status_counts": _counter(
                    item.attempt_recovery_status for item in selected
                ),
                "attempt_failure_stage_counts": _counter(
                    stage for item in selected for stage in item.attempt_failure_stages
                ),
                "attempt_failure_subtype_counts": _counter(
                    subtype
                    for item in selected
                    for subtype in item.attempt_failure_subtypes
                ),
            }
        )
    return summaries


def _capability_summary(
    rows: Sequence[MatrixRow],
    scope: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    routing_by_config = {
        config: (
            None
            if config == "noskill"
            else _macro_f1(tuple(item for item in rows if item.config == config))
        )
        for config in MAIN_CONFIG_ORDER
    }
    for config in MAIN_CONFIG_ORDER:
        for capability in CAPABILITIES:
            selected = tuple(
                item
                for item in rows
                if item.config == config and item.canonical_capability == capability
            )
            if not selected:
                continue
            structural_mean, structural_n = _mean_optional(
                tuple(item.structural_adherence for item in selected)
            )
            gate_mean, gate_n = _mean_optional(
                tuple(item.gate_aligned_adherence for item in selected)
            )
            routing = routing_by_config[config]
            route_class = None if routing is None else routing["classes"][capability]
            result.append(
                {
                    "scope": scope,
                    "config": config,
                    "canonical_capability": capability,
                    "query_count": len(selected),
                    "j_denominator_all_rows": len(selected),
                    "mean_j_project_all_rows": sum(item.j_project for item in selected)
                    / len(selected),
                    "median_j_project_all_rows": statistics.median(
                        item.j_project for item in selected
                    ),
                    "routing_applicability": (
                        "not_applicable" if route_class is None else "applicable"
                    ),
                    "route_accuracy_missing_is_wrong": (
                        None
                        if route_class is None
                        else sum(item.route_correct for item in selected)
                        / len(selected)
                    ),
                    "route_acceptable_rate_missing_is_wrong": (
                        None
                        if route_class is None
                        else sum(item.route_acceptable for item in selected)
                        / len(selected)
                    ),
                    "route_class_f1": (
                        None if route_class is None else route_class["f1"]
                    ),
                    "route_class_precision": (
                        None if route_class is None else route_class["precision"]
                    ),
                    "route_class_recall": (
                        None if route_class is None else route_class["recall"]
                    ),
                    "structural_adherence_mean": structural_mean,
                    "structural_adherence_denominator": structural_n,
                    "gate_aligned_adherence_mean": gate_mean,
                    "gate_aligned_adherence_denominator": gate_n,
                    "hard_error_semantics": "assistant_execution_only",
                    "hard_error_count": sum(
                        item.assistant_hard_error for item in selected
                    ),
                    "assistant_hard_error_count": sum(
                        item.assistant_hard_error for item in selected
                    ),
                    "evaluator_anomaly_count": sum(
                        item.evaluator_anomaly for item in selected
                    ),
                    "assistant_error_count": sum(
                        item.assistant_status == "error" for item in selected
                    ),
                    "judge_non_scored_count": sum(
                        item.evaluator_anomaly for item in selected
                    ),
                    "tool_error_count": sum(item.tool_error_count for item in selected),
                    "card_policy_violation_count": sum(
                        not item.card_policy_compliant for item in selected
                    ),
                    "attempt_receipt_count": sum(
                        item.attempt_receipt_count for item in selected
                    ),
                }
            )
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap(
    paired: Sequence[tuple[str, str, float]],
    *,
    scope: str,
    baseline_config: str,
    treatment_config: str,
) -> dict[str, object]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for _query_id, leakage_group_id, delta in paired:
        clusters[leakage_group_id].append(delta)
    cluster_ids = tuple(sorted(clusters))
    if not cluster_ids:
        raise ValueError("paired bootstrap lacks leakage clusters")
    seed_material = (
        f"{BOOTSTRAP_SEED}|{scope}|{baseline_config}|{treatment_config}"
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = random.Random(derived_seed)
    means: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = [generator.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster in sampled for value in clusters[cluster]]
        means.append(sum(values) / len(values))
    return {
        "policy": "leakage-cluster-resample-with-replacement-v1",
        "fixed_seed": BOOTSTRAP_SEED,
        "derived_seed_sha256_prefix_u64": derived_seed,
        "iterations": BOOTSTRAP_ITERATIONS,
        "cluster_count": len(cluster_ids),
        "mean_delta_ci95_low": _percentile(means, 0.025),
        "mean_delta_ci95_high": _percentile(means, 0.975),
    }


def _paired_summaries(
    rows: Sequence[MatrixRow],
    *,
    scope: str,
    bank_sha256s: Mapping[str, str | None],
) -> list[dict[str, object]]:
    by_key = {(item.query_id, item.config): item for item in rows}
    query_ids = sorted({item.query_id for item in rows})
    result: list[dict[str, object]] = []
    for baseline, treatment in combinations(MAIN_CONFIG_ORDER, 2):
        same_bank = bank_sha256s.get(baseline) is not None and bank_sha256s.get(
            baseline
        ) == bank_sha256s.get(treatment)
        paired: list[tuple[str, str, float]] = []
        baseline_scores: list[float] = []
        treatment_scores: list[float] = []
        for query_id in query_ids:
            baseline_row = by_key[(query_id, baseline)]
            treatment_row = by_key[(query_id, treatment)]
            if baseline_row.leakage_group_id != treatment_row.leakage_group_id:
                raise ValueError(f"paired leakage cluster drift: {query_id}")
            baseline_scores.append(baseline_row.j_project)
            treatment_scores.append(treatment_row.j_project)
            if same_bank and treatment_row.j_project != baseline_row.j_project:
                raise ValueError(
                    "same-output Bank contrast contains independently sampled "
                    f"scores: {baseline}->{treatment}/{query_id}"
                )
            paired.append(
                (
                    query_id,
                    baseline_row.leakage_group_id,
                    treatment_row.j_project - baseline_row.j_project,
                )
            )
        deltas = [item[2] for item in paired]
        if same_bank and any(value != 0.0 for value in deltas):  # pragma: no cover
            raise AssertionError("artifact alias must produce exact zero deltas")
        result.append(
            {
                "scope": scope,
                "baseline_config": baseline,
                "treatment_config": treatment,
                "paired_query_count": len(paired),
                "baseline_mean_j_project_all_rows": sum(baseline_scores)
                / len(baseline_scores),
                "treatment_mean_j_project_all_rows": sum(treatment_scores)
                / len(treatment_scores),
                "paired_mean_delta_j_project": sum(deltas) / len(deltas),
                "paired_median_delta_j_project": statistics.median(deltas),
                "win_count": sum(value > 0.0 for value in deltas),
                "tie_count": sum(value == 0.0 for value in deltas),
                "loss_count": sum(value < 0.0 for value in deltas),
                "same_output_bank_no_op_contrast": same_bank,
                "treatment_attribution_allowed": not same_bank,
                "contrast_interpretation": (
                    "same_bank_no_op_contrast_do_not_attribute_to_treatment"
                    if same_bank
                    else "distinct_bank_observational_contrast"
                ),
                **_cluster_bootstrap(
                    paired,
                    scope=scope,
                    baseline_config=baseline,
                    treatment_config=treatment,
                ),
            }
        )
    return result


def _scope_rows(
    rows: Sequence[MatrixRow],
    *,
    full_matrix: bool,
    batch_id: str | None,
    layout: _PartitionLayout,
) -> dict[str, tuple[MatrixRow, ...]]:
    if not full_matrix and layout.profile == "dev_mini":
        if batch_id is None:
            raise ValueError("partial analysis lacks a batch ID")
        return {f"batch:{batch_id}": tuple(rows)}
    scopes = {
        partition: tuple(item for item in rows if item.partition == partition)
        for partition in layout.partition_order
    }
    if not full_matrix:
        if batch_id is None:
            raise ValueError("partial analysis lacks a batch ID")
        nonempty = {key: value for key, value in scopes.items() if value}
        if len(nonempty) != 1:
            raise ValueError("partial Core batch is not split-atomic")
        return nonempty
    scopes[layout.aggregate_name] = tuple(rows)
    expected_rows = {
        partition: count * len(MAIN_CONFIG_ORDER)
        for partition, count in layout.expected_query_counts.items()
    }
    expected_rows[layout.aggregate_name] = sum(
        layout.expected_query_counts.values()
    ) * len(MAIN_CONFIG_ORDER)
    if any(len(scopes.get(key, ())) != value for key, value in expected_rows.items()):
        raise ValueError("full matrix does not reproduce launch partition row counts")
    return scopes


def _partition_coverage(
    rows: Sequence[MatrixRow],
    *,
    layout: _PartitionLayout,
) -> dict[str, dict[str, int | bool]]:
    query_ids = {item.query_id for item in rows}
    result = {
        partition: {
            "selected_query_count": len(
                {item.query_id for item in rows if item.partition == partition}
            ),
            "expected_query_count": expected,
            "population_complete": len(
                {item.query_id for item in rows if item.partition == partition}
            )
            == expected,
        }
        for partition, expected in layout.expected_query_counts.items()
    }
    expected_total = sum(layout.expected_query_counts.values())
    result[layout.aggregate_name] = {
        "selected_query_count": len(query_ids),
        "expected_query_count": expected_total,
        "population_complete": len(query_ids) == expected_total,
    }
    return result


def _json_cell(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return (
            canonical_json_bytes(_json_projection(value)).decode("utf-8").rstrip("\n")
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("CSV artifact cannot be empty")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError("CSV rows do not share one ordered schema")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _json_cell(value) for key, value in row.items()})
    return output.getvalue().encode("utf-8")


def _row_csv_projection(row: MatrixRow) -> dict[str, object]:
    return asdict(row)


def _summary_csv_projection(row: Mapping[str, object]) -> dict[str, object]:
    return dict(row)


def _report(
    *,
    selection: str,
    partition_coverage: Mapping[str, Mapping[str, object]],
    config_summaries: Sequence[Mapping[str, object]],
    paired_summaries: Sequence[Mapping[str, object]],
    no_op_contrasts: Sequence[Mapping[str, object]],
    audit_records: Sequence[Mapping[str, object]],
    external_frozen_shard: Mapping[str, object] | None = None,
    partition_use_policy: Mapping[str, str] | None = None,
) -> bytes:
    effective_partition_policy = partition_use_policy or {
        "primary_effect_scope": "evaluation175",
    }
    primary_scope = effective_partition_policy.get(
        "primary_effect_scope", "not_selected"
    )
    partition_explanation = (
        "The disjoint evaluation175 partition is the primary effect scope; "
        "optimization25 is tuning evidence and all200 is descriptive only."
        if tuple(partition_coverage)
        == (
            "optimization25",
            "evaluation175",
            "all200",
        )
        else (
            "Core rows retain their frozen corpus split. The primary effect "
            f"scope for this launch is `{primary_scope}`; aggregate selected "
            "rows are descriptive and never relabeled as dev_mini holdout data."
        )
    )
    lines = [
        "# Portfolio matrix analysis",
        "",
        f"Selection: `{selection}`. This is Portfolio-track, read-only analysis.",
        "J_project always uses the complete selected denominator. Routing is not "
        "applicable to NoSkill; for skilled configurations, missing predictions "
        "count as errors in six-class macro-F1.",
        "Assistant execution hard errors and evaluator/Judge anomalies are "
        "reported separately; evaluator anomalies are not algorithm hard errors.",
        partition_explanation,
        "",
        "## Partition coverage",
        "",
        "| Partition | Selected | Expected | Complete |",
        "|---|---:|---:|:---:|",
    ]
    for name in partition_coverage:
        coverage = partition_coverage[name]
        lines.append(
            f"| {name} | {coverage['selected_query_count']} | "
            f"{coverage['expected_query_count']} | "
            f"{'yes' if coverage['population_complete'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Config summary",
            "",
            "| Scope | Config | N | Mean J | 6-class macro-F1 | Structural "
            "adherence | Gate-aligned adherence | Assistant hard errors | "
            "Evaluator anomalies |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in config_summaries:
        structural = item["structural_adherence_mean"]
        gate = item["gate_aligned_adherence_mean"]
        route_macro_f1 = item["route_macro_f1_6class_missing_is_wrong"]
        lines.append(
            f"| {item['scope']} | {item['config']} | {item['query_count']} | "
            f"{float(item['mean_j_project_all_rows']):.4f} | "
            f"{'N/A' if route_macro_f1 is None else f'{float(route_macro_f1):.4f}'} | "
            f"{'' if structural is None else f'{float(structural):.4f}'} | "
            f"{'' if gate is None else f'{float(gate):.4f}'} | "
            f"{item['assistant_hard_error_count']} | "
            f"{item['evaluator_anomaly_count']} |"
        )
    lines.extend(["", "## Same-Bank no-op contrasts", ""])
    if no_op_contrasts:
        lines.append(
            "The following contrasts share exactly the same output Bank and must "
            "not be attributed to a treatment mutation:"
        )
        lines.append("")
        for item in no_op_contrasts:
            lines.append(
                f"- `{item['baseline_config']} -> {item['treatment_config']}` "
                f"(`{item['bank_sha256']}`)"
            )
    else:
        lines.append("No same-output-Bank contrast was detected.")
    lines.extend(
        [
            "",
            "## Paired deltas",
            "",
            "| Scope | Contrast | Mean delta J | Median | W/T/L | Cluster CI95 | "
            "Interpretation |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in paired_summaries:
        lines.append(
            f"| {item['scope']} | {item['baseline_config']} -> "
            f"{item['treatment_config']} | "
            f"{float(item['paired_mean_delta_j_project']):.4f} | "
            f"{float(item['paired_median_delta_j_project']):.4f} | "
            f"{item['win_count']}/{item['tie_count']}/{item['loss_count']} | "
            f"[{float(item['mean_delta_ci95_low']):.4f}, "
            f"{float(item['mean_delta_ci95_high']):.4f}] | "
            f"{item['contrast_interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"{len(audit_records)} batch audits were run before and after row "
            "projection and matched byte-for-value. No model calls were made.",
            "",
        ]
    )
    if external_frozen_shard is not None:
        lines.extend(
            [
                "NoSkill rows were read directly from the strictly validated "
                "external frozen source; no artifacts were copied into the repair "
                "execution root.",
                "",
                f"- Source execution: "
                f"`{external_frozen_shard['source_execution_root']}`",
                f"- Source shard: `{external_frozen_shard['source_shard_id']}`",
                f"- Source artifact set: "
                f"`{external_frozen_shard['source_artifact_set_sha256']}`",
                "",
            ]
        )
    return "\n".join(lines).encode("utf-8")


def _load_static_gcs_repair_query_ids(
    verified: object,
) -> tuple[frozenset[str], dict[str, object]]:
    """Recover the frozen r3 language-repair membership, never from responses."""

    files = getattr(verified, "files", None)
    manifest_path = Path(getattr(files, "materialization_manifest_path")).absolute()
    expected_manifest_file_sha256 = str(
        getattr(files, "expected_materialization_manifest_file_sha256")
    )
    manifest_bytes = read_stable_regular_file(
        manifest_path,
        label="Static GCS Core r3 materialization manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_file_sha256:
        raise ValueError("Static GCS Core r3 materialization manifest drifted")
    manifest = R3RepairManifest.model_validate_json(manifest_bytes, strict=True)
    if manifest.canonical_bytes() != manifest_bytes:
        raise ValueError("Static GCS Core r3 materialization manifest is not canonical")

    matching_files = tuple(
        item for item in manifest.files if item.relative_path == "repair-plan.json"
    )
    if len(matching_files) != 1:
        raise ValueError("Core r3 manifest does not uniquely bind repair-plan.json")
    descriptor = matching_files[0]
    repair_plan_path = manifest_path.parent / descriptor.relative_path
    repair_plan_bytes = read_stable_regular_file(
        repair_plan_path,
        label="Static GCS Core r3 repair plan",
        max_bytes=2 * 1024 * 1024,
    )
    if (
        len(repair_plan_bytes) != descriptor.bytes
        or sha256_bytes(repair_plan_bytes) != descriptor.sha256
    ):
        raise ValueError("Static GCS Core r3 repair plan differs from its manifest")
    repair_plan = R3RepairPlan.model_validate_json(repair_plan_bytes, strict=True)
    if repair_plan.canonical_bytes() != repair_plan_bytes:
        raise ValueError("Static GCS Core r3 repair plan is not canonical")
    if repair_plan.repair_plan_sha256 != manifest.repair_plan_sha256:
        raise ValueError("Static GCS Core r3 repair plan logical identity drifted")

    repair_query_ids = frozenset(
        query_id for batch in repair_plan.batches for query_id in batch.query_ids
    )
    if len(repair_query_ids) != manifest.replacement_query_count:
        raise ValueError("Static GCS Core r3 repair population is incomplete")
    return repair_query_ids, {
        "authority": "core_r3_hash_bound_repair_plan",
        "materialization_manifest_file_sha256": (expected_manifest_file_sha256),
        "materialization_manifest_sha256": manifest.manifest_sha256,
        "repair_plan_file_sha256": descriptor.sha256,
        "repair_plan_sha256": repair_plan.repair_plan_sha256,
        "repair_query_count": len(repair_query_ids),
    }


def _static_gcs_strata_identity(
    query: object,
    *,
    source_dataset: str,
    repair_query_ids: frozenset[str],
) -> dict[str, dict[str, object]]:
    """Build typed strata solely from verified corpus/runtime authorities."""

    query_id = str(getattr(query, "query_id"))
    capability = getattr(query, "canonical_capability")
    if not isinstance(capability, str) or capability not in GCS_CAPABILITY_ORDER:
        raise ValueError("Static GCS stratum has an unknown capability")
    if not source_dataset or source_dataset != source_dataset.strip():
        raise ValueError("Static GCS stratum has an invalid source dataset")
    style_identity: dict[str, object]
    if capability == "product.style_recommendation":
        classification = classify_style_query_v2(str(getattr(query, "text")))
        style_identity = {
            "status": "available",
            "value": classification.style_submode,
        }
    else:
        style_identity = {"status": "not_applicable", "value": None}
    return {
        "capability": {"status": "available", "value": capability},
        "source": {"status": "available", "value": source_dataset},
        "repair": {
            "status": "available",
            "value": (
                "r3_language_repaired"
                if query_id in repair_query_ids
                else "r3_carry_forward"
            ),
        },
        "boundary": {
            "status": "available",
            "value": (
                "boundary" if bool(getattr(query, "is_boundary")) else "non_boundary"
            ),
        },
        "style_submode": style_identity,
    }


def _static_gcs_strata_summary_rows(
    scores: Sequence[GCSQueryScoreV2],
    *,
    strata_by_query: Mapping[str, Mapping[str, Mapping[str, object]]],
    scope: str,
) -> list[dict[str, object]]:
    """Aggregate deterministic fixed-denominator GCS results by frozen strata."""

    score_by_query = {score.query_id: score for score in scores}
    if (
        not scores
        or len(score_by_query) != len(scores)
        or set(score_by_query) != set(strata_by_query)
    ):
        raise ValueError("Static GCS strata do not cover the scored population")
    grouped: dict[tuple[str, str, str | None], list[GCSQueryScoreV2]] = defaultdict(
        list
    )
    for query_id, score in score_by_query.items():
        identities = strata_by_query[query_id]
        if tuple(identities) != _STATIC_GCS_STRATUM_ORDER:
            raise ValueError("Static GCS strata identity schema drifted")
        for stratum in _STATIC_GCS_STRATUM_ORDER:
            identity = identities[stratum]
            if set(identity) != {"status", "value"}:
                raise ValueError("Static GCS stratum identity is not typed")
            status = identity["status"]
            value = identity["value"]
            if status == "available":
                if not isinstance(value, str) or not value:
                    raise ValueError("available Static GCS stratum lacks a value")
            elif status == "not_applicable":
                if value is not None:
                    raise ValueError(
                        "not-applicable Static GCS stratum unexpectedly has a value"
                    )
            else:
                raise ValueError("Static GCS stratum has an unknown typed status")
            grouped[
                (stratum, str(status), value if isinstance(value, str) else None)
            ].append(score)

    status_order = {"available": 0, "not_applicable": 1}
    stratum_order = {
        value: ordinal for ordinal, value in enumerate(_STATIC_GCS_STRATUM_ORDER)
    }
    rows: list[dict[str, object]] = []
    for (stratum, status, value), members in sorted(
        grouped.items(),
        key=lambda item: (
            stratum_order[item[0][0]],
            status_order[item[0][1]],
            "" if item[0][2] is None else item[0][2],
        ),
    ):
        query_count = len(members)
        success_count = sum(item.gcs for item in members)
        hard_error_count = sum(item.hard_error for item in members)
        style_status_counts = Counter(
            item.style_support_status
            for item in members
            if item.style_support_status is not None
        )
        rows.append(
            {
                "config": "llm_static",
                "scope": scope,
                "stratum": stratum,
                "value_status": status,
                "value": value,
                "query_count": query_count,
                "success_count": success_count,
                "gcs_rate": success_count / query_count,
                "hard_error_count": hard_error_count,
                "hard_error_rate": hard_error_count / query_count,
                "oracle_available_count": sum(
                    int(item.oracle_available) for item in members
                ),
                "semantic_claim_support_resolved_count": sum(
                    int(item.semantic_claim_support_resolved) for item in members
                ),
                "style_applicable_count": sum(style_status_counts.values()),
                "style_candidates_count": style_status_counts["candidates"],
                "style_no_result_count": style_status_counts["no_result"],
                "style_unsupported_count": style_status_counts["unsupported"],
                "style_unresolved_count": style_status_counts["unresolved"],
                "component_failure_counts": {
                    component: sum(
                        1 - int(getattr(item, component)) for item in members
                    )
                    for component in GCS_COMPONENTS
                },
                "fixed_denominator": True,
            }
        )
    for stratum in _STATIC_GCS_STRATUM_ORDER:
        if sum(row["query_count"] for row in rows if row["stratum"] == stratum) != len(
            scores
        ):
            raise ValueError("Static GCS stratum denominator is incomplete")
    return rows


def _static_gcs_profile_inputs(launch: object) -> object:
    verified = reconstruct_verified_portfolio_core_inputs(launch.plan)
    launch_query_ids = {
        str(query_id) for shard in launch.plan.shards for query_id in shard.query_ids
    }
    selected = tuple(query for query in verified.queries if query.split == "opt_pool")
    if (
        len(selected) != 800
        or len(launch_query_ids) != 800
        or {query.query_id for query in selected} != launch_query_ids
    ):
        raise ValueError("verified Core opt_pool differs from the Static launch")
    catalog = verified.runtime_asset_catalog()
    source_by_query = {
        query.query_id: catalog.resolve_asset_id(query.asset_id).asset.source_dataset
        for query in verified.queries
    }
    repair_query_ids, repair_binding = _load_static_gcs_repair_query_ids(verified)
    if not repair_query_ids.issubset({query.query_id for query in verified.queries}):
        raise ValueError("Core r3 repair plan refers outside the frozen population")
    return SimpleNamespace(
        profile="core",
        queries=tuple(verified.queries),
        selected_queries=selected,
        split_by_query={query.query_id: query.split for query in verified.queries},
        source_by_query=source_by_query,
        repair_query_ids=repair_query_ids,
        strata_input_bindings={
            "query_fields": {
                "authority": "frozen_query_v2",
                "query_artifact_sha256": launch.plan.query_artifact_sha256,
                "fields": ["canonical_capability", "is_boundary", "text"],
            },
            "source": {
                "authority": "verified_runtime_asset_catalog",
                "runtime_catalog_sha256": catalog.catalog_sha256,
                "runtime_catalog_assets_file_sha256": catalog.manifest.assets.sha256,
                "field": "DatasetAsset.source_dataset",
            },
            "repair": repair_binding,
        },
    )


def _analyze_static_gcs_v2(
    *,
    execution_root: Path,
    control: Mapping[str, object],
    launch: object,
    runtime_root: Path,
    batch_id: str | None,
    output_dir: Path,
) -> dict[str, object]:
    """Analyze a schema-v2 Assistant-only rollout without touching a Judge."""

    _require_static_gcs_v2_contract(control, launch.plan)
    selected_batch_ids = _select_static_gcs_batch_ids(launch.plan, batch_id)
    runtime = load_verified_portfolio_static_opt_runtime(
        runtime_root,
        expected_runtime_lock_file_sha256=str(control["runtime_lock_file_sha256"]),
    )
    if (
        control.get("runtime_lock_sha256")
        != runtime.runtime_lock.get("runtime_lock_sha256")
        or control.get("assistant_checkpoint_schema_version")
        != runtime.runtime_lock.get("assistant_checkpoint_schema_version")
        or control.get("gcs_policy_version")
        != runtime.runtime_lock.get("gcs_policy_version")
        or control.get("gcs_scorer_evidence_policy_version")
        != runtime.runtime_lock.get("gcs_scorer_evidence_policy_version")
    ):
        raise ValueError("execution control differs from Static GCS v2 runtime")
    task_spec, task_spec_source = _load_static_runtime_task_specification(runtime)
    profile_inputs = _static_gcs_profile_inputs(launch)
    private_by_id = {item.query_id: item for item in profile_inputs.queries}
    selected_shards = tuple(
        shard
        for shard in launch.plan.shards
        if shard.accepted_batch_id in selected_batch_ids
    )
    selected_members = tuple(
        sorted(
            (
                member
                for member in launch.instances
                if member.shard_id in {item.shard_id for item in selected_shards}
            ),
            key=lambda item: item.query_ordinal,
        )
    )
    expected_count = 25 * len(selected_batch_ids)
    if (
        len(selected_members) != expected_count
        or len({item.query_id for item in selected_members}) != expected_count
        or any(item.config != "llm_static" for item in selected_members)
    ):
        raise ValueError("Static GCS selection is not a complete 25-query population")
    queries = tuple(private_by_id[item.query_id] for item in selected_members)
    if any(query.split != "opt_pool" for query in queries):
        raise ValueError("Static GCS selection contains a non-opt_pool query")
    population = build_gcs_population_v2(queries)
    oracles = portfolio_gcs_oracles_v2()
    scores: list[GCSQueryScoreV2] = []
    public_rows: list[dict[str, object]] = []
    checkpoint_inventory: list[dict[str, object]] = []
    sidecar_hashes: list[str] = []
    shard_by_id = {item.shard_id: item for item in selected_shards}
    strata_by_query = {
        query.query_id: _static_gcs_strata_identity(
            query,
            source_dataset=profile_inputs.source_by_query[query.query_id],
            repair_query_ids=profile_inputs.repair_query_ids,
        )
        for query in queries
    }

    for member in selected_members:
        shard = shard_by_id.get(member.shard_id)
        if shard is None:
            raise ValueError("Static GCS member references an unselected shard")
        checkpoint_path = execution_root / member.assistant_output_relpath
        checkpoint_bytes = read_stable_regular_file(
            checkpoint_path,
            label=f"GCS v2 Assistant checkpoint {member.query_id}",
            max_bytes=16 * 1024 * 1024,
        )
        raw = json.loads(checkpoint_bytes)
        if not isinstance(raw, dict):
            raise ValueError("GCS v2 Assistant checkpoint is not an object")
        response, receipt, sidecar = _assistant_row(
            checkpoint_path,
            require_schema_v2=True,
            include_scorer_evidence=True,
        )
        if sidecar is None:  # pragma: no cover - loader fails first
            raise ValueError("schema-v2 Assistant checkpoint lacks scorer evidence")
        request = _assistant_request_from_checkpoint(raw.get("request"))
        query = private_by_id[member.query_id]
        result = _assistant_result(launch, request, response)
        if (
            raw.get("kind") != "portfolio-assistant-checkpoint"
            or raw.get("schema_version") != 2
            or raw.get("instance_sha256") != member.instance_sha256
            or raw.get("query_ordinal") != member.query_ordinal
            or request.matrix_run_id != launch.plan.matrix_run_id
            or request.config != "llm_static"
            or request.query.query_id != member.query_id
            or receipt.request_sha256 != request.request_sha256
            or sidecar.matrix_run_id != launch.plan.matrix_run_id
            or sidecar.instance_id != member.instance_sha256
            or sidecar.request_sha256 != request.request_sha256
            or sidecar.query_id != member.query_id
            or sidecar.config != "llm_static"
            or sidecar.query_artifact_sha256 != launch.plan.query_artifact_sha256
        ):
            raise ValueError("GCS v2 checkpoint identity differs from launch")
        final_path = execution_root / member.final_output_relpath
        if final_path.exists():
            raise ValueError("Static GCS rollout unexpectedly contains Final output")
        score = score_portfolio_gcs_v2(
            query,
            result,
            receipt,
            sidecar,
            task_spec,
            oracles,
            population=population,
        )
        scores.append(score)
        public_rows.append(
            {
                "query_ordinal": member.query_ordinal,
                "batch_id": shard.accepted_batch_id,
                "partition": "opt_pool",
                "strata": strata_by_query[member.query_id],
                **score.model_dump(mode="json"),
            }
        )
        checkpoint_inventory.append(
            {
                "assistant_output_relpath": member.assistant_output_relpath,
                "file_sha256": sha256_bytes(checkpoint_bytes),
                "row_sha256": raw.get("row_sha256"),
            }
        )
        sidecar_hashes.append(sidecar.evidence_sha256)
        if (
            read_stable_regular_file(
                checkpoint_path,
                label=f"GCS v2 Assistant checkpoint recheck {member.query_id}",
                max_bytes=16 * 1024 * 1024,
            )
            != checkpoint_bytes
        ):
            raise ValueError("Assistant checkpoint changed during GCS analysis")

    scope = "opt_pool" if batch_id is None else f"opt_pool:batch:{batch_id}"
    summary = summarize_gcs_v2(scores, queries, "llm_static", scope)
    strata_rows = _static_gcs_strata_summary_rows(
        scores,
        strata_by_query=strata_by_query,
        scope=scope,
    )
    bootstrap, system_gate = _static_not_applicable_artifacts(
        scope=scope,
        population_mapping_sha256=summary.population_mapping_sha256,
        query_count=summary.query_count,
    )
    summary_payload = {
        "schema_version": 2,
        "kind": "portfolio-gcs-v2-summary",
        "status": "completed_read_only",
        "selection": scope,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "scorer_evidence_policy_version": GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
        "scorer_evidence_schema_version": 2,
        "population": population.model_dump(mode="json"),
        "config_summaries": [summary.model_dump(mode="json")],
        "model_calls_performed": 0,
    }
    artifact_bytes = {
        "gcs-rows.jsonl": canonical_jsonl_bytes(tuple(public_rows)),
        "gcs-summary.json": canonical_json_bytes(summary_payload),
        "gcs-config-summary.csv": _csv_bytes([_gcs_config_csv_row(summary)]),
        "gcs-capability-summary.csv": _csv_bytes(_gcs_capability_csv_rows(summary)),
        "gcs-strata-summary.csv": _csv_bytes(strata_rows),
        "gcs-paired-bootstrap.json": canonical_json_bytes(bootstrap),
        "gcs-system-gate.json": canonical_json_bytes(system_gate),
    }
    source_paths = {
        "scorer_adapter": (
            SOURCE_ROOT / "skillchain/evaluation/portfolio_gcs_evidence.py"
        ),
        "gcs_scorer": SOURCE_ROOT / "skillchain/evaluation/portfolio_gcs.py",
        "analyzer": Path(__file__),
    }
    source_sha256s = {
        name: sha256_bytes(
            read_stable_regular_file(path, label=f"GCS v2 {name} source")
        )
        for name, path in source_paths.items()
    }
    if source_sha256s["scorer_adapter"] != runtime.runtime_lock.get(
        "portfolio_gcs_evidence_file_sha256"
    ) or source_sha256s["gcs_scorer"] != runtime.runtime_lock.get(
        "portfolio_gcs_file_sha256"
    ):
        raise ValueError("Static runtime GCS source binding drifted")
    bindings = {
        "matrix_run_id": launch.plan.matrix_run_id,
        "launch_plan_sha256": control["launch_plan_sha256"],
        "launch_plan_file_sha256": control["launch_plan_file_sha256"],
        "runtime_lock_sha256": control["runtime_lock_sha256"],
        "runtime_lock_file_sha256": control["runtime_lock_file_sha256"],
        "static_bank_sha256": runtime.bank.bank_sha256,
        "static_bank_file_sha256": runtime.bank_file_sha256,
        "query_artifact_sha256": launch.plan.query_artifact_sha256,
        "task_specification": task_spec_source,
        "gcs_policy_version": GCS_V2_POLICY_VERSION,
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "scorer_evidence_policy_version": GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
        "scorer_evidence_schema_version": 2,
        "population_mapping_sha256": summary.population_mapping_sha256,
        "source_file_sha256s": source_sha256s,
        "strata_inputs": {
            **profile_inputs.strata_input_bindings,
            "style_submode": {
                "authority": "frozen_query_text_plus_runtime_style_classifier",
                "classifier_source_file_sha256": source_sha256s["scorer_adapter"],
                "non_style_status": "not_applicable",
            },
        },
        "checkpoint_set_sha256": sha256_bytes(
            canonical_json_bytes(checkpoint_inventory)
        ),
        "sidecar_set_sha256": sha256_bytes(canonical_json_bytes(sidecar_hashes)),
    }
    unsigned = {
        "schema_version": 2,
        "kind": "portfolio-static-opt-gcs-v2-analysis",
        "track": "portfolio",
        "formal_eligible": False,
        "status": "completed_read_only",
        "execution_scope": _STATIC_OPT_EXECUTION_SCOPE,
        "selection": scope,
        "selected_batch_ids": list(selected_batch_ids),
        "selected_query_count": len(queries),
        "row_count": len(scores),
        "config_order": ["llm_static"],
        "capabilities": list(GCS_CAPABILITY_ORDER),
        "components": list(GCS_COMPONENTS),
        "metric_contract": {
            "primary": GCS_V2_POLICY_VERSION,
            "fixed_denominator": True,
            "terminal_tool_or_model_failure": "row_zero",
            "checkpoint_integrity_failure": "fatal",
            "legacy_j_routing_adherence": "diagnostic_not_run_not_blocking",
            "pairwise_bootstrap": "not_applicable_single_config",
            "system_gain_gate": "not_applicable_single_config",
        },
        "bindings": bindings,
        "config_summary": [summary.model_dump(mode="json")],
        "output_artifacts": {
            name: {"file_sha256": sha256_bytes(content), "bytes": len(content)}
            for name, content in sorted(artifact_bytes.items())
        },
        "row_set_sha256": sha256_bytes(canonical_json_bytes(public_rows)),
        "model_calls_performed": 0,
    }
    analysis = {
        **unsigned,
        "analysis_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    output_dir = output_dir.absolute()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for name, content in artifact_bytes.items():
            atomic_create_file(staging / name, content)
        atomic_create_file(staging / "analysis.json", canonical_json_bytes(analysis))
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis


def analyze_portfolio_matrix(
    execution_root: Path,
    *,
    batch_id: str | None,
    output_dir: Path,
) -> dict[str, object]:
    execution_root = execution_root.absolute()
    control = _load_control(execution_root)
    launch_root = _repository_path(str(control["launch_root"])).absolute()
    runtime_root = _repository_path(str(control["runtime_root"])).absolute()
    _require_external_output(
        output_dir,
        execution_root=execution_root,
        launch_root=launch_root,
        runtime_root=runtime_root,
    )
    launch = load_portfolio_launch_package(
        launch_root,
        expected_plan_file_sha256=str(control["launch_plan_file_sha256"]),
    )
    if getattr(launch.plan, "execution_mode", None) == _STATIC_OPT_EXECUTION_SCOPE:
        return _analyze_static_gcs_v2(
            execution_root=execution_root,
            control=control,
            launch=launch,
            runtime_root=runtime_root,
            batch_id=batch_id,
            output_dir=output_dir,
        )
    selected_batch_ids = _select_batch_ids(launch.plan, batch_id)

    runtime = load_verified_portfolio_treatment_runtime(
        runtime_root,
        expected_runtime_lock_file_sha256=str(control["runtime_lock_file_sha256"]),
    )
    manifest = runtime.chain.manifest
    profile_inputs = _load_launch_profile_inputs(launch)
    partition_layout = _build_partition_layout(
        plan=launch.plan,
        manifest=manifest,
        profile_inputs=profile_inputs,
    )
    private_by_id = {item.query_id: item for item in profile_inputs.queries}

    before_audits: dict[str, dict[str, object]] = {}
    for selected_batch in selected_batch_ids:
        audit = audit_portfolio_batch(
            execution_root,
            batch_id=selected_batch,
            _profile_inputs=profile_inputs,
        )
        expected_split = (
            _core_batch_split(
                launch.plan,
                batch_id=selected_batch,
                layout=partition_layout,
            )
            if partition_layout.profile == "core"
            else None
        )
        _require_usable_audit(
            audit,
            batch_id=selected_batch,
            expected_profile=("core" if partition_layout.profile == "core" else None),
            expected_split=expected_split,
        )
        before_audits[selected_batch] = audit

    external_sources_by_shard, external_frozen_shard = _external_analysis_sources(
        control=control,
        launch=launch,
        selected_batch_ids=selected_batch_ids,
        execution_root=execution_root,
    )
    for selected_batch, audit in before_audits.items():
        bindings = audit.get("bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError(f"batch audit lacks bindings: {selected_batch}")
        audit_external = bindings.get("external_frozen_shard")
        if audit_external != external_frozen_shard:
            raise ValueError(
                "analyzer external source differs from the strict batch audit: "
                f"{selected_batch}"
            )

    output_banks: dict[str, object] = dict(runtime.chain.output_banks)
    adherence_banks = _adherence_contract_banks(runtime.chain)
    rows = _extract_rows(
        execution_root=execution_root,
        launch=launch,
        selected_batch_ids=selected_batch_ids,
        private_by_id=private_by_id,
        partition_by_query=partition_layout.partition_by_query,
        output_banks=output_banks,
        adherence_banks=adherence_banks,
        external_sources_by_shard=external_sources_by_shard,
    )
    _validate_rectangular(rows)

    audit_records: list[dict[str, object]] = []
    for selected_batch in selected_batch_ids:
        after = audit_portfolio_batch(
            execution_root,
            batch_id=selected_batch,
            _profile_inputs=profile_inputs,
        )
        _require_usable_audit(
            after,
            batch_id=selected_batch,
            expected_profile=("core" if partition_layout.profile == "core" else None),
            expected_split=(
                _core_batch_split(
                    launch.plan,
                    batch_id=selected_batch,
                    layout=partition_layout,
                )
                if partition_layout.profile == "core"
                else None
            ),
        )
        if after != before_audits[selected_batch]:
            raise ValueError(
                f"batch artifacts changed during analysis: {selected_batch}"
            )
        audit_records.append(
            {
                "batch_id": selected_batch,
                "status": after["status"],
                "audit_sha256": after["audit_sha256"],
                "artifact_set_sha256": after["artifact_set_sha256"],
                "review_flags": after["review_flags"],
            }
        )
    replayed_rows = _extract_rows(
        execution_root=execution_root,
        launch=launch,
        selected_batch_ids=selected_batch_ids,
        private_by_id=private_by_id,
        partition_by_query=partition_layout.partition_by_query,
        output_banks=output_banks,
        adherence_banks=adherence_banks,
        external_sources_by_shard=external_sources_by_shard,
    )
    if replayed_rows != rows:
        raise ValueError("row projection changed across the validated snapshot")
    replayed_profile_inputs = _load_launch_profile_inputs(launch)
    if replayed_profile_inputs != profile_inputs:
        raise ValueError("launch private inputs changed during matrix analysis")

    full_matrix = batch_id is None
    scopes = _scope_rows(
        rows,
        full_matrix=full_matrix,
        batch_id=batch_id,
        layout=partition_layout,
    )
    config_summaries: list[dict[str, object]] = []
    capability_summaries: list[dict[str, object]] = []
    paired_summaries: list[dict[str, object]] = []
    bank_sha256s: dict[str, str | None] = {
        "noskill": None,
        **{config: bank.bank_sha256 for config, bank in output_banks.items()},
    }
    for scope, selected_rows in scopes.items():
        _validate_rectangular(selected_rows)
        config_summaries.extend(_config_summary(selected_rows, scope))
        capability_summaries.extend(_capability_summary(selected_rows, scope))
        paired_summaries.extend(
            _paired_summaries(
                selected_rows,
                scope=scope,
                bank_sha256s=bank_sha256s,
            )
        )

    no_op_contrasts = [
        {
            "baseline_config": baseline,
            "treatment_config": treatment,
            "bank_sha256": bank_sha256s[baseline],
            "treatment_attribution_allowed": False,
        }
        for baseline, treatment in combinations(MAIN_CONFIG_ORDER, 2)
        if bank_sha256s.get(baseline) is not None
        and bank_sha256s.get(baseline) == bank_sha256s.get(treatment)
    ]
    coverage = _partition_coverage(rows, layout=partition_layout)
    selection = partition_layout.aggregate_name if full_matrix else f"batch:{batch_id}"
    report_bytes = _report(
        selection=selection,
        partition_coverage=coverage,
        config_summaries=config_summaries,
        paired_summaries=paired_summaries,
        no_op_contrasts=no_op_contrasts,
        audit_records=audit_records,
        external_frozen_shard=external_frozen_shard,
        partition_use_policy=partition_layout.use_policy,
    )
    artifact_bytes = {
        "rows.csv": _csv_bytes([_row_csv_projection(item) for item in rows]),
        "config-summary.csv": _csv_bytes(
            [_summary_csv_projection(item) for item in config_summaries]
        ),
        "capability-summary.csv": _csv_bytes(
            [_summary_csv_projection(item) for item in capability_summaries]
        ),
        "paired-deltas.csv": _csv_bytes(
            [_summary_csv_projection(item) for item in paired_summaries]
        ),
        "report.md": report_bytes,
    }
    stage_decisions = [
        {
            "config": record.config,
            "stage": record.stage,
            "decision": record.decision,
            "parent_bank_sha256": record.parent_bank_sha256,
            "candidate_bank_sha256": record.candidate_bank_sha256,
            "output_bank_sha256": record.output_bank_sha256,
        }
        for record in manifest.records
    ]
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-matrix-analysis",
        "track": "portfolio",
        "formal_eligible": False,
        "status": "completed_read_only",
        "selection": selection,
        **(
            {
                "dataset_profile": "core",
                "selected_splits": list(partition_layout.partition_order),
            }
            if partition_layout.profile == "core"
            else {}
        ),
        "selected_batch_ids": list(selected_batch_ids),
        "selected_query_count": len({item.query_id for item in rows}),
        "row_count": len(rows),
        "config_order": list(MAIN_CONFIG_ORDER),
        "capabilities": list(CAPABILITIES),
        "metric_contract": {
            "j_project": "mean_over_all_selected_rows_fixed_denominator",
            "routing": "six_class_macro_f1_missing_or_invalid_prediction_is_wrong",
            "structural_adherence": (
                "selected_output_bank_exact_skill_slug_structural_contract"
            ),
            "gate_aligned_adherence": (
                "static_own_or_frozen_stage_parent_contract_bank_slug_agnostic"
            ),
            "paired_delta": "treatment_minus_baseline_same_query",
            "uncertainty": "leakage_cluster_bootstrap_fixed_seed",
        },
        "partition_use_policy": dict(partition_layout.use_policy),
        "partition_coverage": coverage,
        "bindings": {
            "matrix_run_id": launch.plan.matrix_run_id,
            "launch_plan_sha256": control["launch_plan_sha256"],
            "runtime_lock_sha256": control["runtime_lock_sha256"],
            "treatment_chain_sha256": manifest.chain_sha256,
            "rubric_file_sha256": control["rubric_file_sha256"],
            "rubric_content_sha256": control["rubric_content_sha256"],
            "bank_sha256s": bank_sha256s,
            **(
                {"external_frozen_shard": external_frozen_shard}
                if external_frozen_shard is not None
                else {}
            ),
        },
        "stage_decisions": stage_decisions,
        "no_op_contrasts": no_op_contrasts,
        "batch_audits": audit_records,
        "config_summary": config_summaries,
        "capability_summary": capability_summaries,
        "paired_deltas": paired_summaries,
        "output_artifacts": {
            name: {"file_sha256": sha256_bytes(content), "bytes": len(content)}
            for name, content in sorted(artifact_bytes.items())
        },
        "row_set_sha256": sha256_bytes(
            canonical_json_bytes(_json_projection([asdict(item) for item in rows]))
        ),
        "model_calls_performed": 0,
    }
    unsigned = _json_projection(unsigned)
    if not isinstance(unsigned, dict):  # pragma: no cover - construction invariant
        raise AssertionError("analysis payload projection must remain an object")
    analysis = {
        **unsigned,
        "analysis_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    analysis_bytes = canonical_json_bytes(analysis)

    output_dir = output_dir.absolute()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for name, content in artifact_bytes.items():
            atomic_create_file(staging / name, content)
        atomic_create_file(staging / "analysis.json", analysis_bytes)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        analysis = analyze_portfolio_matrix(
            arguments.execution_root,
            batch_id=arguments.batch_id,
            output_dir=arguments.output_dir,
        )
    except Exception as error:
        print(f"analyze-portfolio-matrix: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(analysis))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
