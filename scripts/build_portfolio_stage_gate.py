"""Recompute one S1/S2/S3 Pareto gate from two treatment smoke runs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Literal

from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_FINAL_RUBRIC_FILE_SHA256,
    PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION,
    PORTFOLIO_STAGE_GATE_QUERY_COUNT,
    PortfolioStageGateRowProvenance,
    PortfolioStageGateResultSet,
    PortfolioTreatmentConfig,
    build_portfolio_stage_gate_result_set,
    build_portfolio_stage_gate_report,
    calculate_portfolio_skill_adherence,
)
from skillchain.evaluation.portfolio_attribution import (
    PORTFOLIO_OPTIMIZATION_CAPABILITY_COUNTS,
    _load_assistant,
    _load_final,
    _load_results,
)
from skillchain.runners.assistant import SharedStage2RouteArtifact
from skillchain.static_authoring import StaticBankArtifact
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


GateConfig = Literal["s1", "s1s2", "full"]
_PARENT_CONFIG: dict[GateConfig, str] = {
    "s1": "llm_static",
    "s1s2": "s1",
    "full": "s1s2",
}
_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class GateObservation:
    query_id: str
    selected_capability: str | None
    route_correct: bool
    j_project: float
    skill_adherence: float
    assistant_hard_error: bool
    evaluator_anomaly: bool
    source_result_sha256: str
    shared_route_artifact_sha256: str | None


@dataclass(frozen=True)
class LoadedSmoke:
    source_config: PortfolioTreatmentConfig
    bank_sha256: str
    adherence_contract_bank_sha256: str
    evaluation_query_ids: tuple[str, ...]
    rubric_file_sha256: str
    shared_route_artifact_sha256s: tuple[str | None, ...]
    source_summary_file_sha256: str
    source_summary_sha256: str
    source_results_file_sha256: str
    source_result_sha256s: tuple[str, ...]
    observations: tuple[GateObservation, ...]


def _self_hash(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    observed = payload.pop(field, None)
    expected = sha256_bytes(canonical_json_bytes(payload))
    if observed != expected:
        raise ValueError(f"{field} mismatch")
    return expected


def _load_bank(path: Path, label: str) -> StaticBankArtifact:
    content = read_stable_regular_file(path, label=label, max_bytes=_MAX_BYTES)
    bank = StaticBankArtifact.model_validate_json(content, strict=True)
    if bank.canonical_bytes() != content:
        raise ValueError(f"{label} is not canonical")
    return bank


def _shared_route_projection(
    content: bytes,
    *,
    query_id: str,
) -> tuple[str | None, str]:
    """Load current routes and the immutable v1 routes used by existing smokes."""

    try:
        route = SharedStage2RouteArtifact.model_validate_json(content, strict=True)
    except Exception:
        raw = parse_canonical_json(content, label=f"legacy shared route {query_id}")
        expected_fields = {
            "schema_version",
            "policy_version",
            "status",
            "matrix_run_id",
            "query_id",
            "query_ordinal",
            "query_sha256",
            "public_input_sha256",
            "backbone_identity_sha256",
            "budget_sha256",
            "registry_sha256",
            "registry_runtime_sha256",
            "capability_ids",
            "description_set_sha256",
            "selected_capability",
            "failure_subtype",
            "route_call",
            "artifact_sha256",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) != expected_fields
            or raw.get("schema_version") != 1
            or raw.get("policy_version") != "shared-stage2-route-v1"
            or raw.get("query_id") != query_id
        ):
            raise ValueError("legacy shared-route artifact shape is invalid")
        unsigned = dict(raw)
        artifact_sha256 = unsigned.pop("artifact_sha256")
        if artifact_sha256 != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ValueError("legacy shared-route artifact digest mismatch")
        capabilities = raw.get("capability_ids")
        selected = raw.get("selected_capability")
        status = raw.get("status")
        failure_subtype = raw.get("failure_subtype")
        if (
            not isinstance(capabilities, list)
            or capabilities != sorted(set(capabilities))
            or (
                status == "selected"
                and (selected not in capabilities or failure_subtype is not None)
            )
            or (
                status == "terminal_route_error"
                and (selected is not None or not isinstance(failure_subtype, str))
            )
            or status not in {"selected", "terminal_route_error"}
            or not isinstance(artifact_sha256, str)
        ):
            raise ValueError("legacy shared-route outcome is invalid")
        return selected if isinstance(selected, str) else None, artifact_sha256
    if (
        canonical_json_bytes(route.model_dump(mode="json")) != content
        or route.query_id != query_id
    ):
        raise ValueError("shared-route artifact projection mismatch")
    return route.selected_capability, route.artifact_sha256


def _load_smoke(
    root: Path,
    *,
    expected_config: PortfolioTreatmentConfig,
    expected_bank: StaticBankArtifact,
    adherence_contract_bank: StaticBankArtifact,
    include_loaded: bool = False,
) -> (
    tuple[PortfolioStageGateResultSet, bytes]
    | tuple[PortfolioStageGateResultSet, bytes, LoadedSmoke]
):
    expected_bank_sha256 = expected_bank.bank_sha256
    summary_bytes = read_stable_regular_file(
        root / "summary.json",
        label=f"{expected_config} smoke summary",
        max_bytes=_MAX_BYTES,
    )
    summary = parse_canonical_json(
        summary_bytes,
        label=f"{expected_config} smoke summary",
    )
    if not isinstance(summary, dict):
        raise ValueError("smoke summary must contain an object")
    _self_hash(summary, "summary_sha256")
    if (
        summary.get("schema_version") != 1
        or summary.get("kind") != "portfolio-treatment-development-smoke-summary"
        or summary.get("policy_version") != PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION
        or summary.get("status") != "complete"
        or summary.get("config") != expected_config
        or summary.get("target_bank_sha256") != expected_bank_sha256
        or summary.get("query_count_completed") != PORTFOLIO_STAGE_GATE_QUERY_COUNT
        or summary.get("results_file") != "results.jsonl"
    ):
        raise ValueError(f"{expected_config} smoke summary binding mismatch")

    expected_results_sha = summary.get("results_file_sha256")
    if not isinstance(expected_results_sha, str):
        raise ValueError(f"{expected_config} smoke lacks a result digest")
    results_bytes, rows = _load_results(
        root / "results.jsonl",
        expected_file_sha256=expected_results_sha,
    )
    query_ids = tuple(row.query_id for row in rows)
    capabilities = tuple(row.canonical_capability for row in rows)
    if (
        len(rows) != PORTFOLIO_STAGE_GATE_QUERY_COUNT
        or len(set(query_ids)) != PORTFOLIO_STAGE_GATE_QUERY_COUNT
        or list(query_ids) != summary.get("query_ids_completed")
        or list(query_ids) != summary.get("query_ids_requested")
        or list(capabilities) != summary.get("capabilities_completed")
        or dict(Counter(capabilities)) != PORTFOLIO_OPTIMIZATION_CAPABILITY_COUNTS
    ):
        raise ValueError(f"{expected_config} smoke denominator mismatch")

    formal_adherence_by_query: dict[str, float] = {}
    for row in rows:
        if (
            row.config != expected_config
            or row.target_bank_sha256 != expected_bank_sha256
        ):
            raise ValueError(f"{expected_config} smoke row binding mismatch")
        assistant, _ = _load_assistant(
            root,
            row,
            expected_config=expected_config,
            expected_bank_sha256=expected_bank_sha256,
        )
        _load_final(root, row)
        target_adherence = calculate_portfolio_skill_adherence(
            bank=expected_bank,
            selected_capability=assistant.response.selected_capability,
            skill_slug=assistant.response.skill_slug,
            response_text=assistant.response.response_text,
            visible_cards=assistant.response.visible_cards,
            hard_error=row.hard_error,
        )
        if not math.isclose(row.skill_adherence, target_adherence, abs_tol=1e-12):
            raise ValueError(
                f"{expected_config} smoke Skill-adherence projection drifted"
            )
        formal_adherence_by_query[row.query_id] = (
            calculate_portfolio_skill_adherence(
                bank=adherence_contract_bank,
                selected_capability=assistant.response.selected_capability,
                skill_slug=assistant.response.skill_slug,
                response_text=assistant.response.response_text,
                visible_cards=assistant.response.visible_cards,
                hard_error=row.assistant_error_code is not None,
                require_skill_slug_match=False,
            )
        )

    route_accuracy = sum(row.route_correct for row in rows) / len(rows)
    mean_j = sum(row.j_project for row in rows) / len(rows)
    legacy_combined_hard_errors = sum(row.hard_error for row in rows)
    assistant_hard_errors = sum(row.assistant_error_code is not None for row in rows)
    evaluator_anomalies = sum(
        row.assistant_error_code is None and row.final_status != "scored"
        for row in rows
    )
    target_mean_adherence = sum(row.skill_adherence for row in rows) / len(rows)
    mean_skill_adherence = sum(formal_adherence_by_query.values()) / len(
        formal_adherence_by_query
    )
    if (
        not math.isclose(
            route_accuracy,
            float(summary.get("route_accuracy", -1.0)),
            abs_tol=1e-12,
        )
        or not math.isclose(
            mean_j,
            float(summary.get("mean_j_project", -1.0)),
            abs_tol=1e-12,
        )
        or legacy_combined_hard_errors != summary.get("hard_error_count")
        or not math.isclose(
            target_mean_adherence,
            float(summary.get("mean_skill_adherence", -1.0)),
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{expected_config} smoke summary metrics drifted")
    if "assistant_hard_error_count" in summary and (
        summary.get("assistant_hard_error_count") != assistant_hard_errors
        or summary.get("evaluator_anomaly_count") != evaluator_anomalies
    ):
        raise ValueError(f"{expected_config} smoke split error metrics drifted")

    sorted_rows = tuple(sorted(rows, key=lambda item: item.query_id))
    shared_route_sha256s: list[str | None] = []
    for row in sorted_rows:
        route_file = row.shared_route_file
        route_file_sha256 = row.shared_route_file_sha256
        if expected_config not in {"s1s2", "full"}:
            if route_file is not None or route_file_sha256 is not None:
                raise ValueError(
                    f"{expected_config} smoke unexpectedly claims a shared route"
                )
            shared_route_sha256s.append(None)
            continue
        expected_route_file = f"shared-routes/{row.query_id}.json"
        if route_file != expected_route_file or not isinstance(route_file_sha256, str):
            raise ValueError(f"{expected_config} smoke shared-route binding mismatch")
        route_bytes = read_stable_regular_file(
            root / expected_route_file,
            label=f"{expected_config} shared route {row.query_id}",
            max_bytes=_MAX_BYTES,
        )
        if sha256_bytes(route_bytes) != route_file_sha256:
            raise ValueError(
                f"{expected_config} smoke shared-route file digest mismatch"
            )
        try:
            selected_capability, route_artifact_sha256 = (
                _shared_route_projection(route_bytes, query_id=row.query_id)
            )
        except Exception as error:
            raise ValueError(
                f"{expected_config} smoke shared-route artifact is invalid"
            ) from error
        if selected_capability != row.selected_capability:
            raise ValueError(
                f"{expected_config} smoke shared-route projection mismatch"
            )
        shared_route_sha256s.append(route_artifact_sha256)

    rubric_file_sha256 = summary.get("rubric_file_sha256")
    if rubric_file_sha256 != PORTFOLIO_FINAL_RUBRIC_FILE_SHA256:
        raise ValueError(f"{expected_config} smoke lacks a frozen rubric binding")
    observations = tuple(
        GateObservation(
            query_id=row.query_id,
            selected_capability=row.selected_capability,
            route_correct=row.route_correct,
            j_project=row.j_project,
            skill_adherence=formal_adherence_by_query[row.query_id],
            assistant_hard_error=row.assistant_error_code is not None,
            evaluator_anomaly=(
                row.assistant_error_code is None and row.final_status != "scored"
            ),
            source_result_sha256=row.result_sha256,
            shared_route_artifact_sha256=shared_route_sha256,
        )
        for row, shared_route_sha256 in zip(
            sorted_rows,
            shared_route_sha256s,
            strict=True,
        )
    )
    loaded = LoadedSmoke(
        source_config=expected_config,
        bank_sha256=expected_bank_sha256,
        adherence_contract_bank_sha256=adherence_contract_bank.bank_sha256,
        evaluation_query_ids=tuple(item.query_id for item in observations),
        rubric_file_sha256=rubric_file_sha256,
        shared_route_artifact_sha256s=tuple(shared_route_sha256s),
        source_summary_file_sha256=sha256_bytes(summary_bytes),
        source_summary_sha256=summary["summary_sha256"],
        source_results_file_sha256=sha256_bytes(results_bytes),
        source_result_sha256s=tuple(item.source_result_sha256 for item in observations),
        observations=observations,
    )
    result = _build_result_set(loaded, observations)
    if include_loaded:
        return result, result.canonical_bytes(), loaded
    return result, result.canonical_bytes()


def _build_result_set(
    loaded: LoadedSmoke,
    observations: tuple[GateObservation, ...],
    *,
    row_provenance: tuple[PortfolioStageGateRowProvenance, ...] | None = None,
) -> PortfolioStageGateResultSet:
    if tuple(item.query_id for item in observations) != loaded.evaluation_query_ids:
        raise ValueError("gate observations are not query-aligned")
    if row_provenance is None:
        row_provenance = tuple(
            PortfolioStageGateRowProvenance(
                query_id=item.query_id,
                metric_source="candidate",
                reuse_basis="direct_observation",
                candidate_source_result_sha256=item.source_result_sha256,
                reused_parent_source_result_sha256=None,
                selected_source_result_sha256=item.source_result_sha256,
            )
            for item in observations
        )
    return build_portfolio_stage_gate_result_set(
        source_config=loaded.source_config,
        bank_sha256=loaded.bank_sha256,
        adherence_contract_bank_sha256=loaded.adherence_contract_bank_sha256,
        evaluation_query_ids=loaded.evaluation_query_ids,
        route_accuracy=sum(item.route_correct for item in observations)
        / len(observations),
        mean_j=sum(item.j_project for item in observations) / len(observations),
        mean_skill_adherence=sum(item.skill_adherence for item in observations)
        / len(observations),
        hard_error_count=sum(item.assistant_hard_error for item in observations),
        evaluator_anomaly_count=sum(item.evaluator_anomaly for item in observations),
        causal_reuse_unaffected=any(
            item.metric_source == "reused_parent" for item in row_provenance
        ),
        rubric_file_sha256=loaded.rubric_file_sha256,
        shared_route_artifact_sha256s=loaded.shared_route_artifact_sha256s,
        source_summary_file_sha256=loaded.source_summary_file_sha256,
        source_summary_sha256=loaded.source_summary_sha256,
        source_results_file_sha256=loaded.source_results_file_sha256,
        source_result_sha256s=loaded.source_result_sha256s,
        row_provenance=row_provenance,
    )


def _selected_skill_body_and_operators(
    bank: StaticBankArtifact,
    selected_capability: str | None,
) -> tuple[str, tuple[str, ...]] | None:
    if selected_capability is None:
        return None
    matches = tuple(
        item
        for item in bank.skills
        if item.capability_id == selected_capability
    )
    if len(matches) != 1:
        raise ValueError(
            "selected capability does not resolve to exactly one Bank Skill"
        )
    return matches[0].body, tuple(matches[0].operators)


def _causal_candidate_result(
    *,
    config: GateConfig,
    parent_loaded: LoadedSmoke,
    candidate_loaded: LoadedSmoke,
    parent_bank: StaticBankArtifact,
    candidate_bank: StaticBankArtifact,
) -> PortfolioStageGateResultSet:
    if config == "s1":
        raise ValueError("causal unaffected-row reuse is only defined for S2/S3")
    selected: list[GateObservation] = []
    provenance: list[PortfolioStageGateRowProvenance] = []
    for parent, candidate in zip(
        parent_loaded.observations,
        candidate_loaded.observations,
        strict=True,
    ):
        if parent.query_id != candidate.query_id:
            raise ValueError("causal gate rows are not query-paired")
        if config == "s1s2":
            unaffected = (
                parent.selected_capability == candidate.selected_capability
            )
            reuse_basis = "same_selected_route"
        else:
            if parent.selected_capability != candidate.selected_capability:
                raise ValueError("S3 common-route selected capability drifted")
            unaffected = _selected_skill_body_and_operators(
                parent_bank,
                parent.selected_capability,
            ) == _selected_skill_body_and_operators(
                candidate_bank,
                candidate.selected_capability,
            )
            reuse_basis = "same_selected_skill_body_and_operators"
        if unaffected:
            selected.append(parent)
            provenance.append(
                PortfolioStageGateRowProvenance(
                    query_id=candidate.query_id,
                    metric_source="reused_parent",
                    reuse_basis=reuse_basis,
                    candidate_source_result_sha256=candidate.source_result_sha256,
                    reused_parent_source_result_sha256=(
                        parent.source_result_sha256
                    ),
                    selected_source_result_sha256=parent.source_result_sha256,
                )
            )
        else:
            selected.append(candidate)
            provenance.append(
                PortfolioStageGateRowProvenance(
                    query_id=candidate.query_id,
                    metric_source="candidate",
                    reuse_basis="direct_observation",
                    candidate_source_result_sha256=candidate.source_result_sha256,
                    reused_parent_source_result_sha256=None,
                    selected_source_result_sha256=candidate.source_result_sha256,
                )
            )
    return _build_result_set(
        candidate_loaded,
        tuple(selected),
        row_provenance=tuple(provenance),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("s1", "s1s2", "full"), required=True)
    parser.add_argument("--parent-bank", type=Path, required=True)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--parent-smoke-root", type=Path, required=True)
    parser.add_argument("--candidate-smoke-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--causal-reuse-unaffected",
        action="store_true",
        help=(
            "Reuse paired parent metrics for causally unaffected S2/S3 rows; "
            "S2 uses unchanged selected route and S3 uses unchanged selected "
            "Skill Body/operators under the exact common route."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config: GateConfig = arguments.config
    if arguments.output_dir.exists():
        print("build-portfolio-stage-gate: output directory exists", file=sys.stderr)
        return 2
    arguments.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_dir.name}.",
            dir=arguments.output_dir.parent,
        )
    )
    try:
        parent_bank = _load_bank(arguments.parent_bank, "parent Bank")
        candidate_bank = _load_bank(arguments.candidate_bank, "candidate Bank")
        parent, parent_bytes, parent_loaded = _load_smoke(
            arguments.parent_smoke_root,
            expected_config=_PARENT_CONFIG[config],
            expected_bank=parent_bank,
            adherence_contract_bank=parent_bank,
            include_loaded=True,
        )
        candidate, candidate_bytes, candidate_loaded = _load_smoke(
            arguments.candidate_smoke_root,
            expected_config=config,
            expected_bank=candidate_bank,
            adherence_contract_bank=parent_bank,
            include_loaded=True,
        )
        query_ids = parent.evaluation_query_ids
        if query_ids != candidate.evaluation_query_ids:
            raise ValueError("parent and candidate gate query sets differ")
        if parent.rubric_file_sha256 != candidate.rubric_file_sha256:
            raise ValueError("parent and candidate gate rubrics differ")
        paired_routes = None
        if config == "full":
            if (
                parent.shared_route_artifact_sha256s
                != candidate.shared_route_artifact_sha256s
            ):
                raise ValueError(
                    "S3 parent and candidate must reuse exact per-query routes"
                )
            paired_routes = tuple(
                item
                for item in parent.shared_route_artifact_sha256s
                if item is not None
            )
        if arguments.causal_reuse_unaffected:
            candidate = _causal_candidate_result(
                config=config,
                parent_loaded=parent_loaded,
                candidate_loaded=candidate_loaded,
                parent_bank=parent_bank,
                candidate_bank=candidate_bank,
            )
            candidate_bytes = candidate.canonical_bytes()
        nonregressed = (
            candidate.hard_error_count <= parent.hard_error_count
            and candidate.route_accuracy >= parent.route_accuracy
            and candidate.mean_j >= parent.mean_j
            and candidate.mean_skill_adherence >= parent.mean_skill_adherence
        )
        improved = (
            candidate.hard_error_count < parent.hard_error_count
            or candidate.route_accuracy > parent.route_accuracy
            or candidate.mean_j > parent.mean_j
            or candidate.mean_skill_adherence > parent.mean_skill_adherence
        )
        if (
            parent.evaluator_anomaly_count > 0
            or candidate.evaluator_anomaly_count > 0
        ):
            decision = "inconclusive"
        else:
            decision = "accepted" if nonregressed and improved else "rolled_back"
        parent_relative = f"gates/{config}/parent-result.json"
        candidate_relative = f"gates/{config}/candidate-result.json"
        report = build_portfolio_stage_gate_report(
            config=config,
            parent_bank_sha256=parent_bank.bank_sha256,
            candidate_bank_sha256=candidate_bank.bank_sha256,
            adherence_contract_bank_sha256=parent_bank.bank_sha256,
            evaluation_query_ids=query_ids,
            rubric_file_sha256=parent.rubric_file_sha256,
            paired_shared_route_artifact_sha256s=paired_routes,
            parent_route_accuracy=parent.route_accuracy,
            candidate_route_accuracy=candidate.route_accuracy,
            parent_mean_j=parent.mean_j,
            candidate_mean_j=candidate.mean_j,
            parent_mean_skill_adherence=parent.mean_skill_adherence,
            candidate_mean_skill_adherence=candidate.mean_skill_adherence,
            parent_hard_error_count=parent.hard_error_count,
            candidate_hard_error_count=candidate.hard_error_count,
            parent_evaluator_anomaly_count=parent.evaluator_anomaly_count,
            candidate_evaluator_anomaly_count=candidate.evaluator_anomaly_count,
            causal_reuse_unaffected=candidate.causal_reuse_unaffected,
            decision=decision,
            parent_result_file=parent_relative,
            parent_result_file_sha256=sha256_bytes(parent_bytes),
            candidate_result_file=candidate_relative,
            candidate_result_file_sha256=sha256_bytes(candidate_bytes),
        )
        atomic_create_file(staging / "parent-result.json", parent_bytes)
        atomic_create_file(staging / "candidate-result.json", candidate_bytes)
        atomic_create_file(staging / "gate-report.json", report.canonical_bytes())
        staging.replace(arguments.output_dir)
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"build-portfolio-stage-gate: {error}", file=sys.stderr)
        return 2
    print(arguments.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
